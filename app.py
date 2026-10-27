from flask import Flask, render_template, request, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from helpers import *
from critic import review as critic_review
from positions import position_facts, ensure_schema as ensure_positions_schema
import conversations
from sources import gather, provenance
import sqlite3
import logging
import time

# Flask app
app = Flask(__name__)

logger = logging.getLogger("financeai")
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler = logging.FileHandler("app-error.log")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    # Also to the terminal, so monitoring activity is visible while the app runs.
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

# Rate limiting, keyed on client IP. Only the question endpoint is limited --
# a question costs LLM time plus one call to each of NewsAPI, Finnhub and Reddit,
# whereas the watchlist pages only touch SQLite.
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    headers_enabled=True,
)


@app.errorhandler(429)
def rate_limited(e):
    """Render the refusal in the normal page instead of a bare error screen."""
    limit = getattr(e, "description", "") or "the rate limit"
    logger.warning("Rate limit hit by %s (%s)", get_remote_address(), limit)
    return render_template(
        'index.html',
        answer=f"You're asking faster than Fina can answer ({limit}). "
               f"Give it a moment and try again.",
        question='',
    ), 429

# Tickers the watchlist starts with on a fresh database. Edit this list to change
# the defaults. It is applied only when the table is empty, so anything added or
# removed at /watchlist afterwards is left alone. Set it to [] to start blank.
DEFAULT_WATCHLIST = [
    # ticker,    weight, alert_threshold_pct
    ('IVV',       0.30,  5.0),
    ('IWM',       0.20,  5.0),
    ('AAPL',      0.20,  5.0),
    ('MSFT',      0.13,  5.0),   # the remainder: the others come to 87%
    ('PLTR',      0.10,  8.0),   # higher bar: 5% days are routine for PLTR
    ('NVDA',      0.05,  7.0),
    ('BTC-USD',   0.02, 10.0),   # 'BTC' on Yahoo is the Grayscale ETF, not Bitcoin
]


# Lots you actually bought: (ticker, entry_date, entry_price, quantity).
# Unlike DEFAULT_WATCHLIST these are facts, not targets, so there is no sensible
# default -- leave empty and add your real purchases, one row per buy.
#   ('PLTR', '2026-05-27', 132.51, 60),
DEFAULT_POSITIONS = []

USER_ID = 'default_user'


# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                 (user_id TEXT, ticker TEXT, weight REAL, 
                  alert_threshold REAL DEFAULT 5.0, PRIMARY KEY (user_id, ticker))''')
    c.execute('''CREATE TABLE IF NOT EXISTS price_history
                 (ticker TEXT, price REAL, volume INTEGER, 
                  timestamp DATETIME, PRIMARY KEY (ticker, timestamp))''')
    # What you actually bought, and when. portfolios holds target weights;
    # this holds the lots those weights are supposed to describe.
    from positions import SCHEMA as POSITIONS_SCHEMA
    c.execute(POSITIONS_SCHEMA)
    # Durable record of every turn. The app is expected to run continuously,
    # so a question months from now can see what was said today.
    c.execute(conversations.SCHEMA)
    c.execute(conversations.INDEX)

    # Seed lots on a fresh database, same rule as DEFAULT_WATCHLIST: only into
    # an empty table, so nothing you enter later is overwritten.
    if DEFAULT_POSITIONS and not c.execute('SELECT 1 FROM positions LIMIT 1').fetchone():
        c.executemany('INSERT INTO positions VALUES (?,?,?,?,?,?)',
                      [('default_user', tk, d, pr, q, None)
                       for tk, d, pr, q in DEFAULT_POSITIONS])
        logger.info("Seeded %d position lot(s)", len(DEFAULT_POSITIONS))

    # Seed only into an empty table, so a restart never resurrects a ticker the
    # user deleted or overwrites a weight they changed.
    if DEFAULT_WATCHLIST and not c.execute('SELECT 1 FROM portfolios LIMIT 1').fetchone():
        c.executemany('INSERT INTO portfolios VALUES (?, ?, ?, ?)',
                      [('default_user', t, w, a) for t, w, a in DEFAULT_WATCHLIST])
        logger.info("Seeded watchlist with %d ticker(s): %s",
                    len(DEFAULT_WATCHLIST), ", ".join(t for t, _, _ in DEFAULT_WATCHLIST))

    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])
@limiter.limit(QUESTION_RATE_LIMIT, methods=['POST'])
def index():
    answer = ""
    question = ""
    if request.method == 'POST':
        question = request.form.get('question', '').strip()

        if not question:
            answer = "Please ask a question!"
            return render_template('index.html', answer=answer, question=question)

        started = time.monotonic()
        classification, mentioned = None, []
        critic_review.last_finding_codes = None

        try:
            # Classification is a network call too, so it belongs inside the handler.
            classifier = QuestionClassifier()
            classification = classifier.classify(question)
            intent = classification['intent']

            if intent == 'rank_portfolio':
                tickerList, weightList = classification['tickers'], classification['weights']
                answer = analyze_portfolio(tickerList, weightList)

            elif intent == 'should_buy_stock':
                tickerList = classification['tickers']
                weightList = classification['weights']
                new_stock = classification['new_stock']
                if not new_stock:
                    answer = "Please specify the stock you want to consider buying."
                elif not tickerList:
                    # "Should I buy X" compares X against a portfolio. With no
                    # holdings stated there is nothing to compare against, so
                    # assess the stock alone rather than raising.
                    answer = evaluate_stock_standalone(new_stock)
                else:
                    answer = should_buy_stock(tickerList, weightList, new_stock)

            elif intent == 'top_performers':
                typeStocks, n = classification['type'], classification['count']
                answer = get_top_performing_stocks(typeStocks, n)

            elif intent == 'stock_trend':
                symbol = classification['symbol']
                if not symbol:
                    answer = "Please tell me which stock you mean."
                else:
                    answer = get_stock_trend(symbol)

            elif intent == 'get_finance_news':
                # The classifier works out the topic, so actually use it instead of
                # always serving the same generic finance headlines.
                topic = classification['news_topic']
                category = topic if topic in FINNHUB_CATEGORIES else 'general'

                # Each source is fetched behind its own guard, concurrently.
                # One of them being down costs that source's contribution, not
                # the whole answer.
                results = gather([
                    ("NewsAPI", lambda: fetch_newsapi(topic)),
                    ("Reddit", lambda: fetch_reddit_posts()),
                    ("Finnhub", lambda: get_finnhub_news(category)),
                ])

                blocks = []
                for r in results:
                    blocks.extend(format_context_items(r.items))
                context = "\n\n".join(blocks)

                if not context.strip():
                    # Nothing came back. Refuse rather than hand an empty
                    # context to a model and let it answer from memory.
                    answer = ("No news sources are reachable right now, so I have "
                              "nothing to base an answer on.\n\n" + provenance(results))
                else:
                    answer = ask_llm(question, context)
                    # Say what the answer actually rests on. A degraded answer
                    # that looks complete is the failure mode worth avoiding.
                    answer = f"{answer}\n\n[{provenance(results)}]"

            # Second agent. It reads the figures behind the answer, never
            # recomputes them, and appends caveats. review() returns None for
            # intents that carry no structured facts, and never raises, so a
            # broken critic cannot take the answer down with it.
            mentioned = (classification.get('tickers') or []) + [
                x for x in (classification.get('new_stock'),
                            classification.get('symbol')) if x]

            if ENABLE_CRITIC and answer:
                # What this user has said before about these tickers, however
                # long ago, plus the last couple of turns for context.
                history = conversations.history_for(DB_PATH, USER_ID, mentioned)
                caveats = critic_review(question, answer, history=history)
                if caveats:
                    answer = f"{answer}\n\n{caveats}"

            # 'other', or anything the classifier invents, used to fall through here
            # and render an empty page.
            if not answer:
                answer = (
                    "I couldn't match that to something I can answer. I can:\n"
                    "  - rank a portfolio: \"rank my portfolio: AAPL 50%, TSLA 30%, MSFT 20%\"\n"
                    "  - assess a purchase: \"should I buy NVDA? I hold IVV 60%, PLTR 40%\"\n"
                    "  - list top performers: \"what are the top 5 large cap stocks?\"\n"
                    "  - check one stock: \"what's going on with Apple?\"\n"
                    "  - summarise finance news: \"tell me some crypto news\""
                )

        except Exception as e:
            logger.exception("Error processing question")
            answer = f"Error processing your question: {str(e)}"

        conversations.record(
            DB_PATH, USER_ID, question,
            intent=(classification or {}).get('intent'),
            tickers=mentioned,
            answer=answer,
            findings=getattr(critic_review, 'last_finding_codes', None),
            latency_ms=int((time.monotonic() - started) * 1000))

        return render_template('index.html', answer=answer, question=question)

    return render_template('index.html', answer="Welcome to Fina!", question=question)


@app.route('/watchlist', methods=['GET'])
def watchlist():
    """Display the watchlist page"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute('SELECT ticker, weight, alert_threshold FROM portfolios WHERE user_id=?', ('default_user',))
    items = c.fetchall()
    conn.close()
    
    watchlist_items = [
        {'ticker': ticker, 'weight': weight, 'alert_threshold': threshold}
        for ticker, weight, threshold in items
    ]
    
    return render_template('watchlist.html', watchlist=watchlist_items, message=None, message_type=None)


@app.route('/add_to_watchlist', methods=['POST'])
def add_to_watchlist():
    """Add a ticker to the watchlist"""
    user_id = 'default_user'
    ticker = request.form.get('ticker', '').strip().upper()
    weight = float(request.form.get('weight', 0)) / 100.0  # Convert percentage to decimal
    alert_threshold = float(request.form.get('alert_threshold', 5.0))
    
    if not ticker:
        return redirect(url_for('watchlist'))
    
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO portfolios VALUES (?, ?, ?, ?)',
                  (user_id, ticker, weight, alert_threshold))
        conn.commit()
        conn.close()
        
        # Get updated watchlist
        conn = sqlite3.connect(DB_PATH, timeout=30)
        c = conn.cursor()
        c.execute('SELECT ticker, weight, alert_threshold FROM portfolios WHERE user_id=?', (user_id,))
        items = c.fetchall()
        conn.close()
        
        watchlist_items = [
            {'ticker': t, 'weight': w, 'alert_threshold': th}
            for t, w, th in items
        ]
        
        return render_template('watchlist.html', 
                             watchlist=watchlist_items, 
                             message=f'{ticker} added to watchlist!',
                             message_type='success')
    except Exception as e:
        return render_template('watchlist.html', 
                             watchlist=[], 
                             message=f'Error: {str(e)}',
                             message_type='error')


@app.route('/remove_from_watchlist', methods=['POST'])
def remove_from_watchlist():
    """Remove a ticker from the watchlist"""
    user_id = 'default_user'
    ticker = request.form.get('ticker', '').strip().upper()
    
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        c = conn.cursor()
        c.execute('DELETE FROM portfolios WHERE user_id=? AND ticker=?', (user_id, ticker))
        conn.commit()
        conn.close()
        
        # Get updated watchlist
        conn = sqlite3.connect(DB_PATH, timeout=30)
        c = conn.cursor()
        c.execute('SELECT ticker, weight, alert_threshold FROM portfolios WHERE user_id=?', (user_id,))
        items = c.fetchall()
        conn.close()
        
        watchlist_items = [
            {'ticker': t, 'weight': w, 'alert_threshold': th}
            for t, w, th in items
        ]
        
        return render_template('watchlist.html', 
                             watchlist=watchlist_items, 
                             message=f'{ticker} removed from watchlist.',
                             message_type='success')
    except Exception as e:
        return render_template('watchlist.html', 
                             watchlist=[], 
                             message=f'Error: {str(e)}',
                             message_type='error')


if __name__ == '__main__':
    # Started here rather than at module scope so importing app.py (tests, or a
    # WSGI server with several workers) does not spin up a scheduler per process.
    if ENABLE_MONITORING:
        from monitoring_service import StockMonitor
        StockMonitor().start()
    else:
        logger.info("Monitoring disabled (ENABLE_MONITORING=0)")

    # Use debug=False for production deployment
    #app.run(debug=True, port=5002)
    app.run(debug=False, host='0.0.0.0', port=5000)