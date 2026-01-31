from flask import Flask, render_template, request, redirect, url_for
from helpers import *
import sqlite3

# Flask app
app = Flask(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('portfolios.db', timeout=30)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                 (user_id TEXT, ticker TEXT, weight REAL, 
                  alert_threshold REAL DEFAULT 5.0, PRIMARY KEY (user_id, ticker))''')
    c.execute('''CREATE TABLE IF NOT EXISTS price_history
                 (ticker TEXT, price REAL, volume INTEGER, 
                  timestamp DATETIME, PRIMARY KEY (ticker, timestamp))''')
    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    question = ""
    if request.method == 'POST':
        question = request.form.get('question', '').strip()

        if not question:
            answer = "Please ask a question!"
            return render_template('index.html', answer=answer, question=question)

        classifier = QuestionClassifier()
        classification = classifier.classify(question)

        # print('classification')
        # print(classification)

        try:
            if classification['intent'] == 'rank_portfolio':
                tickerList, weightList = classification['tickers'], classification['weights']
                #print(tickerList, weightList)
                answer=analyze_portfolio(tickerList, weightList)

            elif classification['intent'] == 'should_buy_stock':
                tickerList = classification['tickers']
                weightList = classification['weights']
                new_stock = classification['new_stock']
                if not new_stock:
                    answer = "Please specify the stock you want to consider buying."
                else:
                    answer = should_buy_stock(tickerList, weightList, new_stock)

            elif classification['intent'] == 'top_performers':
                typeStocks, n = classification['type'], classification['count']
                answer = get_top_performing_stocks(typeStocks, n)
            
            elif classification['intent'] == 'stock_trend':
                symbol = classification['symbol']
                answer = get_stock_trend(symbol)
            
            elif classification['intent'] == 'get_finance_news':
                news = fetch_newsapi()
                reddit_posts = fetch_reddit_posts()
                finnhub = get_finnhub_news()

                formatted_news = format_context_items(news)
                formatted_reddit = format_context_items(reddit_posts)
                formatted_finnhub = format_context_items(finnhub)
                context = "\n\n".join(formatted_news + formatted_reddit + formatted_finnhub)

                answer = ask_openai(question, context)

        except Exception as e:
            answer = f"Error processing your question: {str(e)}"
        
        return render_template('index.html', answer=answer, question=question)

        
    return render_template('index.html', answer="Welcome to Fina!", question=question)



@app.route('/watchlist', methods=['GET'])
def watchlist():
    """Display the watchlist page"""
    conn = sqlite3.connect('portfolios.db', timeout=30)
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
        conn = sqlite3.connect('portfolios.db', timeout=30)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO portfolios VALUES (?, ?, ?, ?)',
                  (user_id, ticker, weight, alert_threshold))
        conn.commit()
        conn.close()
        
        # Get updated watchlist
        conn = sqlite3.connect('portfolios.db', timeout=30)
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
        conn = sqlite3.connect('portfolios.db', timeout=30)
        c = conn.cursor()
        c.execute('DELETE FROM portfolios WHERE user_id=? AND ticker=?', (user_id, ticker))
        conn.commit()
        conn.close()
        
        # Get updated watchlist
        conn = sqlite3.connect('portfolios.db', timeout=30)
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


# Uncomment when you want to enable monitoring
# from monitoring_service import StockMonitor
# monitor = StockMonitor()
# monitor.start()

if __name__ == '__main__':
    # Use debug=False for production deployment
    #app.run(debug=True, port=5002)
    app.run(debug=False, host='0.0.0.0', port=5000)