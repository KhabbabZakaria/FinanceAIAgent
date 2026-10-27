from config import *
import requests
import praw
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List
import logging
import re
import json

from llm import chat
from sources import DEFAULT_TIMEOUT, SourceError

logger = logging.getLogger("financeai.helpers")






CLASSIFIER_SYSTEM_PROMPT = """You are a classification assistant for a finance app.
Output ONLY valid JSON (no explanations or extra text).

Schema:
{
"intent": "<one of: rank_portfolio, top_performers, stock_trend, get_finance_news, should_buy_stock, other>",
"tickers": [<the stocks the user ALREADY HOLDS, as ticker symbols>],
"weights": [<numbers; if user gave percents convert to decimals; else empty list>],
"new_stock": "<the stock the user is CONSIDERING BUYING, or null>",
"type": "<large|mid|small|all>",
"count": <integer, default 3>,
"symbol": "<single ticker the user is asking about, or null>",
"news_topic": "<short topic for news or 'general'>"
}

Two distinctions decide most questions:

1. "tickers" is what the user ALREADY OWNS. "new_stock" is what they are
   THINKING OF BUYING. A stock being considered never goes in "tickers",
   whatever its symbol is. If no holdings are named, "tickers" is empty.

2. Asking about the risk, volatility, return or quality of holdings they
   already have is rank_portfolio. It is only should_buy_stock when the user
   names a specific stock they do not yet own and asks whether to add it.

Examples:

Q: Rank my portfolio: IVV 50%, KO 30%, BND 20%
A: {"intent":"rank_portfolio","tickers":["IVV","KO","BND"],"weights":[0.5,0.3,0.2],"new_stock":null,"type":"all","count":3,"symbol":null,"news_topic":"general"}

Q: rank my portfolio: VOO 60%, GLD 25%, ETH-USD 15%
A: {"intent":"rank_portfolio","tickers":["VOO","GLD","ETH-USD"],"weights":[0.6,0.25,0.15],"new_stock":null,"type":"all","count":3,"symbol":null,"news_topic":"general"}

Q: how risky is UNP 70%, PEP 30%?
A: {"intent":"rank_portfolio","tickers":["UNP","PEP"],"weights":[0.7,0.3],"new_stock":null,"type":"all","count":3,"symbol":null,"news_topic":"general"}

Q: Should I buy NVDA? My portfolio is SPY 40%, IWM 30%, PLTR 20%, XOM 10%
A: {"intent":"should_buy_stock","tickers":["SPY","IWM","PLTR","XOM"],"weights":[0.4,0.3,0.2,0.1],"new_stock":"NVDA","type":"all","count":3,"symbol":null,"news_topic":"general"}

Q: should I buy COST? I hold VTI 60%, JNJ 40%
A: {"intent":"should_buy_stock","tickers":["VTI","JNJ"],"weights":[0.6,0.4],"new_stock":"COST","type":"all","count":3,"symbol":null,"news_topic":"general"}

Q: is UNH worth buying?
A: {"intent":"should_buy_stock","tickers":[],"weights":[],"new_stock":"UNH","type":"all","count":3,"symbol":null,"news_topic":"general"}

Q: What are the top 5 large cap stocks?
A: {"intent":"top_performers","tickers":[],"weights":[],"new_stock":null,"type":"large","count":5,"symbol":null,"news_topic":"general"}

Q: What's going on with Walmart?
A: {"intent":"stock_trend","tickers":[],"weights":[],"new_stock":null,"type":"all","count":3,"symbol":"WMT","news_topic":"general"}

Q: what's going on in the energy sector?
A: {"intent":"get_finance_news","tickers":[],"weights":[],"new_stock":null,"type":"all","count":3,"symbol":null,"news_topic":"energy"}

Q: Tell me some crypto news
A: {"intent":"get_finance_news","tickers":[],"weights":[],"new_stock":null,"type":"all","count":3,"symbol":null,"news_topic":"crypto"}

Now classify the user's question."""

DEFAULT_CLASSIFICATION = {
    "intent": "other",
    "tickers": [],
    "weights": [],
    "new_stock": None,
    "type": "all",
    "count": 3,
    "symbol": None,
    "news_topic": "general",
}


class QuestionClassifier:
    def __init__(self, temperature: float = 0.0):
        """Classify questions via the provider chain in llm.py.

        The model is chosen by LLM_PROVIDERS rather than pinned here, so the local
        small model handles this and a hosted model takes over only if it is down.
        """
        self.temperature = temperature

    @staticmethod
    def _try_extract_json(text: str) -> str:
        """Find the first {...} JSON block in the model output and return it."""
        text = text.strip()
        match = re.search(r'(\{.*\})', text, flags=re.S)
        if match:
            return match.group(1)
        return text

    @staticmethod
    def _normalize_ticker(t: str) -> str:
        return t.strip().upper().replace(' ', '')

    @staticmethod
    def _normalize_weights(weights: List[Any]) -> List[float]:
        """Turn things like ['50%', '30%', '20%'] or [0.5, 0.3, 0.2] into decimals summing to 1."""
        out = []
        for w in weights:
            try:
                f = float(w)
                if f > 1 and f <= 100:
                    f = f / 100.0
                out.append(f)
            except Exception:
                s = str(w).strip().replace('%', '')
                try:
                    f = float(s) / 100.0
                except Exception:
                    f = 0.0
                out.append(f)
        s = sum(out)
        if s > 0:
            out = [x / s for x in out]
        return out

    # Uppercase words that look like tickers but are not. Kept short on purpose:
    # a false positive here would invent a holding the user never mentioned.
    _NOT_TICKERS = {
        "I", "A", "AN", "THE", "AND", "OR", "IF", "IS", "IT", "ME", "MY", "DO",
        "USD", "EUR", "GBP", "ETF", "IPO", "CEO", "CFO", "GDP", "CPI", "FED",
        "USA", "UK", "EU", "AI", "ML", "IT", "OK", "PE", "EPS", "ROI", "YTD",
        "ATH", "FAQ", "NYSE", "SEC", "IRA", "ESG", "TL", "DR",
    }

    @staticmethod
    def _example_tickers() -> set:
        """Every ticker that appears in the few-shot answers.

        Derived from the prompt rather than hardcoded, so it cannot drift out
        of sync when the examples are edited.
        """
        found = set()
        for line in CLASSIFIER_SYSTEM_PROMPT.splitlines():
            line = line.strip()
            if not line.startswith("A: {"):
                continue
            try:
                d = json.loads(line[3:])
            except Exception:
                continue
            found.update(x.upper() for x in d.get("tickers") or [])
            for key in ("new_stock", "symbol"):
                if d.get(key):
                    found.add(str(d[key]).upper())
        return found

    @staticmethod
    def _strip_null_tickers(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Drop null/blank entries, keeping tickers and weights aligned.

        The model emits null inside the list occasionally. Unfiltered it used
        to reach _normalize_ticker(None) and raise.
        """
        raw = parsed.get("tickers") or []
        weights = list(parsed.get("weights") or [])
        if not raw:
            return parsed
        if len(raw) == len(weights):
            pairs = [(str(x).upper(), w) for x, w in zip(raw, weights) if x]
            parsed["tickers"] = [x for x, _ in pairs]
            parsed["weights"] = [w for _, w in pairs]
        else:
            parsed["tickers"] = [str(x).upper() for x in raw if x]
        return parsed

    @classmethod
    def _drop_invented_holdings(cls, question: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Remove holdings the model copied out of the few-shot examples.

        Asked "should I add PLTR to my portfolio" with no holdings named, the
        model returns the example's portfolio verbatim (VTI 60%, JNJ 40%).
        Analysing a portfolio the user does not own, and recommending on it, is
        worse than admitting we do not know what they hold.

        Only fires when the question states no quantity at all. "40 percent
        Nvidia, 60 percent Coca Cola" names real holdings by company name, so
        the symbols NVDA and KO are absent from the text and would otherwise
        look invented. A stated quantity means stated holdings; leave them be.
        """
        if parsed.get("intent") not in ("rank_portfolio", "should_buy_stock"):
            return parsed

        raw = parsed.get("tickers") or []
        weights = list(parsed.get("weights") or [])
        if not raw or len(raw) != len(weights):
            return parsed
        tickers = [str(x).upper() for x in raw]

        asked = question.upper()
        if cls._QUANTITY.search(asked):
            return parsed

        examples = cls._example_tickers()
        keep_t, keep_w = [], []
        for ticker, weight in zip(tickers, weights):
            invented = ticker in examples and not re.search(
                rf"\b{re.escape(ticker)}\b", asked)
            if invented:
                logger.warning("Dropping %s: a few-shot example holding the "
                               "question never mentions", ticker)
                continue
            keep_t.append(ticker)
            keep_w.append(weight)

        parsed["tickers"] = keep_t
        parsed["weights"] = keep_w
        return parsed

    # Any sign the user quantified a holding: a percent, a written percent, or
    # a fraction word. Its presence means they described a real portfolio.
    _QUANTITY = re.compile(r"\d+\s*%|\d+\s*PERCENT|\bHALF\b|\bTHIRD\b|"
                           r"\bQUARTER\b|\bEQUALLY\b|\d+\s*/\s*\d+")

    # "IVV 30%" or "BTC-USD 2%" -- a ticker immediately followed by a percent.
    _WEIGHTED = re.compile(r"\b([A-Z][A-Z.\-]{1,9})\b[\s:=]*(\d+(?:\.\d+)?)\s*%")

    @classmethod
    def _repair_dropped_holdings(cls, question: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Put back a weighted holding the model left out of `tickers`.

        Dropping one is bad; the normalisation that follows is worse, because it
        rescales whatever survived back to 100% and the loss becomes invisible.
        The question states both the symbol and the percent, so recover them
        rather than analysing a portfolio the user never described.
        """
        if parsed.get("intent") not in ("rank_portfolio", "should_buy_stock"):
            return parsed

        tickers = [t.upper() for t in parsed.get("tickers") or []]
        weights = list(parsed.get("weights") or [])
        if len(tickers) != len(weights):
            return parsed                       # can't align; leave it alone

        candidate = (parsed.get("new_stock") or "").upper()
        for symbol, pct in cls._WEIGHTED.findall(question):
            if symbol in cls._NOT_TICKERS or symbol in tickers or symbol == candidate:
                continue
            logger.warning("Model dropped weighted holding %s %s%%; restoring it",
                           symbol, pct)
            tickers.append(symbol)
            weights.append(float(pct))

        parsed["tickers"] = tickers
        parsed["weights"] = weights
        return parsed

    @classmethod
    def _repair_should_buy(cls, question: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Recover the candidate stock when the model leaves `new_stock` empty.

        The model intermittently files the stock being *considered* under
        `tickers`, which are supposed to be existing holdings. Both repairs are
        deterministic and only ever run when `new_stock` is already empty, so a
        correct classification is never overwritten.
        """
        if parsed.get("intent") != "should_buy_stock" or parsed.get("new_stock"):
            return parsed

        tickers = parsed.get("tickers") or []
        held = {t.upper() for t in tickers}

        # 1. A ticker-shaped word in the question that was not filed as a holding
        #    is almost certainly the candidate.
        mentioned = {w for w in re.findall(r"\b[A-Z][A-Z.\-]{1,5}\b", question)
                     if w not in cls._NOT_TICKERS and w not in held}
        if len(mentioned) == 1:
            parsed["new_stock"] = mentioned.pop()
            return parsed

        # Several unexplained tickers ("should I buy AAPL or MSFT?") is a
        # genuinely ambiguous question. Asking beats guessing, and in
        # particular beats falling through to rule 2 and promoting a real
        # holding the user did state.
        if mentioned:
            return parsed

        # 2. Nothing else named, and a single "holding": that is the candidate
        #    mis-filed. Nobody asks whether to buy a one-stock portfolio they
        #    already own in full.
        if len(tickers) == 1:
            parsed["new_stock"] = tickers[0]
            parsed["tickers"] = []
            parsed["weights"] = []

        return parsed

    def classify(self, question: str) -> Dict[str, Any]:
        # The question goes in the user turn only. It used to be appended to the
        # system prompt as well, which smaller models read as two separate asks.
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        result = chat(messages, json_mode=True, max_tokens=300, temperature=self.temperature)
        logger.info("Classified via %s/%s in %dms", result.provider, result.model, result.latency_ms)

        try:
            parsed = json.loads(self._try_extract_json(result.text))
        except Exception:
            logger.warning("Unparseable classification from %s: %r", result.provider, result.text[:200])
            parsed = dict(DEFAULT_CLASSIFICATION)

        parsed = self._strip_null_tickers(parsed)
        parsed = self._drop_invented_holdings(question, parsed)
        parsed = self._repair_dropped_holdings(question, parsed)
        parsed = self._repair_should_buy(question, parsed)

        intent = parsed.get("intent", "other")
        tickers = [self._normalize_ticker(str(t))
                   for t in parsed.get("tickers") or [] if t]
        weights = self._normalize_weights(parsed.get("weights") or [])
        typ = (parsed.get("type") or "all").lower()
        try:
            count = int(parsed.get("count") or 3)
        except (TypeError, ValueError):
            count = 3
        symbol = parsed.get("symbol") or None
        news_topic = (parsed.get("news_topic") or "general").lower()
        new_stock = parsed.get("new_stock") or None

        return {
            "intent": intent,
            "tickers": tickers,
            "weights": weights,
            "new_stock": (str(new_stock).strip().upper() if new_stock else None),
            "type": typ,
            "count": count,
            "symbol": (str(symbol).strip().upper() if symbol else None),
            "news_topic": news_topic,
            "_provider": result.provider,
        }


# A small local model is likelier than GPT to answer from memory when the context
# is thin, so the grounding rule is explicit rather than implied by the layout.
SYNTHESIS_SYSTEM_PROMPT = (
    "You are a finance news assistant. Base your answer on the supplied context, "
    "summarising and connecting what it reports. Use only facts present in the context: "
    "never invent prices, tickers, dates or figures, and never add market knowledge of "
    "your own. If the context contains nothing relevant to the question, say so plainly."
)


def ask_llm(question, context):
    """Answer `question` from `context` using the first available provider."""
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
    ]
    result = chat(messages, max_tokens=300)
    logger.info("Synthesised via %s/%s in %dms", result.provider, result.model, result.latency_ms)
    return result.text


# Kept so existing callers keep working after the provider swap.
ask_openai = ask_llm


class Analysis(str):
    """The formatted answer, with the numbers behind it attached.

    Subclasses str so every existing caller -- the router, the template, the
    tests -- keeps treating it as the answer text. `.facts` carries the figures
    the critic agent reasons over, so the critic never has to re-parse prose or
    recompute anything.
    """

    def __new__(cls, text: str, facts: dict):
        obj = super().__new__(cls, text)
        obj.facts = facts
        return obj


# Finnhub only accepts this fixed set of news categories.
FINNHUB_CATEGORIES = {"general", "forex", "crypto", "merger"}


def fetch_newsapi(topic="general"):
    query = topic if topic and topic != "general" else "finance OR stock OR crypto"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
    }
    response = requests.get("https://newsapi.org/v2/everything", params=params,
                            timeout=DEFAULT_TIMEOUT)

    if response.status_code != 200:
        raise SourceError(f"NewsAPI error {response.status_code}: {response.text[:200]}",
                          status=response.status_code)

    articles = response.json().get("articles", [])[:5]
    return [f"{a['title']} - {a['source']['name']}\n{a['description']}" for a in articles]


def fetch_reddit_posts(subreddit_name="wallstreetbets", limit=5):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        timeout=DEFAULT_TIMEOUT[1],
    )
    reddit.read_only = True
    subreddit = reddit.subreddit(subreddit_name)

    posts = []
    for post in subreddit.hot(limit=limit):
        if not post.stickied:
            posts.append({
                "title": post.title,
                "content": post.selftext
            })
    return posts




def get_finnhub_news(category="general", max_articles=5):
    url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}"
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)

    if response.status_code != 200:
        raise SourceError(f"Finnhub error {response.status_code}: {response.text[:200]}",
                          status=response.status_code)
    
    news = response.json()
    
    results = []
    for item in news[:max_articles]:
        results.append({
            "datetime": datetime.fromtimestamp(item['datetime']),
            "headline": item['headline'],
            "source": item['source'],
            "summary": item.get('summary', 'No summary available'),
            "url": item['url']
        })
    
    return results    





def get_finnhub_company_news(symbol, hours=1, max_articles=5):
    """Company-specific news for one ticker, limited to the last `hours`."""
    to_date = datetime.now()
    params = {
        "symbol": symbol,
        "from": (to_date - timedelta(days=(hours // 24) + 1)).strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "token": FINNHUB_API_KEY,
    }
    response = requests.get("https://finnhub.io/api/v1/company-news", params=params,
                            timeout=DEFAULT_TIMEOUT)

    if response.status_code != 200:
        raise SourceError(f"Finnhub company-news error {response.status_code}: "
                          f"{response.text[:200]}", status=response.status_code)

    cutoff = to_date - timedelta(hours=hours)
    results = []
    for item in response.json():
        published = datetime.fromtimestamp(item['datetime'])
        if published < cutoff:
            continue
        results.append({
            "datetime": published,
            "headline": item['headline'],
            "source": item['source'],
            "summary": item.get('summary', 'No summary available'),
            "url": item['url']
        })
        if len(results) >= max_articles:
            break

    return results




def format_context_items(items):
    # Format the returns of some of the fucntions to make sure they are string type
    formatted = []
    for item in items:
        if isinstance(item, dict):
            text = "\n".join(f"{k}: {v}" for k, v in item.items())
            formatted.append(text)
        else:
            formatted.append(str(item))
    return formatted




def get_stock_trend(symbol):
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(days=7)

    # Download historical data for last 8 days to cover weekends/holidays
    data = yf.download(symbol, start=last_week - timedelta(days=2), end=today + timedelta(days=1), progress=False, threads=False)

    if data.empty:
        return f"No data found for {symbol}"

    # yfinance returns MultiIndex columns even for a single ticker; flatten to a 1-D series.
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    trading_days = close.index.date
    earliest = trading_days.min()

    # Get closing prices, adjust for market holidays/weekends
    def get_closest_close(date):
        # If market closed, find closest previous day with data. Stop at the start of the
        # window rather than walking backwards forever across a long closure.
        while date not in trading_days:
            date -= timedelta(days=1)
            if date < earliest:
                return None
        return float(close[trading_days == date].iloc[0])

    close_today = get_closest_close(today)
    close_yesterday = get_closest_close(yesterday)
    close_last_week = get_closest_close(last_week)

    if close_today is None or close_yesterday is None or close_last_week is None:
        return f"Not enough trading-day data in the last week to assess {symbol}."

    # Calculate % changes
    change_today = ((close_today - close_yesterday) / close_yesterday) * 100
    change_week = ((close_today - close_last_week) / close_last_week) * 100

    # Determine trend
    trend = "Bullish" if change_week > 0 else "Bearish"

    # Simple behavior assessment
    if abs(change_today) < 0.5 and abs(change_week) < 1:
        behavior = "Stable"
    elif change_week > 2:
        behavior = "Strong Uptrend"
    elif change_week < -2:
        behavior = "Strong Downtrend"
    else:
        behavior = "Moderate Movement"

    result = (
        f"Stock: {symbol}\n"
        f"Close Price Today: ${close_today:.2f}\n"
        f"Close Price Yesterday: ${close_yesterday:.2f} ({change_today:.2f}%)\n"
        f"Close Price Last Week: ${close_last_week:.2f} ({change_week:.2f}%)\n"
        f"Trend over last week: {trend}\n"
        f"Behavior: {behavior}"
    )

    return result


def get_top_performing_stocks_helpers(stock_symbols, top_n=5, period="1y"):

    try:
        # Download historical data for all stocks at once for efficiency.
        data = yf.download(stock_symbols, period=period, group_by='ticker', progress=False)

        if data.empty:
            return "Could not download any stock data. Please check symbols and network."

        performance_data = {}

        for symbol in stock_symbols:
            if symbol in data.columns:
                stock_data = data[symbol]
                
                # Remove any rows with missing 'Close' prices
                stock_data = stock_data.dropna(subset=['Close'])

                if not stock_data.empty:
                    # Calculate performance from the first available day to the last
                    start_price = stock_data['Close'].iloc[0]
                    end_price = stock_data['Close'].iloc[-1]
                    performance = ((end_price - start_price) / start_price) * 100
                    performance_data[symbol] = performance

    except Exception as e:
        return f"An error occurred: {e}"

    if not performance_data:
        return "No performance data could be calculated."

    # Convert the performance dictionary to a pandas DataFrame for easy sorting
    performance_df = pd.DataFrame(
        list(performance_data.items()),
        columns=['Symbol', 'Performance (%)']
    )

    # Sort by performance in descending order and get the top N
    top_performers = performance_df.sort_values(by='Performance (%)', ascending=False).head(top_n)
    
    return top_performers


def get_top_performing_stocks(typeStocks, n=5):
    large_cap_symbols = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
        "TSLA", "BRK-B", "JPM", "WMT", "UNH"
    ]

    mid_cap_symbols = [
        "PLTR", "VRT", "RBLX", "SNV", "ELF",
        "RGEN", "BOOT", "CVLT", "ORA", "PEN"
    ]

    small_cap_symbols = [
        "CELH", "MGNI", "AMPL", "SG", "SERV",
        "QBTS", "RGTI", "SEZL", "TGEN", "NUTX"
    ]

    all_stock_lists = {
        "large": large_cap_symbols,
        "mid": mid_cap_symbols,
        "small": small_cap_symbols,
        "all": large_cap_symbols + mid_cap_symbols + small_cap_symbols
    }

    symbols_to_check = all_stock_lists[typeStocks]
    analysis_period = "1y"

    top_stocks_df = get_top_performing_stocks_helpers(
        stock_symbols=symbols_to_check,
        top_n=n,
        period=analysis_period
    )

    if isinstance(top_stocks_df, str):  # In case of an error message
        return top_stocks_df

    output = "Top Performing Stocks (Based on 1-Year Performance):\n"
    output += top_stocks_df.to_string(index=False)
    return output





def analyze_portfolio(tickers: list, weights: list) -> str:
    # Validate input
    if len(tickers) != len(weights):
        raise ValueError("Length of tickers and weights must match.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1.0")

    # Dates: today and one year ago
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Convert weights to Series for alignment
    weights_series = pd.Series(weights, index=tickers)

    # Download historical data
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=False)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])

    # Drop tickers that returned nothing before dropping incomplete rows. Otherwise one
    # bad symbol is an all-NaN column, every row is dropped, and the metrics below are
    # computed on an empty frame.
    data = data.dropna(axis=1, how='all').dropna()

    if data.empty:
        return f"No overlapping historical data for: {', '.join(tickers)}"

    # Say so rather than silently renormalising: the weights were validated to sum to
    # 1.0, and dropping a holding quietly breaks that promise.
    missing = [t for t in tickers if t not in data.columns]
    if missing:
        return (f"No historical data for: {', '.join(missing)}. "
                f"Correct or remove those symbols so the weights still sum to 100%.")

    # Calculate daily returns
    returns = np.log(data / data.shift(1)).dropna()

    # Expected returns and covariances
    mean_daily_returns = returns.mean()
    expected_annual_returns = mean_daily_returns * 252
    cov_matrix = returns.cov() * 252

    # Align weights
    aligned_weights = weights_series.loc[data.columns]

    # Portfolio metrics
    portfolio_return = np.dot(aligned_weights, expected_annual_returns.loc[data.columns])
    portfolio_variance = np.dot(aligned_weights.T, np.dot(cov_matrix.loc[data.columns, data.columns], aligned_weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)

    # Risk-free rate from 10-year Treasury
    risk_free_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

    # Build asset-level summary
    asset_metrics = pd.DataFrame({
        'Expected Annual Return': expected_annual_returns,
        'Weight in Portfolio': aligned_weights
    })
    asset_metrics['Weighted Return'] = asset_metrics['Expected Annual Return'] * asset_metrics['Weight in Portfolio']

    # Format output
    output = []
    output.append("=== Portfolio Performance Metrics ===")
    output.append(f"Date Range: {start_date} to {end_date}")
    output.append(f"Expected Annual Return: {portfolio_return:.2%}")
    output.append(f"Portfolio Volatility (Std Dev): {portfolio_std_dev:.2%}")
    output.append(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    output.append(f"Risk-Free Rate (10Y Treasury): {risk_free_rate:.2%}\n")

    output.append("--- Asset-level Summary ---")
    asset_metrics_str = asset_metrics.to_string(float_format='{:,.2%}'.format)
    output.append(asset_metrics_str)

    facts = {
        "kind": "rank_portfolio",
        "weights": {t: float(aligned_weights[t]) for t in data.columns},
        "expected_annual_return": float(portfolio_return),
        "volatility": float(portfolio_std_dev),
        "sharpe": float(sharpe_ratio),
        "risk_free_rate": float(risk_free_rate),
        "per_asset_return": {t: float(expected_annual_returns[t]) for t in data.columns},
    }
    return Analysis("\n".join(output), facts)


def evaluate_stock_standalone(ticker: str) -> str:
    """Assess one stock on its own, for a user who stated no holdings.

    should_buy_stock answers "does this improve what I already have", which has
    no meaning without a portfolio. Rather than erroring, say what can be said
    about the stock by itself and ask for holdings to go further.
    """
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)['Close']
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    data = data.dropna()

    if data.empty:
        return f"No historical data available for {ticker}."

    returns = np.log(data / data.shift(1)).dropna()
    annual_return = float(returns.mean() * 252)
    volatility = float(returns.std() * np.sqrt(252))
    risk_free_rate = float(yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100)
    sharpe = (annual_return - risk_free_rate) / volatility if volatility else float('nan')

    output = [
        f"=== {ticker} on its own ===",
        f"Expected Annual Return: {annual_return:.2%}",
        f"Volatility (Std Dev): {volatility:.2%}",
        f"Sharpe Ratio: {sharpe:.2f}",
        f"Risk-Free Rate (10Y Treasury): {risk_free_rate:.2%}",
        "",
        "These are the stock's own figures over the last year. Whether buying it "
        "is a good idea depends on what you already hold -- tell me your current "
        'positions (for example "I hold IVV 60%, KO 40%") and I can work out '
        "whether it improves the mix.",
    ]
    facts = {
        "kind": "stock_standalone",
        "ticker": ticker,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "risk_free_rate": risk_free_rate,
    }
    return Analysis("\n".join(output), facts)


def should_buy_stock(tickers: list, weights: list, new_stock: str) -> str:

    if len(tickers) != len(weights):
        raise ValueError("Length of tickers and weights must match.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1.0")
    if new_stock in tickers:
        return (f"{new_stock} is already in your portfolio at "
                f"{weights[tickers.index(new_stock)]:.1%}.")

    # Dates
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Fetch data for current portfolio and new stock
    all_tickers = tickers + [new_stock]
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, threads=False)['Close']
    data = data.dropna(axis=1, how='all').dropna()

    if data.empty:
        return f"No historical data available for one or more tickers: {all_tickers}"

    missing = [t for t in all_tickers if t not in data.columns]
    if missing:
        return f"No historical data available for: {', '.join(missing)}"

    # yfinance returns 'Close' columns sorted alphabetically, not in the order they were
    # requested. Without this reindex the weight vectors below land on the wrong stocks.
    data = data[all_tickers]

    # Daily returns
    returns = np.log(data / data.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Risk-free rate
    risk_free_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100

    # Helper: portfolio performance for a weight vector over the columns `cols`,
    # selected by name so weights can never drift onto the wrong ticker.
    def portfolio_performance(w, cols):
        w = np.array(w)
        port_return = np.dot(w, mean_returns.loc[cols])
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.loc[cols, cols], w)))
        sharpe = (port_return - risk_free_rate) / port_vol
        return port_return, port_vol, sharpe

    # Helper: max achievable Sharpe using only `cols`, long-only and fully invested
    def best_sharpe(cols):
        n = len(cols)
        result = minimize(
            lambda w: -portfolio_performance(w, cols)[2],
            np.ones(n) / n,
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(n)),
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        return result.x, portfolio_performance(result.x, cols)[2]

    # What the user holds today, at their own weights
    curr_sharpe = portfolio_performance(weights, tickers)[2]

    # Compare like with like: best achievable WITHOUT the new stock against best
    # achievable WITH it. Measuring the user's own weights against an optimised
    # portfolio would credit the new stock for gains that come purely from
    # re-weighting the holdings they already have.
    _, best_without = best_sharpe(tickers)
    new_weights, best_with = best_sharpe(all_tickers)

    decision = "BUY" if best_with > best_without + 1e-6 else "DON'T BUY"

    # Format output
    output = []
    output.append(f"=== Should You Buy {new_stock}? ===")
    output.append(f"Your portfolio today, at your weights: {curr_sharpe:.2f}")
    output.append(f"Best achievable Sharpe without {new_stock}: {best_without:.2f}")
    output.append(f"Best achievable Sharpe with {new_stock}: {best_with:.2f}")
    output.append(f"Marginal gain from adding {new_stock}: {best_with - best_without:+.3f}")
    output.append(f"Decision: {decision}\n")

    suggested = {t: float(w) for t, w in zip(all_tickers, new_weights)}
    if decision == "BUY":
        new_alloc_df = pd.DataFrame({
            'Ticker': all_tickers,
            'Weight': new_weights
        })
        output.append("Suggested New Weights:")
        output.append(new_alloc_df.to_string(index=False, float_format="{:.2%}".format))

    facts = {
        "kind": "should_buy_stock",
        "new_stock": new_stock,
        "current_weights": dict(zip(tickers, weights)),
        "current_sharpe": float(curr_sharpe),
        "best_without": float(best_without),
        "best_with": float(best_with),
        "marginal_gain": float(best_with - best_without),
        "decision": decision,
        "suggested_weights": suggested if decision == "BUY" else None,
    }
    return Analysis("\n".join(output), facts)
