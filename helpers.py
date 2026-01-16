from config import *
from openai import OpenAI
import requests
import praw
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List
import re
import json

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage






class QuestionClassifier:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """Initialize the classifier with an OpenAI chat model."""
        self.chat = ChatOpenAI(model_name=model_name, temperature=temperature)

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

    def classify(self, question: str) -> Dict[str, Any]:
        system = """You are a classification assistant for a finance app.
    Output ONLY valid JSON (no explanations or extra text).

    Schema:
    {
    "intent": "<one of: rank_portfolio, top_performers, stock_trend, get_finance_news, should_buy_stock, other>",
    "tickers": [<tickers, as strings, use ticker symbols if user gave names>],
    "weights": [<numbers; if user gave percents convert to decimals; else empty list>],
    "new_stock": "<single ticker for should_buy_stock, or null>",
    "type": "<large|mid|small|all>",
    "count": <integer, default 3>,
    "symbol": "<single ticker or null>",
    "news_topic": "<short topic for news or 'general'>"
    }

    Examples:

    Q: Rank my portfolio: AAPL 50%, TSLA 30%, MSFT 20%
    A: {"intent":"rank_portfolio","tickers":["AAPL","TSLA","MSFT"],"weights":[0.5,0.3,0.2],"new_stock":null,"type":"all","count":3,"symbol":null,"news_topic":"general"}

    Q: Should I buy AAPL? My portfolio is IVV 40%, IWM 30%, SOL 20%, PLTR 10%
    A: {"intent":"should_buy_stock","tickers":["IVV","IWM","SOL","PLTR"],"weights":[0.4,0.3,0.2,0.1],"new_stock":"AAPL","type":"all","count":3,"symbol":null,"news_topic":"general"}

    Q: What are the top 5 large cap stocks?
    A: {"intent":"top_performers","tickers":[],"weights":[],"new_stock":null,"type":"large","count":5,"symbol":null,"news_topic":"general"}

    Q: What's going on with Apple?
    A: {"intent":"stock_trend","tickers":[],"weights":[],"new_stock":null,"type":"all","count":3,"symbol":"AAPL","news_topic":"general"}

    Q: Tell me some crypto news
    A: {"intent":"get_finance_news","tickers":[],"weights":[],"new_stock":null,"type":"all","count":3,"symbol":null,"news_topic":"crypto"}

    Now classify the user's question.

    User question: """ + question

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=question)
        ]

        resp = self.chat(messages)
        text = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
        json_text = self._try_extract_json(text)

        try:
            parsed = json.loads(json_text)
        except Exception:
            parsed = {
                "intent": "other",
                "tickers": [],
                "weights": [],
                "new_stock": None,
                "type": "all",
                "count": 3,
                "symbol": None,
                "news_topic": "general"
            }

        intent = parsed.get("intent", "other")
        tickers = [self._normalize_ticker(t) for t in parsed.get("tickers") or []]
        weights = self._normalize_weights(parsed.get("weights") or [])
        typ = (parsed.get("type") or "all").lower()
        count = int(parsed.get("count") or 3)
        symbol = parsed.get("symbol") or None
        news_topic = (parsed.get("news_topic") or "general").lower()
        new_stock = parsed.get("new_stock") or None

        return {
            "intent": intent,
            "tickers": tickers,
            "weights": weights,
            "new_stock": (new_stock.strip().upper() if new_stock else None),
            "type": typ,
            "count": count,
            "symbol": (symbol.strip().upper() if symbol else None),
            "news_topic": news_topic
        }


# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
def ask_openai(question, context):
    # OpenAI chat completion function
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def fetch_newsapi():
    url = f"https://newsapi.org/v2/everything?q=finance OR stock OR crypto&sortBy=publishedAt&language=en&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"NewsAPI error {response.status_code}: {response.text}")

    articles = response.json().get("articles", [])[:5]
    return [f"{a['title']} - {a['source']['name']}\n{a['description']}" for a in articles]



def fetch_reddit_posts(subreddit_name="wallstreetbets", limit=5):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
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
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching news: {response.status_code} - {response.text}")
    
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
    data = yf.download(symbol, start=last_week - timedelta(days=2), end=today + timedelta(days=1))
    
    if data.empty:
        return f"No data found for {symbol}"
    
    # Get closing prices, adjust for market holidays/weekends
    def get_closest_close(date):
        # If market closed, find closest previous day with data
        while date not in data.index.date:
            date -= timedelta(days=1)
        return float(data.loc[data.index.date == date]['Close'].iloc[0])
    
    close_today = get_closest_close(today)
    close_yesterday = get_closest_close(yesterday)
    close_last_week = get_closest_close(last_week)
    
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
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.dropna()

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

    return "\n".join(output)



def should_buy_stock(tickers: list, weights: list, new_stock: str) -> str:

    if len(tickers) != len(weights):
        raise ValueError("Length of tickers and weights must match.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1.0")

    # Dates
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Fetch data for current portfolio and new stock
    all_tickers = tickers + [new_stock]
    data = yf.download(all_tickers, start=start_date, end=end_date)['Close'].dropna()

    if data.empty:
        return f"No historical data available for one or more tickers: {all_tickers}"

    # Daily returns
    returns = np.log(data / data.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Risk-free rate
    risk_free_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100

    # Helper: portfolio performance
    def portfolio_performance(w):
        w = np.array(w)  # Ensure numpy array
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = (port_return - risk_free_rate) / port_vol
        return port_return, port_vol, sharpe

    # Helper: negative Sharpe for minimization
    def neg_sharpe(w):
        return -portfolio_performance(w)[2]

    # Constraints & bounds
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in all_tickers)

    # Optimize weights for max Sharpe (WITH new stock)
    result_new = minimize(
        neg_sharpe,
        np.ones(len(all_tickers)) / len(all_tickers),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    new_weights = result_new.x

    # Current portfolio performance (WITHOUT new stock)
    curr_weights_padded = np.array(weights + [0])  # pad with zero weight for the new stock
    curr_return, curr_vol, curr_sharpe = portfolio_performance(curr_weights_padded)

    # New portfolio performance (WITH new stock optimized)
    new_return, new_vol, new_sharpe = portfolio_performance(new_weights)

    # Decide if buy is good
    decision = "BUY ✅" if new_sharpe > curr_sharpe else "DON'T BUY ❌"

    # Format output
    output = []
    output.append(f"=== Should You Buy {new_stock}? ===")
    output.append(f"Current Sharpe Ratio: {curr_sharpe:.2f}")
    output.append(f"New Sharpe Ratio (with optimization): {new_sharpe:.2f}")
    output.append(f"Decision: {decision}\n")

    if decision.startswith("BUY"):
        new_alloc_df = pd.DataFrame({
            'Ticker': all_tickers,
            'Weight': new_weights
        })
        output.append("Suggested New Weights:")
        output.append(new_alloc_df.to_string(index=False, float_format="{:.2%}".format))

    return "\n".join(output)



