from config import *
from openai import OpenAI
import requests
import praw
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd



# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)



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




def ask_openai(question, context):
    # OpenAI chat completion function
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()



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

