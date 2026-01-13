from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import yfinance as yf
import sqlite3
import yagmail
from helpers import fetch_newsapi, get_finnhub_news
from config import *

class StockMonitor:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.db_path = 'portfolios.db'
        self.setup_database()
        
    def setup_database(self):
        """Create tables for portfolios and price history"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # User portfolios
        c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                     (user_id TEXT, ticker TEXT, weight REAL, 
                      alert_threshold REAL DEFAULT 5.0, PRIMARY KEY (user_id, ticker))''')
        
        # Price history (for change detection)
        c.execute('''CREATE TABLE IF NOT EXISTS price_history
                     (ticker TEXT, price REAL, volume INTEGER, 
                      timestamp DATETIME, PRIMARY KEY (ticker, timestamp))''')
        
        conn.commit()
        conn.close()
    
    def check_price_drops(self):
        """Check all monitored stocks for significant drops"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get unique tickers being monitored
        c.execute('SELECT DISTINCT ticker, alert_threshold FROM portfolios')
        stocks = c.fetchall()
        
        alerts = []
        for ticker, threshold in stocks:
            # Get price from 1 day ago
            c.execute('''SELECT price FROM price_history 
                        WHERE ticker=? AND timestamp > datetime('now', '-1 day')
                        ORDER BY timestamp ASC LIMIT 1''', (ticker,))
            old_data = c.fetchone()
            
            if not old_data:
                continue
                
            old_price = old_data[0]
            
            # Get current price
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('currentPrice', 0)
            
            # Calculate drop percentage
            if old_price > 0:
                drop_pct = ((old_price - current_price) / old_price) * 100
                
                if drop_pct >= threshold:
                    alerts.append({
                        'ticker': ticker,
                        'old_price': old_price,
                        'current_price': current_price,
                        'drop_pct': drop_pct
                    })
            
            # Save current price
            c.execute('INSERT OR REPLACE INTO price_history VALUES (?, ?, ?, datetime("now"))',
                     (ticker, current_price, 0))
        
        conn.commit()
        conn.close()
        
        if alerts:
            self.send_alerts(alerts, 'price_drop')
    
    def check_unusual_volume(self):
        """Detect unusual trading volume (>2x average)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT DISTINCT ticker FROM portfolios')
        tickers = [row[0] for row in c.fetchall()]
        
        alerts = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='10d')
            
            if len(hist) < 2:
                continue
                
            avg_volume = hist['Volume'][:-1].mean()
            current_volume = hist['Volume'].iloc[-1]
            
            if current_volume > avg_volume * 2:
                alerts.append({
                    'ticker': ticker,
                    'avg_volume': avg_volume,
                    'current_volume': current_volume,
                    'multiplier': current_volume / avg_volume
                })
        
        conn.close()
        
        if alerts:
            self.send_alerts(alerts, 'unusual_volume')
    
    def check_breaking_news(self):
        """Check for breaking news about monitored stocks"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT DISTINCT ticker FROM portfolios')
        tickers = [row[0] for row in c.fetchall()]
        
        # Check news from last hour
        news_alerts = []
        for ticker in tickers:
            try:
                news = get_finnhub_news(symbol=ticker, hours=1)
                if news:
                    news_alerts.append({
                        'ticker': ticker,
                        'news_count': len(news),
                        'headlines': [item['headline'] for item in news[:3]]
                    })
            except:
                pass
        
        conn.close()
        
        if news_alerts:
            self.send_alerts(news_alerts, 'breaking_news')
    
    def generate_weekly_report(self):
        """Generate comprehensive portfolio health report"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT user_id, ticker, weight FROM portfolios')
        portfolios = {}
        
        for user_id, ticker, weight in c.fetchall():
            if user_id not in portfolios:
                portfolios[user_id] = []
            portfolios[user_id].append((ticker, weight))
        
        conn.close()
        
        for user_id, holdings in portfolios.items():
            report = self.create_portfolio_report(holdings)
            self.send_email(user_id, 'Weekly Portfolio Report', report)
    
    def create_portfolio_report(self, holdings):
        """Generate detailed portfolio analysis"""
        report = "📊 WEEKLY PORTFOLIO HEALTH REPORT\n\n"
        
        total_return = 0
        for ticker, weight in holdings:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='7d')
            
            if len(hist) >= 2:
                week_return = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]) * 100
                total_return += week_return * weight
                
                report += f"{ticker}: {week_return:.2f}% (Weight: {weight*100:.1f}%)\n"
        
        report += f"\n💰 Total Portfolio Return: {total_return:.2f}%\n"
        report += f"📅 Report Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        
        return report
    
    def send_alerts(self, alerts, alert_type):
        """Send alerts via email"""
        if alert_type == 'price_drop':
            subject = "🚨 Stock Price Alert"
            body = "Price drops detected:\n\n"
            for alert in alerts:
                body += f"{alert['ticker']}: -{alert['drop_pct']:.2f}% "
                body += f"(${alert['old_price']:.2f} → ${alert['current_price']:.2f})\n"
        
        elif alert_type == 'unusual_volume':
            subject = "📈 Unusual Volume Alert"
            body = "Unusual trading volume detected:\n\n"
            for alert in alerts:
                body += f"{alert['ticker']}: {alert['multiplier']:.1f}x average volume\n"
        
        elif alert_type == 'breaking_news':
            subject = "📰 Breaking News Alert"
            body = "Breaking news for your stocks:\n\n"
            for alert in alerts:
                body += f"\n{alert['ticker']} ({alert['news_count']} articles):\n"
                for headline in alert['headlines']:
                    body += f"  • {headline}\n"
        
        self.send_email('user@example.com', subject, body)
    
    def send_email(self, to_email, subject, body):
        """Send email notification"""
        try:
            yag = yagmail.SMTP(ALERT_EMAIL, ALERT_EMAIL_PASSWORD)
            yag.send(to=to_email, subject=subject, contents=body)
            print(f"✅ Alert sent: {subject}")
        except Exception as e:
            print(f"❌ Email failed: {e}")
    
    def start(self):
        """Start all monitoring jobs"""
        # Check prices every 15 minutes during market hours
        self.scheduler.add_job(self.check_price_drops, 'interval', minutes=15)
        
        # Check volume every 30 minutes
        self.scheduler.add_job(self.check_unusual_volume, 'interval', minutes=30)
        
        # Check news every hour
        self.scheduler.add_job(self.check_breaking_news, 'interval', hours=1)
        
        # Weekly report (every Monday at 9am)
        self.scheduler.add_job(self.generate_weekly_report, 'cron', 
                              day_of_week='mon', hour=9)
        
        self.scheduler.start()
        print("🤖 Monitoring service started!")