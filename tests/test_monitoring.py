"""Regression tests for the background monitor.

Three of these cover bugs that made a feature silently never fire: the
price_history seeding deadlock, the ETF price field, and the company-news
signature that raised TypeError into a bare `except: pass`.
"""
import sqlite3
import types

import pandas as pd
import pytest

import monitoring_service as M


def _seed_watchlist(path, tickers):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS portfolios (user_id TEXT, ticker TEXT, "
                 "weight REAL, alert_threshold REAL DEFAULT 5.0, PRIMARY KEY (user_id, ticker))")
    conn.execute("CREATE TABLE IF NOT EXISTS price_history (ticker TEXT, price REAL, "
                 "volume INTEGER, timestamp DATETIME, PRIMARY KEY (ticker, timestamp))")
    conn.executemany("INSERT OR REPLACE INTO portfolios VALUES (?,?,?,?)",
                     [("u", t, 1.0 / len(tickers), thr) for t, thr in tickers])
    conn.commit(); conn.close()


@pytest.fixture
def monitor(db, monkeypatch):
    """A StockMonitor on an isolated DB with email captured, not sent."""
    sent = []
    monkeypatch.setattr(M.StockMonitor, "send_email",
                        lambda self, to, subject, body: sent.append((to, subject, body)))
    m = M.StockMonitor()
    m.db_path = db
    m.sent = sent
    return m


def _ticker_returning(info):
    return lambda symbol: types.SimpleNamespace(
        info=info, history=lambda period=None: pd.DataFrame({"Volume": [1, 1, 1]}))


def prices(path):
    conn = sqlite3.connect(path)
    out = conn.execute("SELECT ticker, price FROM price_history").fetchall()
    conn.close()
    return out


def test_first_pass_records_a_baseline(monitor, db, monkeypatch):
    """The bug: `if not old_data: continue` sat above the INSERT, so the table
    was never seeded and no comparison was ever possible."""
    _seed_watchlist(db, [("AAPL", 5.0)])
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({"currentPrice": 100.0}))
    monitor.check_price_drops()
    assert prices(db) == [("AAPL", 100.0)]
    assert monitor.sent == [], "no alert on the very first observation"


def test_etf_price_is_read_from_regular_market_price(monitor, db, monkeypatch):
    """ETFs and crypto leave currentPrice empty; reading only that field
    silently skipped every non-stock holding."""
    _seed_watchlist(db, [("IVV", 5.0)])
    monkeypatch.setattr(M.yf, "Ticker",
                        _ticker_returning({"currentPrice": None, "regularMarketPrice": 768.9}))
    monitor.check_price_drops()
    assert prices(db) == [("IVV", 768.9)]


def test_a_drop_past_the_threshold_alerts(monitor, db, monkeypatch):
    _seed_watchlist(db, [("AAPL", 5.0)])
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({"currentPrice": 100.0}))
    monitor.check_price_drops()                       # baseline 100
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({"currentPrice": 90.0}))
    monitor.check_price_drops()                       # -10%, threshold 5
    assert len(monitor.sent) == 1
    assert "Price Alert" in monitor.sent[0][1]


def test_a_drop_under_the_threshold_stays_quiet(monitor, db, monkeypatch):
    _seed_watchlist(db, [("AAPL", 5.0)])
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({"currentPrice": 100.0}))
    monitor.check_price_drops()
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({"currentPrice": 97.0}))
    monitor.check_price_drops()                       # -3%, under 5
    assert monitor.sent == []


def test_a_rise_never_alerts(monitor, db, monkeypatch):
    _seed_watchlist(db, [("AAPL", 5.0)])
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({"currentPrice": 100.0}))
    monitor.check_price_drops()
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({"currentPrice": 150.0}))
    monitor.check_price_drops()
    assert monitor.sent == []


def test_missing_price_is_skipped_not_treated_as_a_total_crash(monitor, db, monkeypatch):
    """`.get('currentPrice', 0)` made an unavailable price look like -100%."""
    _seed_watchlist(db, [("AAPL", 5.0)])
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({"currentPrice": 100.0}))
    monitor.check_price_drops()
    monkeypatch.setattr(M.yf, "Ticker", _ticker_returning({}))
    monitor.check_price_drops()
    assert monitor.sent == []
    assert prices(db) == [("AAPL", 100.0)], "a bogus 0 must not be stored"


def test_company_news_is_called_with_a_symbol(monitor, db, monkeypatch):
    """get_finnhub_news never accepted `symbol`; the TypeError was swallowed by
    `except: pass`, so breaking-news alerts never fired once."""
    _seed_watchlist(db, [("AAPL", 5.0)])
    calls = []

    def fake(symbol, hours=1):
        calls.append((symbol, hours))
        return [{"headline": "something happened"}]

    monkeypatch.setattr(M, "get_finnhub_company_news", fake)
    monitor.check_breaking_news()
    assert calls == [("AAPL", M.NEWS_ALERT_WINDOW_HOURS)]
    assert len(monitor.sent) == 1 and "Breaking News" in monitor.sent[0][1]


def test_a_failing_news_source_does_not_stop_the_other_tickers(monitor, db, monkeypatch):
    _seed_watchlist(db, [("AAPL", 5.0), ("MSFT", 5.0)])

    def flaky(symbol, hours=1):
        if symbol == "AAPL":
            raise RuntimeError("finnhub 429")
        return [{"headline": "msft news"}]

    monkeypatch.setattr(M, "get_finnhub_company_news", flaky)
    monitor.check_breaking_news()
    assert len(monitor.sent) == 1
    assert "MSFT" in monitor.sent[0][2] and "AAPL" not in monitor.sent[0][2]


def test_unknown_alert_type_is_refused_cleanly(monitor):
    """subject/body were only bound inside the if-branches, so an unexpected
    type raised NameError."""
    monitor.send_alerts([{"ticker": "AAPL"}], "not_a_real_alert_type")
    assert monitor.sent == []
