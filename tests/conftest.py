"""Shared fixtures.

Every test runs offline: no API keys, no network, no LLM. Anything that would
reach out (yfinance, requests, the provider chain) is stubbed per test, so the
suite is deterministic and runs in seconds.
"""
import os
import pathlib
import sqlite3
import sys
import tempfile

# Point the app at a throwaway database *before* config is imported, so importing
# app.py can never touch the developer's real portfolios.db.
_TMPDIR = tempfile.mkdtemp(prefix="fina-tests-")
os.environ.setdefault("DB_PATH", os.path.join(_TMPDIR, "test.db"))
os.environ.setdefault("ENABLE_MONITORING", "0")
os.environ.setdefault("LLM_PROVIDERS", "")          # no provider => no accidental calls

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def db(monkeypatch, tmp_path):
    """A fresh, empty database for one test."""
    import config
    path = str(tmp_path / "t.db")
    monkeypatch.setattr(config, "DB_PATH", path)
    import app
    monkeypatch.setattr(app, "DB_PATH", path)
    import monitoring_service
    monkeypatch.setattr(monitoring_service, "DB_PATH", path)

    # app.py runs init_db() at import time, so whether this database arrives
    # empty depends on whether some earlier test already imported app. Clear it
    # explicitly: every test starts from the same state regardless of order.
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS portfolios (user_id TEXT, ticker TEXT, "
                 "weight REAL, alert_threshold REAL DEFAULT 5.0, "
                 "PRIMARY KEY (user_id, ticker))")
    conn.execute("DELETE FROM portfolios")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def client(db, monkeypatch):
    """Flask test client with the database isolated and the LLM stubbed out."""
    import app
    monkeypatch.setattr(app, "DEFAULT_WATCHLIST", [])
    app.init_db()
    app.app.config.update(TESTING=True)
    app.limiter.enabled = False          # rate limiting has its own test module
    return app.app.test_client()


@pytest.fixture
def stub_classify(monkeypatch):
    """Force the router down a chosen branch without calling a model."""
    import helpers
    base = dict(helpers.DEFAULT_CLASSIFICATION)

    def _apply(**overrides):
        result = {**base, **overrides}
        monkeypatch.setattr(helpers.QuestionClassifier, "classify",
                            lambda self, q: result)
        return result
    return _apply


def close_frame(prices: dict, days: int = 400) -> pd.DataFrame:
    """A deterministic 'Close' frame shaped like yfinance's, columns sorted.

    yfinance returns columns alphabetically rather than in request order, which
    is the root of the weight-misalignment bug, so the helper reproduces it.
    """
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    data = {}
    for ticker in sorted(prices):
        start, drift = prices[ticker]
        data[ticker] = [start * (1 + drift) ** i for i in range(days)]
    return pd.DataFrame(data, index=index)
