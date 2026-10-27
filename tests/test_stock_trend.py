"""Regression tests for get_stock_trend.

The bug: the walk-back to the previous trading day had no floor, so a long
market closure at the start of the window looped forever and hung the worker.
"""
import signal

import pandas as pd
import pytest

import helpers


def _frame(dates, price=100.0):
    """yfinance shape: MultiIndex columns even for a single ticker."""
    idx = pd.to_datetime(dates)
    df = pd.DataFrame({("Close", "X"): [price + i for i in range(len(idx))],
                       ("Open", "X"): [price for _ in idx]}, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class _Timeout(Exception):
    pass


def test_walk_back_terminates_when_window_has_no_earlier_data(monkeypatch):
    """Only two trading days, both AFTER the date being looked up.

    The old loop decremented the date forever. Guarded with SIGALRM so a
    regression fails the test instead of hanging the whole suite.
    """
    today = pd.Timestamp.today().normalize()
    monkeypatch.setattr(helpers.yf, "download",
                        lambda *a, **k: _frame([today - pd.Timedelta(days=1), today]))

    def _boom(signum, frame):
        raise _Timeout("get_stock_trend did not terminate")

    signal.signal(signal.SIGALRM, _boom)
    signal.alarm(10)
    try:
        out = helpers.get_stock_trend("X")
    finally:
        signal.alarm(0)
    assert "Not enough trading-day data" in out


def test_empty_download_is_reported(monkeypatch):
    monkeypatch.setattr(helpers.yf, "download", lambda *a, **k: pd.DataFrame())
    assert "No data found" in helpers.get_stock_trend("ZZZZ")


def test_multiindex_columns_are_flattened(monkeypatch):
    """float(Series) on a single-ticker MultiIndex frame is deprecated in
    pandas 2 and removed in 3; the Close series must be reduced to 1-D."""
    today = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=today, periods=12)
    monkeypatch.setattr(helpers.yf, "download", lambda *a, **k: _frame(dates))
    out = helpers.get_stock_trend("X")
    assert "Close Price Today" in out
    assert "Trend over last week" in out
