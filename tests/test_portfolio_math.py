"""Regression tests for the portfolio maths.

The headline bug: yfinance returns 'Close' columns sorted alphabetically, not in
the order they were requested, so weights were applied to the wrong stocks and
the BUY/DON'T BUY verdict was computed from a scrambled portfolio.
"""
import numpy as np
import pytest

import helpers
from conftest import close_frame


@pytest.fixture
def fake_market(monkeypatch):
    """Deterministic prices: KO flat-ish, NVDA strong, PLTR volatile-weak."""
    frame = close_frame({
        "IVV": (400.0, 0.0004),
        "KO": (60.0, 0.0001),
        "NVDA": (100.0, 0.0012),
        "PLTR": (20.0, -0.0003),
    })
    monkeypatch.setattr(helpers.yf, "download", lambda *a, **k: _wrap(frame))

    class _Hist:
        @staticmethod
        def history(period=None):
            import pandas as pd
            return pd.DataFrame({"Close": [4.2]})
    monkeypatch.setattr(helpers.yf, "Ticker", lambda s: _Hist)
    return frame


def _wrap(frame):
    import pandas as pd
    out = pd.concat({"Close": frame}, axis=1)
    return out


def test_weights_follow_their_own_tickers_not_column_order(fake_market):
    """The bug: data columns are alphabetical, weights were positional.

    'IVV 70%, KO 30%' plus NVDA must value IVV at 70%, whatever order yfinance
    hands the columns back in.
    """
    out = helpers.should_buy_stock(["PLTR", "IVV"], [0.9, 0.1], "NVDA")
    # Portfolio is 90% PLTR (a loser). If the weights had landed alphabetically
    # the 0.9 would sit on IVV and the reported Sharpe would be far higher.
    current = float(out.split("at your weights:")[1].split("\n")[0])
    assert current < 0.5, f"90% PLTR should give a poor Sharpe, got {current}"


def test_buy_verdict_compares_like_with_like(fake_market):
    """Old code compared user weights against an optimised portfolio, so the
    optimiser's re-weighting was credited to the new stock and it always said BUY."""
    out = helpers.should_buy_stock(["IVV", "KO"], [0.5, 0.5], "NVDA")
    without = float(out.split("without NVDA:")[1].split("\n")[0])
    with_ = float(out.split("with NVDA:")[1].split("\n")[0])
    assert with_ >= without, "adding an asset can never lower the achievable optimum"
    assert "Marginal gain" in out


def test_already_held_stock_is_rejected(fake_market):
    """Without this guard the ticker appears twice and is double counted."""
    out = helpers.should_buy_stock(["IVV", "KO"], [0.5, 0.5], "KO")
    assert "already in your portfolio" in out


def test_missing_ticker_is_reported_not_silently_dropped(monkeypatch):
    """One dead symbol used to be an all-NaN column; dropna() then removed every
    row and the metrics were computed on an empty frame."""
    frame = close_frame({"IVV": (400.0, 0.0004), "KO": (60.0, 0.0001)})
    frame["ZZZZ"] = np.nan
    monkeypatch.setattr(helpers.yf, "download", lambda *a, **k: _wrap(frame))
    out = helpers.analyze_portfolio(["IVV", "KO", "ZZZZ"], [0.4, 0.4, 0.2])
    assert "ZZZZ" in out and "No historical data" in out


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        helpers.analyze_portfolio(["IVV", "KO"], [0.5, 0.2])


def test_ticker_and_weight_lengths_must_match():
    with pytest.raises(ValueError, match="must match"):
        helpers.analyze_portfolio(["IVV", "KO"], [1.0])
