"""Derived position figures: cost basis, holding period, P&L, benchmark excess.

Prices are stubbed, so every number below is checkable by hand. The point of
these is that the figures the critic will reason over are arithmetic, not
estimates.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

import positions
from positions import add_lot, get_lots, position_facts


@pytest.fixture
def pdb(tmp_path):
    return str(tmp_path / "pos.db")


@pytest.fixture
def prices(monkeypatch):
    """Install a deterministic price history: linear from `start` to `end`."""
    def _install(series: dict, days: int = 500):
        # Calendar-day index on purpose, so ago(N) lines up exactly with index
        # position N. Real markets skip weekends; this is arithmetic under test,
        # not a market simulation.
        idx = pd.date_range(end=pd.Timestamp(date.today()), periods=days, freq="D")
        frame = pd.DataFrame(
            {t: [s + (e - s) * i / (days - 1) for i in range(days)]
             for t, (s, e) in series.items()}, index=idx)
        monkeypatch.setattr(positions, "_close_frame", lambda tickers, start: frame)
        return frame
    return _install


def ago(days):
    return (date.today() - timedelta(days=days)).isoformat()


# ---------- storage ----------

def test_a_lot_round_trips(pdb):
    add_lot(pdb, "u", "pltr", ago(100), 100.0, 5)
    lots = get_lots(pdb, "u")
    assert len(lots) == 1
    assert lots[0].ticker == "PLTR", "tickers are normalised on the way in"
    assert lots[0].entry_price == 100.0 and lots[0].quantity == 5


def test_buying_the_same_stock_twice_is_two_lots(pdb):
    """One row per purchase: averaging in must not overwrite the first buy."""
    add_lot(pdb, "u", "PLTR", ago(300), 80.0, 10)
    add_lot(pdb, "u", "PLTR", ago(100), 120.0, 10)
    assert len(get_lots(pdb, "u")) == 2


def test_no_positions_yields_empty_facts(pdb):
    f = position_facts(pdb, "u")
    assert f["positions"] == {} and f["total_market_value"] == 0.0


# ---------- the derived arithmetic ----------

def test_cost_basis_is_quantity_weighted(pdb, prices):
    prices({"PLTR": (100.0, 200.0), "^GSPC": (1000.0, 1000.0)})
    add_lot(pdb, "u", "PLTR", ago(300), 100.0, 30)   # 3000
    add_lot(pdb, "u", "PLTR", ago(100), 200.0, 10)   # 2000
    p = position_facts(pdb, "u")["positions"]["PLTR"]
    assert p["quantity"] == 40
    assert p["cost_basis"] == pytest.approx(125.0)   # 5000 / 40, not 150
    assert p["lot_count"] == 2


def test_holding_period_runs_from_the_earliest_lot(pdb, prices):
    prices({"PLTR": (100.0, 100.0), "^GSPC": (1000.0, 1000.0)})
    add_lot(pdb, "u", "PLTR", ago(400), 100.0, 1)
    add_lot(pdb, "u", "PLTR", ago(10), 100.0, 1)
    p = position_facts(pdb, "u")["positions"]["PLTR"]
    assert p["holding_period_days"] == 400


@pytest.mark.parametrize("days,short", [(364, True), (365, False), (400, False)])
def test_short_term_lot_boundary(pdb, prices, days, short):
    """Under a year is a short-term lot, which changes what selling costs."""
    prices({"PLTR": (100.0, 100.0), "^GSPC": (1000.0, 1000.0)})
    add_lot(pdb, "u", "PLTR", ago(days), 100.0, 1)
    assert position_facts(pdb, "u")["positions"]["PLTR"]["short_term_lot"] is short


def test_unrealised_return_against_cost_basis(pdb, prices):
    prices({"PLTR": (100.0, 150.0), "^GSPC": (1000.0, 1000.0)})
    add_lot(pdb, "u", "PLTR", ago(200), 100.0, 10)
    p = position_facts(pdb, "u")["positions"]["PLTR"]
    assert p["current_price"] == pytest.approx(150.0)
    assert p["unrealized_pct"] == pytest.approx(0.5)
    assert p["unrealized_abs"] == pytest.approx(500.0)
    assert p["market_value"] == pytest.approx(1500.0)


def test_excess_over_the_benchmark_is_what_matters(pdb, prices):
    """Up 50% while the market doubled is underperformance, not a win."""
    prices({"PLTR": (100.0, 150.0), "^GSPC": (1000.0, 2000.0)})
    add_lot(pdb, "u", "PLTR", ago(499), 100.0, 1)
    p = position_facts(pdb, "u")["positions"]["PLTR"]
    assert p["unrealized_pct"] == pytest.approx(0.5, abs=0.01)
    assert p["benchmark_pct"] == pytest.approx(1.0, abs=0.01)
    assert p["excess_vs_benchmark"] == pytest.approx(-0.5, abs=0.02)


def test_drawdown_records_the_worst_point_since_entry(pdb, prices):
    """A position can end up green having been deeply underwater."""
    prices({"PLTR": (50.0, 200.0), "^GSPC": (1000.0, 1000.0)})
    add_lot(pdb, "u", "PLTR", ago(499), 100.0, 1)    # bought above the low
    p = position_facts(pdb, "u")["positions"]["PLTR"]
    assert p["unrealized_pct"] > 0
    assert p["drawdown_from_entry"] == pytest.approx(-0.5, abs=0.01)


def test_weights_are_computed_from_live_prices(pdb, prices):
    """The whole point: hand-entered weights go stale the moment prices move."""
    prices({"A": (100.0, 300.0), "B": (100.0, 100.0), "^GSPC": (1000.0, 1000.0)})
    add_lot(pdb, "u", "A", ago(200), 100.0, 1)
    add_lot(pdb, "u", "B", ago(200), 100.0, 1)
    w = position_facts(pdb, "u")["computed_weights"]
    assert w["A"] == pytest.approx(0.75)    # bought 50/50, now 75/25
    assert w["B"] == pytest.approx(0.25)


def test_a_ticker_with_no_price_data_degrades_quietly(pdb, prices):
    """A delisted or mistyped symbol must not take the other positions down."""
    prices({"PLTR": (100.0, 150.0), "^GSPC": (1000.0, 1000.0)})
    add_lot(pdb, "u", "PLTR", ago(200), 100.0, 1)
    add_lot(pdb, "u", "ZZZZ", ago(200), 10.0, 1)
    f = position_facts(pdb, "u")
    assert f["positions"]["ZZZZ"]["current_price"] is None
    assert f["positions"]["ZZZZ"]["cost_basis"] == 10.0      # still reported
    assert f["positions"]["PLTR"]["unrealized_pct"] == pytest.approx(0.5)


def test_a_failed_download_still_returns_the_lots(pdb, monkeypatch):
    def boom(tickers, start):
        raise RuntimeError("yahoo is down")
    monkeypatch.setattr(positions, "_close_frame", boom)
    add_lot(pdb, "u", "PLTR", ago(200), 100.0, 3)
    p = position_facts(pdb, "u")["positions"]["PLTR"]
    assert p["cost_basis"] == 100.0 and p["current_price"] is None
