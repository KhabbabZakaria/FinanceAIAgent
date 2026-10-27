"""What you actually hold, when you bought it, and what it has done since.

`portfolios` records target weights. That answers "how should this be split",
but not "I bought this in May and it has been falling" -- which needs an entry
date and an entry price. This module adds that, and derives the figures the
critic reasons over.

Nothing here is stored that can be computed. Cost basis, holding period,
unrealised return and the benchmark comparison are all derived at read time
from the lots plus live prices, so they can never go stale.

One row per *lot*, not per ticker: buying the same stock three times is three
rows, and the cost basis is their quantity-weighted average.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import logging
import sqlite3

import pandas as pd
import yfinance as yf

logger = logging.getLogger("financeai.positions")

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    user_id     TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    entry_date  DATE NOT NULL,
    entry_price REAL NOT NULL,
    quantity    REAL NOT NULL,
    notes       TEXT,
    PRIMARY KEY (user_id, ticker, entry_date)
)
"""

# Anything held under a year is a short-term lot in most tax regimes, which
# changes what selling costs. Deterministic, so the critic can say it plainly.
LONG_TERM_DAYS = 365
BENCHMARK = "^GSPC"


@dataclass
class Lot:
    ticker: str
    entry_date: date
    entry_price: float
    quantity: float
    notes: Optional[str] = None


def ensure_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def add_lot(db_path: str, user_id: str, ticker: str, entry_date: str,
            entry_price: float, quantity: float, notes: str = None) -> None:
    ensure_schema(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?)",
                 (user_id, ticker.upper(), entry_date, float(entry_price),
                  float(quantity), notes))
    conn.commit()
    conn.close()


def remove_lot(db_path: str, user_id: str, ticker: str, entry_date: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("DELETE FROM positions WHERE user_id=? AND ticker=? AND entry_date=?",
                 (user_id, ticker.upper(), entry_date))
    conn.commit()
    conn.close()


def get_lots(db_path: str, user_id: str) -> List[Lot]:
    ensure_schema(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    rows = conn.execute(
        "SELECT ticker, entry_date, entry_price, quantity, notes FROM positions "
        "WHERE user_id=? ORDER BY ticker, entry_date", (user_id,)).fetchall()
    conn.close()
    return [Lot(t, _as_date(d), p, q, n) for t, d, p, q, n in rows]


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _close_frame(tickers: List[str], start: date) -> pd.DataFrame:
    """One download for everything, indexed by date with ticker columns."""
    data = yf.download(sorted(set(tickers)), start=start.isoformat(),
                       progress=False, threads=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(sorted(set(tickers))[0])
    return data.dropna(axis=1, how="all")


def _price_on_or_before(series: pd.Series, when: date) -> Optional[float]:
    """Last close at or before `when`; None if the series starts later."""
    upto = series[series.index.date <= when].dropna()
    return float(upto.iloc[-1]) if len(upto) else None


def position_facts(db_path: str, user_id: str,
                   benchmark: str = BENCHMARK) -> Dict[str, Any]:
    """Derived figures for every open position. Never raises on a bad ticker."""
    lots = get_lots(db_path, user_id)
    if not lots:
        return {"kind": "positions", "positions": {}, "total_market_value": 0.0,
                "computed_weights": {}, "benchmark": benchmark}

    earliest = min(l.entry_date for l in lots)
    tickers = sorted({l.ticker for l in lots})
    try:
        frame = _close_frame(tickers + [benchmark], earliest)
    except Exception as e:                       # pragma: no cover - network
        logger.warning("Price download failed, returning lots only: %s", e)
        frame = pd.DataFrame()

    today = date.today()
    out: Dict[str, Any] = {}
    for ticker in tickers:
        mine = [l for l in lots if l.ticker == ticker]
        qty = sum(l.quantity for l in mine)
        cost = sum(l.entry_price * l.quantity for l in mine)
        basis = cost / qty if qty else 0.0
        first = min(l.entry_date for l in mine)
        held_days = (today - first).days

        series = frame[ticker] if ticker in getattr(frame, "columns", []) else None
        current = float(series.dropna().iloc[-1]) if series is not None and len(series.dropna()) else None

        entry: Dict[str, Any] = {
            "lots": [{"entry_date": l.entry_date.isoformat(), "entry_price": l.entry_price,
                      "quantity": l.quantity, "notes": l.notes} for l in mine],
            "lot_count": len(mine),
            "quantity": qty,
            "cost_basis": basis,
            "cost_total": cost,
            "first_entry": first.isoformat(),
            "holding_period_days": held_days,
            "short_term_lot": held_days < LONG_TERM_DAYS,
            "current_price": current,
        }

        if current is not None and basis:
            entry["market_value"] = current * qty
            entry["unrealized_abs"] = (current - basis) * qty
            entry["unrealized_pct"] = (current / basis) - 1.0
            # Falling is weak on its own. Falling while the market rose is the
            # signal worth acting on, so compare over the same window.
            bench = frame[benchmark] if benchmark in getattr(frame, "columns", []) else None
            if bench is not None:
                b_then = _price_on_or_before(bench, first)
                b_now = float(bench.dropna().iloc[-1]) if len(bench.dropna()) else None
                if b_then and b_now:
                    entry["benchmark_pct"] = (b_now / b_then) - 1.0
                    entry["excess_vs_benchmark"] = entry["unrealized_pct"] - entry["benchmark_pct"]
            # Worst close since entry: how much pain this position has caused.
            since = series[series.index.date >= first].dropna()
            if len(since):
                entry["low_since_entry"] = float(since.min())
                entry["drawdown_from_entry"] = (float(since.min()) / basis) - 1.0
        out[ticker] = entry

    total = sum(p.get("market_value") or 0.0 for p in out.values())
    weights = {t: (p.get("market_value") or 0.0) / total for t, p in out.items()} if total else {}

    return {"kind": "positions", "positions": out, "total_market_value": total,
            "computed_weights": weights, "benchmark": benchmark,
            "as_of": today.isoformat()}
