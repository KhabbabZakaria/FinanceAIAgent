"""The durable turn log, and retrieval keyed on tickers rather than similarity."""
import sqlite3
from datetime import datetime, timedelta

import pytest

import conversations as C


@pytest.fixture
def cdb(tmp_path):
    path = str(tmp_path / "c.db")
    C.ensure_schema(path)
    return path


def backdate(path, days, question, intent=None, tickers=None, findings=None):
    """Insert a turn as it would look on a server that has been up for months."""
    import json
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO conversations (user_id,asked_at,question,intent,tickers,answer,"
        "findings,provider,latency_ms) VALUES (?,?,?,?,?,?,?,?,?)",
        ("u", (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds"),
         question, intent, json.dumps(tickers or []), "...",
         json.dumps(findings or []), "ollama", 100))
    conn.commit()
    conn.close()


def test_a_turn_round_trips(cdb):
    C.record(cdb, "u", "should I buy NVDA?", intent="should_buy_stock",
             tickers=["nvda"], answer="BUY", findings=["position_over_cap"])
    t = C.recent(cdb, "u")[0]
    assert t.question == "should I buy NVDA?"
    assert t.tickers == ["NVDA"], "tickers are upper-cased for keyed lookup"
    assert t.findings == ["position_over_cap"]


def test_logging_never_breaks_the_answer(tmp_path):
    """A lost log line must not cost the user their reply."""
    C.record(str(tmp_path / "nonexistent-dir" / "x.db"), "u", "q")   # must not raise


def test_an_old_turn_is_retrieved_by_ticker(cdb):
    """The point of the whole thing: five months is not too old."""
    backdate(cdb, 152, "I just bought BTC-USD", "should_buy_stock", ["BTC-USD"])
    backdate(cdb, 1, "top large caps?", "top_performers", [])
    found = C.mentioning(cdb, "u", ["BTC-USD"])
    assert len(found) == 1 and found[0].age_days() == 152


def test_unrelated_tickers_are_not_retrieved(cdb):
    backdate(cdb, 10, "about BTC-USD", "stock_trend", ["BTC-USD"])
    assert C.mentioning(cdb, "u", ["AAPL"]) == []


def test_a_substring_ticker_does_not_false_match(cdb):
    """'BTC' must not match a turn that only mentioned 'BTC-USD'."""
    backdate(cdb, 5, "about BTC-USD", "stock_trend", ["BTC-USD"])
    assert C.mentioning(cdb, "u", ["BTC"]) == []


def test_history_combines_matched_and_recent_oldest_first(cdb):
    backdate(cdb, 152, "bought BTC-USD", "should_buy_stock", ["BTC-USD"])
    backdate(cdb, 3, "top large caps?", "top_performers", [])
    backdate(cdb, 0, "crypto news", "get_finance_news", [])
    turns = C.history_for(cdb, "u", ["BTC-USD"])
    assert [t.age_days() for t in turns] == sorted((t.age_days() for t in turns), reverse=True)
    assert any(t.age_days() == 152 for t in turns), "the old BTC turn must survive"


def test_history_is_deduplicated(cdb):
    """A turn that is both recent and ticker-matched appears once."""
    backdate(cdb, 0, "about BTC-USD", "stock_trend", ["BTC-USD"])
    assert len(C.history_for(cdb, "u", ["BTC-USD"])) == 1


def test_no_tickers_means_recent_context_only(cdb):
    """Enough recent turns that the old one falls outside the window -- with
    only two rows in the table, "the last two turns" would include it."""
    backdate(cdb, 200, "bought BTC-USD", "should_buy_stock", ["BTC-USD"])
    backdate(cdb, 3, "top large caps?", "top_performers", [])
    backdate(cdb, 1, "crypto news", "get_finance_news", [])
    backdate(cdb, 0, "forex news", "get_finance_news", [])
    turns = C.history_for(cdb, "u", [])
    assert turns, "recent context should still be returned"
    assert all(t.age_days() < 100 for t in turns)


def test_summarise_is_one_compact_line(cdb):
    """Raw transcript is noise; a small model garbles it."""
    t = C.Turn(datetime.now() - timedelta(days=5), "should I buy NVDA?" * 20,
               "should_buy_stock", ["NVDA"], "...", ["position_over_cap"])
    line = t.summarise()
    assert "\n" not in line and len(line) < 220
    assert "5d ago" in line and "position_over_cap" in line
