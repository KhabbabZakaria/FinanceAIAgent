"""Per-source isolation, the retry policy, and provenance.

The whole point of this layer is that one source failing costs that source's
contribution and nothing else, so most of these assert on what *survives*.
"""
import time

import pytest
import requests

import sources
from sources import SourceError, SourceResult, fetch_one, gather, provenance


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Keep the suite fast; the retry policy is what's under test, not the wait."""
    monkeypatch.setattr(sources, "BACKOFF_SECONDS", 0)


def raiser(exc, succeed_on=None):
    """A source that raises, optionally succeeding on the nth attempt."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if succeed_on and calls["n"] >= succeed_on:
            return ["recovered"]
        raise exc
    fn.calls = calls
    return fn


# ---------- the retry policy ----------

@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
def test_transient_status_is_retried(status):
    fn = raiser(SourceError("boom", status=status))
    r = fetch_one("X", fn)
    assert fn.calls["n"] == 2 and r.attempts == 2 and not r.ok


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_status_is_not_retried(status):
    """A bad key fails identically the second time; retrying only adds latency."""
    fn = raiser(SourceError("boom", status=status))
    r = fetch_one("X", fn)
    assert fn.calls["n"] == 1 and r.attempts == 1 and not r.ok


@pytest.mark.parametrize("exc", [requests.Timeout("slow"), requests.ConnectionError("refused")])
def test_network_errors_are_retried(exc):
    fn = raiser(exc)
    fetch_one("X", fn)
    assert fn.calls["n"] == 2


def test_a_retry_that_succeeds_returns_the_data():
    fn = raiser(SourceError("flaky", status=503), succeed_on=2)
    r = fetch_one("X", fn)
    assert r.ok and r.items == ["recovered"] and r.attempts == 2


def test_an_unknown_exception_is_not_retried_but_is_absorbed():
    fn = raiser(ValueError("something odd"))
    r = fetch_one("X", fn)
    assert fn.calls["n"] == 1
    assert not r.ok and "ValueError" in r.error


# ---------- isolation ----------

def test_a_failing_source_never_raises_to_the_caller():
    r = fetch_one("X", lambda: (_ for _ in ()).throw(RuntimeError("kaboom")))
    assert isinstance(r, SourceResult) and not r.ok


def test_one_dead_source_does_not_stop_the_others():
    results = gather([
        ("A", lambda: ["a1", "a2"]),
        ("B", lambda: (_ for _ in ()).throw(SourceError("down", status=500))),
        ("C", lambda: ["c1"]),
    ])
    assert [r.name for r in results] == ["A", "B", "C"]      # order preserved
    assert [r.ok for r in results] == [True, False, True]
    assert sum(len(r.items) for r in results) == 3


def test_a_source_returning_none_is_treated_as_empty_not_broken():
    r = fetch_one("X", lambda: None)
    assert r.ok and r.items == [] and not r.usable


def test_sources_are_fetched_concurrently():
    """Sequential costs the sum of the timeouts, which makes a generous timeout
    unaffordable. Three 0.4s sources must take ~0.4s, not ~1.2s."""
    slow = lambda: (time.sleep(0.4), ["x"])[1]
    started = time.monotonic()
    gather([("A", slow), ("B", slow), ("C", slow)])
    assert time.monotonic() - started < 0.9


def test_gather_of_nothing_is_empty():
    assert gather([]) == []


# ---------- provenance ----------

def test_provenance_names_what_was_used_and_what_failed():
    line = provenance([
        SourceResult("NewsAPI", ["a"]),
        SourceResult("Reddit", [], ok=False, error="boom"),
        SourceResult("Finnhub", ["c"]),
    ])
    assert "Sources used: NewsAPI, Finnhub" in line
    assert "unavailable: Reddit" in line


def test_provenance_separates_empty_from_broken():
    """'returned nothing' and 'was down' are different facts about an answer."""
    line = provenance([
        SourceResult("NewsAPI", []),                       # ok, just nothing today
        SourceResult("Reddit", [], ok=False, error="503"),
    ])
    assert "returned nothing: NewsAPI" in line and "unavailable: Reddit" in line


def test_provenance_when_everything_is_down():
    line = provenance([SourceResult("A", [], ok=False), SourceResult("B", [], ok=False)])
    assert "Sources used: none" in line


def test_usable_requires_both_ok_and_items():
    assert SourceResult("A", ["x"]).usable
    assert not SourceResult("A", []).usable
    assert not SourceResult("A", ["x"], ok=False).usable
