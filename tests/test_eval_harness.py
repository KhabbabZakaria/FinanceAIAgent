"""The eval harness has deterministic parts of its own, and a dataset that can rot.

These run in CI (no model needed) and catch the failure mode where the golden
set silently stops measuring what it claims to.
"""
import pytest

from evals.golden_set import CASES
from evals.run_intent import slots_match

VALID_INTENTS = {"rank_portfolio", "should_buy_stock", "top_performers",
                 "stock_trend", "get_finance_news", "other"}
VALID_SLOTS = {"tickers", "weights", "new_stock", "symbol", "type",
               "count", "news_topic"}


# ---------- the comparison logic ----------

def test_only_asserted_slots_are_checked():
    """A case that says nothing about weights must not fail on weights."""
    ok, problems = slots_match({"symbol": "AAPL"},
                               {"symbol": "AAPL", "weights": [0.9]})
    assert ok and problems == []


def test_mismatched_slot_is_reported_with_both_values():
    ok, problems = slots_match({"symbol": "AAPL"}, {"symbol": "MSFT"})
    assert not ok
    assert "want 'AAPL'" in problems[0] and "got 'MSFT'" in problems[0]


def test_weights_compare_within_tolerance():
    """Normalisation leaves tiny float drift; 1/3 splits must not fail."""
    ok, _ = slots_match({"weights": [0.3333, 0.3333, 0.3334]},
                        {"weights": [1 / 3, 1 / 3, 1 / 3]})
    assert ok


def test_weights_of_different_length_fail():
    ok, _ = slots_match({"weights": [0.5, 0.5]}, {"weights": [1.0]})
    assert not ok


def test_missing_slot_fails_rather_than_passing_silently():
    ok, _ = slots_match({"new_stock": "NVDA"}, {})
    assert not ok


def test_empty_expectation_means_must_be_empty():
    assert slots_match({"tickers": []}, {"tickers": []})[0]
    assert not slots_match({"tickers": []}, {"tickers": ["AAPL"]})[0]


# ---------- dataset integrity ----------

def test_every_label_is_a_real_intent():
    bad = {c[0] for c in CASES} - VALID_INTENTS
    assert not bad, f"golden set labels an intent the router has no branch for: {bad}"


def test_every_asserted_slot_is_a_real_slot():
    """Catches a typo like 'ticker' that would silently never be checked."""
    for intent, question, slots, _ in CASES:
        bad = set(slots) - VALID_SLOTS
        assert not bad, f"{question!r} asserts unknown slot(s) {bad}"


def test_no_duplicate_questions():
    seen = [c[1] for c in CASES]
    dupes = {q for q in seen if seen.count(q) > 1}
    assert not dupes, f"duplicate cases inflate the score: {dupes}"


def test_every_case_is_tagged():
    for intent, question, _, tags in CASES:
        assert tags, f"{question!r} has no tags, so failures can't be grouped"


def test_every_intent_has_several_cases():
    """A single example per intent gives per-intent accuracy no resolution."""
    for intent in VALID_INTENTS:
        n = sum(1 for c in CASES if c[0] == intent)
        assert n >= 5, f"{intent} has only {n} cases"


def test_the_set_is_not_all_easy():
    """A set everything passes measures nothing. Keep real difficulty in it."""
    hard = [c for c in CASES
            if {"paraphrase", "company-name", "adversarial", "malformed"} & set(c[3])]
    assert len(hard) / len(CASES) > 0.5, "golden set has drifted towards easy cases"


def test_weights_in_expectations_sum_to_one_where_given():
    """Guards the labels themselves: a mislabelled split would fail the model
    for the labeller's mistake."""
    for intent, question, slots, tags in CASES:
        w = slots.get("weights")
        if w and "malformed" not in tags:
            assert sum(w) == pytest.approx(1.0, abs=0.01), f"{question!r} labels sum to {sum(w)}"
