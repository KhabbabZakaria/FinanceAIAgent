"""Critic agent: the deterministic half, and the guarantee that it never
takes the answer down with it."""
import pytest

import critic
from critic import Finding, checks, review

PROFILE = {
    "summary": "test",
    "horizon_years": 20,
    "risk_tolerance": "high",
    "max_single_position": 0.35,
    "max_sector_like_cluster": 0.60,
    "min_positions": 3,
    "target_fixed_income": 0.20,
    "max_speculative": 0.15,
    "speculative_tickers": {"BTC-USD", "PLTR"},
    "fixed_income_tickers": {"BND"},
    "cluster_tickers": {"AAPL", "MSFT", "NVDA"},
    "cluster_name": "big-tech",
}


def portfolio(weights, **extra):
    return {"kind": "rank_portfolio", "weights": weights, **extra}


def codes(facts, profile=PROFILE):
    return {f.code for f in checks(facts, profile)}


# ---------- the rules ----------

def test_oversized_position_is_flagged():
    assert "position_over_cap" in codes(portfolio({"AAPL": 0.5, "BND": 0.3, "KO": 0.2}))


def test_position_exactly_at_the_cap_is_not_flagged():
    """A boundary that fires one notch early trains the user to ignore it."""
    assert "position_over_cap" not in codes(portfolio({"AAPL": 0.35, "BND": 0.35, "KO": 0.30}))


def test_zero_weight_holdings_do_not_count_as_positions():
    """The optimiser routinely returns 0% legs; they are not real holdings."""
    facts = portfolio({"AAPL": 0.5, "BND": 0.5, "KO": 0.0, "MSFT": 0.0})
    assert "too_few_positions" in codes(facts)


def test_speculative_cap_sums_across_tickers():
    found = codes(portfolio({"PLTR": 0.10, "BTC-USD": 0.10, "BND": 0.5, "KO": 0.3}))
    assert "speculative_over_cap" in found


def test_speculative_under_cap_is_quiet():
    assert "speculative_over_cap" not in codes(
        portfolio({"PLTR": 0.05, "BND": 0.55, "KO": 0.40}))


def test_cluster_cap_sums_across_declared_cluster():
    assert "cluster_over_cap" in codes(
        portfolio({"AAPL": 0.25, "MSFT": 0.25, "NVDA": 0.25, "BND": 0.25}))


def test_fixed_income_shortfall_is_flagged():
    assert "fixed_income_short" in codes(portfolio({"AAPL": 0.34, "KO": 0.33, "IVV": 0.33}))


def test_fixed_income_check_is_off_when_target_is_zero():
    p = {**PROFILE, "target_fixed_income": 0.0}
    assert "fixed_income_short" not in codes(portfolio({"AAPL": 0.34, "KO": 0.33, "IVV": 0.33}), p)


def test_negligible_sharpe_gain_on_a_buy_is_flagged():
    """A BUY on +0.005 is noise; saying so is the critic's whole point."""
    facts = {"kind": "should_buy_stock", "decision": "BUY", "marginal_gain": 0.005,
             "suggested_weights": {"IVV": 0.34, "KO": 0.33, "NVDA": 0.33},
             "current_weights": {"IVV": 0.6, "KO": 0.4}}
    assert "marginal_gain_negligible" in codes(facts)


def test_a_real_sharpe_gain_is_not_flagged():
    facts = {"kind": "should_buy_stock", "decision": "BUY", "marginal_gain": 0.84,
             "suggested_weights": {"IVV": 0.34, "KO": 0.33, "NVDA": 0.33},
             "current_weights": {"IVV": 0.6, "KO": 0.4}}
    assert "marginal_gain_negligible" not in codes(facts)


def test_buy_findings_describe_the_suggested_allocation():
    """Mis-attributing a weight to the current portfolio misleads the user."""
    facts = {"kind": "should_buy_stock", "decision": "BUY", "marginal_gain": 0.5,
             "suggested_weights": {"KO": 0.9, "IVV": 0.1},
             "current_weights": {"IVV": 0.6, "KO": 0.4}}
    detail = next(f.detail for f in checks(facts, PROFILE) if f.code == "position_over_cap")
    assert "suggested new allocation" in detail


def test_findings_are_ordered_most_severe_first():
    facts = portfolio({"AAPL": 0.9, "KO": 0.1})
    sev = [f.severity for f in checks(facts, PROFILE)]
    assert sev == sorted(sev, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


# ---------- the guarantees ----------

def test_review_is_none_when_there_are_no_facts():
    assert review("q", "a plain string answer") is None


def test_review_is_none_for_a_clean_portfolio():
    # every rule satisfied: nothing over 35%, 4 positions, BND 30% clears the
    # 20% fixed-income target, no speculative names, big-tech only 30%
    facts = portfolio({"AAPL": 0.30, "BND": 0.30, "KO": 0.25, "IVV": 0.15})
    assert review("q", critic_answer(facts), PROFILE, use_model=False) is None


def test_review_falls_back_to_raw_findings_without_a_model(monkeypatch):
    facts = portfolio({"AAPL": 0.9, "BND": 0.1})
    out = review("q", critic_answer(facts), PROFILE, use_model=False)
    assert "single-position limit" in out and out.startswith("--- Worth knowing ---")


def test_a_dead_model_degrades_to_the_rule_engine(monkeypatch):
    """The critic is additive. No provider must mean fewer words, not no answer."""
    def boom(*a, **k):
        raise critic.AllProvidersFailed("nothing running")
    monkeypatch.setattr(critic, "chat", boom)
    out = review("q", critic_answer(portfolio({"AAPL": 0.9, "BND": 0.1})), PROFILE)
    assert "single-position limit" in out


def test_a_broken_critic_never_takes_the_answer_down(monkeypatch):
    monkeypatch.setattr(critic, "checks", lambda *a, **k: 1 / 0)
    assert review("q", critic_answer(portfolio({"AAPL": 0.9}))) is None


def test_stray_sentinel_tokens_are_stripped(monkeypatch):
    """Small models leak instruction words into the reply."""
    class R:
        text, provider, model, latency_ms = "KO is too large. NONE", "stub", "s", 1
    monkeypatch.setattr(critic, "chat", lambda *a, **k: R())
    out = review("q", critic_answer(portfolio({"AAPL": 0.9, "BND": 0.1})), PROFILE)
    assert "NONE" not in out and "KO is too large." in out


def critic_answer(facts):
    from helpers import Analysis
    return Analysis("the answer text", facts)


# ---------- findings only the record can supply ----------
#
# Measured: handing prior turns to the model as prose context produced
# near-identical output with and without them. What the record knows has to
# become a finding like any other, or it is decoration.

def turns(*specs):
    from datetime import datetime, timedelta
    from conversations import Turn
    now = datetime.now()
    return [Turn(now - timedelta(days=d), q, "should_buy_stock", tk, "...", f)
            for d, q, tk, f in specs]


def test_a_repeatedly_flagged_holding_is_itself_a_finding():
    h = turns((152, "bought BTC-USD", ["BTC-USD"], ["speculative_over_cap"]),
              (45, "still worth holding?", ["BTC-USD"], ["speculative_over_cap"]))
    codes = {f.code for f in critic.checks(portfolio({"BTC-USD": 0.3, "IVV": 0.7}), PROFILE, h)}
    assert "repeat_concern" in codes


def test_one_previous_mention_is_not_a_repeat():
    h = turns((20, "bought BTC-USD", ["BTC-USD"], ["speculative_over_cap"]))
    codes = {f.code for f in critic.checks(portfolio({"BTC-USD": 0.3, "IVV": 0.7}), PROFILE, h)}
    assert "repeat_concern" not in codes


def test_a_long_held_name_is_noted_even_when_never_flagged():
    h = turns((152, "bought BND", ["BND"], []))
    codes = {f.code for f in critic.checks(portfolio({"BND": 0.3, "IVV": 0.4, "KO": 0.3}), PROFILE, h)}
    assert "long_standing" in codes


def test_history_about_other_tickers_is_ignored():
    h = turns((152, "bought TSLA", ["TSLA"], ["speculative_over_cap"]),
              (45, "TSLA again", ["TSLA"], ["speculative_over_cap"]))
    codes = {f.code for f in critic.checks(portfolio({"BND": 0.3, "IVV": 0.4, "KO": 0.3}), PROFILE, h)}
    assert "repeat_concern" not in codes


def test_no_history_changes_nothing():
    facts = portfolio({"AAPL": 0.9, "BND": 0.1})
    assert ({f.code for f in critic.checks(facts, PROFILE)} ==
            {f.code for f in critic.checks(facts, PROFILE, [])})
