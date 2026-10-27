"""Critic agent: a second pass that reviews an answer against the user's profile.

It never recomputes anything. Agent A produces the numbers deterministically;
the critic reads those numbers plus the profile and says what a human adviser
would say about them.

The split inside the critic matters as much as the split between the agents:

    checks()    pure Python. Every constraint that can be expressed as a rule is
                evaluated here, so the findings themselves are exact, testable
                and identical every run.
    critique()  the model. Given findings it did not have to derive, it decides
                which matter for this question and writes them in context.

A model asked to critique a portfolio unaided produces generic hedging. Handed
concrete violations, it does the one thing it is good at: prioritising and
explaining. If it is unavailable the deterministic findings are rendered
directly, so the critic degrades to a plain rule engine rather than vanishing.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from llm import chat, AllProvidersFailed
from profile import USER_PROFILE

logger = logging.getLogger("financeai.critic")

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Finding:
    severity: str          # high | medium | low
    code: str              # stable identifier, for tests and metrics
    detail: str            # one sentence, already containing the numbers

    def __str__(self):
        return f"[{self.severity}] {self.detail}"


def _weights_under_review(facts: Dict[str, Any]) -> tuple[Dict[str, float], str]:
    """The allocation the user would hold if they acted on this answer, and a
    label naming it.

    The label is threaded into every finding because a suggested reallocation
    and a current portfolio are different things, and a small model handed
    "KO would be 58%" will otherwise attribute it to whatever the question was
    about.
    """
    if facts.get("kind") == "should_buy_stock":
        suggested = facts.get("suggested_weights")
        if suggested:
            return suggested, "the suggested new allocation"
        return facts.get("current_weights") or {}, "your current portfolio"
    return facts.get("weights") or {}, "your portfolio"


def history_checks(history: List[Any], facts: Dict[str, Any]) -> List[Finding]:
    """Findings that only the record can supply.

    Handing prior turns to the model as context does not reliably change what
    it writes -- measured, it produced near-identical output with and without.
    What the record knows has to become a finding like any other, or it is
    decoration.
    """
    found: List[Finding] = []
    if not history:
        return found

    weights, _ = _weights_under_review(facts)
    held = {t for t, w in weights.items() if w > 0.001}

    for ticker in sorted(held):
        mentions = [h for h in history if ticker in (h.tickers or [])]
        if not mentions:
            continue
        oldest = min(mentions, key=lambda h: h.asked_at)
        age = oldest.age_days()

        # Repeating a warning the user has already had, and ignored, is a
        # different message from raising it for the first time.
        repeats = sum(1 for h in mentions if h.findings)
        if repeats >= 2:
            found.append(Finding("high", "repeat_concern",
                                 f"{ticker} has been flagged {repeats} times before, "
                                 f"first {age} days ago, and is still outside your limits."))
        elif age >= 90:
            found.append(Finding("low", "long_standing",
                                 f"You first asked about {ticker} {age} days ago; "
                                 f"it has been in the portfolio across that period."))
    return found


def checks(facts: Dict[str, Any], profile: Dict[str, Any] = None,
           history: List[Any] = None) -> List[Finding]:
    """Every rule the profile can express, evaluated exactly."""
    profile = profile or USER_PROFILE
    weights, where = _weights_under_review(facts)
    found: List[Finding] = []
    if not weights:
        return found

    held = {t: w for t, w in weights.items() if w > 0.001}

    cap = profile.get("max_single_position")
    if cap is not None:
        for ticker, w in sorted(held.items(), key=lambda kv: -kv[1]):
            if w > cap:
                found.append(Finding("high", "position_over_cap",
                                     f"In {where}, {ticker} is {w:.0%}, above your "
                                     f"{cap:.0%} single-position limit."))

    floor = profile.get("min_positions")
    if floor is not None and len(held) < floor:
        found.append(Finding("medium", "too_few_positions",
                             f"In {where}, only {len(held)} position(s) carry any "
                             f"weight; your profile asks for at least {floor}."))

    spec = profile.get("speculative_tickers") or set()
    spec_cap = profile.get("max_speculative")
    spec_total = sum(w for t, w in held.items() if t in spec)
    if spec_cap is not None and spec_total > spec_cap:
        names = ", ".join(sorted(t for t in held if t in spec))
        found.append(Finding("high", "speculative_over_cap",
                             f"In {where}, speculative holdings ({names}) total "
                             f"{spec_total:.0%}, above your {spec_cap:.0%} limit."))

    cluster = profile.get("cluster_tickers") or set()
    cluster_cap = profile.get("max_sector_like_cluster")
    cluster_total = sum(w for t, w in held.items() if t in cluster)
    if cluster_cap is not None and cluster_total > cluster_cap:
        names = ", ".join(sorted(t for t in held if t in cluster))
        found.append(Finding("medium", "cluster_over_cap",
                             f"{profile.get('cluster_name', 'Clustered')} names ({names}) "
                             f"total {cluster_total:.0%} of {where}, above your "
                             f"{cluster_cap:.0%} limit."))

    fixed = profile.get("fixed_income_tickers") or set()
    target = profile.get("target_fixed_income")
    if target:
        actual = sum(w for t, w in held.items() if t in fixed)
        if actual + 1e-9 < target:
            found.append(Finding("medium", "fixed_income_short",
                                 f"In {where}, fixed income is {actual:.0%} against a "
                                 f"target of {target:.0%}."))

    # A BUY on a gain this small is noise, not signal.
    if facts.get("kind") == "should_buy_stock" and facts.get("decision") == "BUY":
        gain = facts.get("marginal_gain") or 0.0
        if gain < 0.05:
            found.append(Finding("medium", "marginal_gain_negligible",
                                 f"The Sharpe improvement is {gain:+.3f}, well inside the "
                                 f"error of a one-year estimate; this is close to a coin flip."))

    found.extend(history_checks(history, facts))
    found.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    return found


CRITIC_SYSTEM_PROMPT = (
    "You are a risk reviewer for a finance assistant. You are shown a user's "
    "profile, the figures behind an answer they were given, and a list of "
    "findings ALREADY verified by exact calculation.\n"
    "Write one short sentence per finding, most important first, in plain "
    "language, addressed to the user as 'you'. Say why each one matters for "
    "their situation.\n"
    "Rules: cover every finding given. Never recompute or dispute the figures, "
    "they are correct. Never introduce a number you were not shown. No generic "
    "advice such as 'diversify' or 'past performance does not guarantee "
    "results'. No preamble, no closing summary, no bullet markers."
)


def _render(findings: List[Finding]) -> str:
    return "\n".join(f"  - {f.detail}" for f in findings)


def critique(question: str, facts: Dict[str, Any], findings: List[Finding],
             profile: Dict[str, Any] = None, max_tokens: int = 160,
             history: List[Any] = None) -> Optional[str]:
    """Ask the model which findings matter here. Returns None if none do."""
    profile = profile or USER_PROFILE
    if not findings:
        return None

    numbers = {k: v for k, v in facts.items() if k != "per_asset_return"}
    past = ""
    if history:
        # One summarised line per turn, not raw transcript: a small model
        # handed a wall of prior conversation writes worse, not better.
        lines = "\n".join(f"  - {t.summarise()}" for t in history)
        past = (f"What this user has asked before:\n{lines}\n"
                "Mention it only if it changes what you would say.\n\n")

    user_msg = (
        f"User profile: {profile.get('summary', '')}\n"
        f"Horizon: {profile.get('horizon_years')} years. "
        f"Risk tolerance: {profile.get('risk_tolerance')}.\n\n"
        f"{past}"
        f"Their question: {question}\n\n"
        f"Figures behind the answer: {numbers}\n\n"
        f"Verified findings:\n{_render(findings)}\n\n"
        "Which of these matter here, and why?"
    )

    try:
        result = chat([{"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                       {"role": "user", "content": user_msg}],
                      max_tokens=max_tokens)
    except AllProvidersFailed as e:
        # The critic is additive. If no model is available the user still gets
        # the exact findings, just without the prioritisation.
        logger.warning("Critic model unavailable (%s); falling back to raw findings", e)
        return _render(findings)

    logger.info("Critique via %s in %dms", result.provider, result.latency_ms)
    text = " ".join(result.text.split())
    # Small models leak instruction tokens into the reply; drop them rather than
    # show the user the word NONE in the middle of a sentence.
    for stray in ("NONE.", "NONE"):
        text = text.replace(stray, "").strip()
    return text or _render(findings)


def review(question: str, answer: Any, profile: Dict[str, Any] = None,
           use_model: bool = True, history: List[Any] = None) -> Optional[str]:
    """Entry point: returns a caveat block to append to the answer, or None.

    Never raises. A critic that breaks must not take the answer down with it.
    """
    facts = getattr(answer, "facts", None)
    if not facts:
        return None                      # nothing structured to review

    try:
        findings = [f for f in checks(facts, profile, history)
                    if f.severity in ("high", "medium")]
        if not findings:
            return None
        body = (critique(question, facts, findings, profile, history=history)
                if use_model else _render(findings))
        if not body:
            return None
        review.last_finding_codes = [f.code for f in findings]
        return "--- Worth knowing ---\n" + body
    except Exception as e:
        logger.warning("Critic failed, answer returned unreviewed: %s", e)
        return None
