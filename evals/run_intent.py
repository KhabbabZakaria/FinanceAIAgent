"""Intent + slot eval for the question classifier.

    python -m evals.run_intent                      # current provider chain
    python -m evals.run_intent --provider openai    # compare a hosted model
    python -m evals.run_intent --update-baseline    # accept the current scores

Unlike the pytest suite this calls a real model, so it needs a provider running
and is not part of CI. Run it before a release, or after touching the prompt.
Exit code is non-zero if accuracy regressed past the gate, so it can gate a
nightly job.
"""
import argparse
import json
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evals.golden_set import CASES, intents  # noqa: E402

BASELINE = pathlib.Path(__file__).parent / "baseline.json"
# How far accuracy may fall before the run fails, in percentage points. Small
# models are stochastic even at temperature 0, so the gate is not zero.
TOLERANCE_PP = 5.0


def slots_match(expected: Dict, got: Dict) -> Tuple[bool, List[str]]:
    """Compare only the slots the case actually asserts."""
    problems = []
    for key, want in expected.items():
        have = got.get(key)
        if key == "weights":
            ok = (len(want) == len(have or []) and
                  all(abs(a - b) < 0.02 for a, b in zip(want, have or [])))
        else:
            ok = want == have
        if not ok:
            problems.append(f"{key}: want {want!r} got {have!r}")
    return not problems, problems


def run(provider: Optional[str]):
    import config
    if provider:
        config.LLM_PROVIDERS = provider
        import llm
        llm.LLM_PROVIDERS = provider
        llm._clients.clear()

    import helpers
    clf = helpers.QuestionClassifier()

    rows = []
    for expected_intent, question, expected_slots, tags in CASES:
        started = time.monotonic()
        error = None
        try:
            got = clf.classify(question)
        except Exception as e:                     # a hard failure, not a wrong answer
            got, error = {}, f"{type(e).__name__}: {e}"
        latency_ms = int((time.monotonic() - started) * 1000)

        intent_ok = got.get("intent") == expected_intent
        slot_ok, slot_problems = slots_match(expected_slots, got) if got else (False, ["no output"])
        rows.append({
            "question": question, "tags": tags, "latency_ms": latency_ms,
            "expected": expected_intent, "got": got.get("intent"),
            "intent_ok": intent_ok, "slot_ok": slot_ok,
            "slot_problems": slot_problems, "error": error,
            "provider": got.get("_provider"),
        })
    return rows


def report(rows):
    graded = [r for r in rows if "known-gap" not in r["tags"]]
    gaps = [r for r in rows if "known-gap" in r["tags"]]

    intent_ok = sum(r["intent_ok"] for r in graded)
    slot_ok = sum(r["slot_ok"] for r in graded)
    failures = sum(bool(r["error"]) for r in rows)
    lat = sorted(r["latency_ms"] for r in rows)
    n = len(graded)

    print(f"\n{'='*66}\nINTENT EVAL  ({len(rows)} cases, {n} graded, {len(gaps)} known gaps)\n{'='*66}")
    print(f"  intent accuracy   {intent_ok:>3}/{n}  = {intent_ok/n:6.1%}")
    print(f"  slot accuracy     {slot_ok:>3}/{n}  = {slot_ok/n:6.1%}")
    print(f"  hard failures     {failures:>3}")
    print(f"  latency           p50 {statistics.median(lat)/1000:.1f}s   "
          f"p90 {lat[int(len(lat)*0.9)]/1000:.1f}s   max {lat[-1]/1000:.1f}s")

    print("\n  per intent:")
    for want in intents():
        sub = [r for r in graded if r["expected"] == want]
        if sub:
            hit = sum(r["intent_ok"] for r in sub)
            print(f"    {want:<20} {hit:>2}/{len(sub):<2} {hit/len(sub):>6.0%}")

    print("\n  per tag:")
    for tag in sorted({t for r in graded for t in r["tags"]}):
        sub = [r for r in graded if tag in r["tags"]]
        hit = sum(r["intent_ok"] for r in sub)
        print(f"    {tag:<20} {hit:>2}/{len(sub):<2} {hit/len(sub):>6.0%}")

    confusion = Counter((r["expected"], r["got"]) for r in graded if not r["intent_ok"])
    if confusion:
        print("\n  confusion (expected -> got):")
        for (want, got), count in confusion.most_common():
            print(f"    {want:<20} -> {str(got):<20} x{count}")

    wrong_slots = [r for r in graded if r["intent_ok"] and not r["slot_ok"]]
    if wrong_slots:
        print("\n  right intent, wrong slots:")
        for r in wrong_slots:
            print(f"    {r['question'][:52]:<54} {'; '.join(r['slot_problems'])[:60]}")

    if gaps:
        print("\n  known gaps (documented, not counted):")
        for r in gaps:
            mark = "now passes" if r["intent_ok"] else "still unsupported"
            print(f"    [{mark:<17}] {r['question'][:56]}")

    return {"intent_accuracy": intent_ok / n, "slot_accuracy": slot_ok / n,
            "hard_failures": failures, "n_graded": n,
            "p50_ms": statistics.median(lat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", help="override LLM_PROVIDERS, e.g. openai")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--json", help="write the per-case results here")
    args = ap.parse_args()

    rows = run(args.provider)
    scores = report(rows)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\n  per-case results -> {args.json}")

    if args.update_baseline:
        BASELINE.write_text(json.dumps(scores, indent=2))
        print(f"\n  baseline updated -> {BASELINE.name}")
        return 0

    if not BASELINE.exists():
        print("\n  no baseline yet; run with --update-baseline to set one")
        return 0

    old = json.loads(BASELINE.read_text())
    print(f"\n{'-'*66}\n  REGRESSION GATE (tolerance {TOLERANCE_PP}pp)")
    failed = False
    for metric in ("intent_accuracy", "slot_accuracy"):
        delta = (scores[metric] - old[metric]) * 100
        bad = delta < -TOLERANCE_PP
        failed |= bad
        print(f"    {metric:<18} {old[metric]:6.1%} -> {scores[metric]:6.1%}  "
              f"{delta:+5.1f}pp  {'FAIL' if bad else 'ok'}")
    if scores["hard_failures"] > old.get("hard_failures", 0):
        print(f"    hard_failures      {old.get('hard_failures',0)} -> {scores['hard_failures']}  FAIL")
        failed = True
    print(f"{'-'*66}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
