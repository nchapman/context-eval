"""Pre-flight gate for evaluation runs.

Dry-runs question assembly for every (dataset x budget x strategy) leg and
prints the plan: runs, questions, expected LLM calls, wall-clock and cost
estimates. Hard-fails when a leg exceeds its ceiling unless --allow is
passed. Born from the LoCoMo runaway (uncapped QA -> 109-203 questions
per run, 10h+ leg discovered 70 minutes in).

Usage:
  uv run python scripts/preflight.py --leg swe
  uv run python scripts/preflight.py --leg locomo --allow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from context_eval.datasets import read_jsonl  # noqa: E402
from context_eval.run_eval import build_questions  # noqa: E402

# Crude but explicit forward model, calibrated on the smoke rounds:
# - answering: 1 LLM call/question + tool steps (memory arms ~1.5x)
# - judging:   1 call/question
# - compaction: flat ~raw/budget calls, lcm ~raw/(0.25*budget)*1.3,
#               representation ~raw/2048 (amortized delta-triggered merges),
#               everything else 0
CALLS_PER_SEC_PER_WORKER = 0.35  # sequential-in-worker, flash latency
WORKERS = 3
USD_PER_1K_CALL_TOKENS = 0.000118  # flash fp8 blended, cache-aware ~half
AVG_CALL_TOKENS = 4000

CEILINGS = {  # questions per leg, wall-clock minutes per leg
    "questions": 3_000,
    "minutes": 90,
}

LEGS = {
    "swe": dict(path="data/swe_smoke.jsonl", budgets=(64_000, 128_000), samples=22,
                strategies=8, needles=4, max_q=0),
    "locomo": dict(path="data/locomo.jsonl", budgets=(8_000, 16_000), samples=10,
                   strategies=8, needles=4, max_q=16),
    "pi": dict(path="data/pi_sessions.jsonl", budgets=(128_000, 256_000), samples=3,
               strategies=8, needles=4, max_q=0),
    "thinking": dict(path="data/thinking_drop.jsonl", budgets=(128_000,), samples=1,
                     strategies=8, needles=4, max_q=0),
    # 10x-length stress leg: one mega-session (built by scripts/mega_sessions.py),
    # 3 sawtooth arms, 2 seeds run sequentially outside the suite. NOTE:
    # the forward model below does not price sawtooth extraction (~raw/(budget-16K)
    # calls per run, ~110 here) -- add it mentally; it is why this leg needs --allow.
    "mega": dict(path="data/pi_mega.jsonl", budgets=(64_000,), samples=2,
                 strategies=3, needles=4, max_q=8),
}


def plan(leg: str) -> dict:
    cfg = LEGS[leg]
    samples = read_jsonl(Path(cfg["path"]))[: cfg["samples"]]
    questions = 0
    raw_tokens = 0
    for s in samples:
        from context_eval.tokens import CharEstimator

        _, qs = build_questions(s, needles=cfg["needles"], seed=0,
                                max_q=cfg["max_q"], skip_locomo=False)
        questions += len(qs)
        raw_tokens += sum(CharEstimator().count_text(m.content) for m in s.messages)
    runs = cfg["samples"] * cfg["strategies"] * len(cfg["budgets"])
    budget = cfg["budgets"][0]
    # memory-arm compaction calls, worst budget (flat + lcm + representation
    # at flat's cadence)
    compaction = 2 * raw_tokens / budget + raw_tokens / (0.25 * budget) * 1.3
    answering = questions * cfg["strategies"] * 1.5
    judging = questions * cfg["strategies"]
    calls = answering + judging + compaction * 2  # both budgets crudely
    minutes = calls / CALLS_PER_SEC_PER_WORKER / WORKERS / 60
    cost = calls * AVG_CALL_TOKENS / 1000 * USD_PER_1K_CALL_TOKENS
    return dict(runs=runs, questions=questions * cfg["strategies"], calls=int(calls),
                minutes=minutes, cost=cost)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leg", choices=LEGS, required=True)
    p.add_argument("--allow", action="store_true", help="run despite ceiling breach")
    args = p.parse_args(argv)

    info = plan(args.leg)
    print(f"leg={args.leg}: {info['runs']} runs, {info['questions']} questions, "
          f"~{info['calls']:,} LLM calls, ~{info['minutes']:.0f} min at {WORKERS} workers, "
          f"~${info['cost']:.2f}")

    breaches = []
    if info["questions"] > CEILINGS["questions"]:
        breaches.append(f"questions {info['questions']} > {CEILINGS['questions']}")
    if info["minutes"] > CEILINGS["minutes"]:
        breaches.append(f"est {info['minutes']:.0f} min > {CEILINGS['minutes']} min")
    if breaches and not args.allow:
        print("CEILING BREACH: " + "; ".join(breaches))
        print("Fix the leg config or pass --allow with an explicit reason in the run log.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
