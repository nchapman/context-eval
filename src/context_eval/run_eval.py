"""Eval runner CLI.

Runs (strategy x budget) over replay samples with needle questions, writes
one RunResult JSONL line per run, prints a summary table.

Example:
  uv run python -m context_eval.run_eval --path data/swe_smoke.jsonl \
      --strategies full-context,sliding-window,arc,lcm --budgets 16000,32000 \
      --samples 4 --needles 4 --out results/swe.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from context_eval.datasets.base import read_jsonl
from context_eval.driver import SYSTEM_PROMPT, run_sample
from context_eval.env import load_env
from context_eval.provider import OpenRouterClient
from context_eval.questions import (
    CONTINUITY_QUESTION,
    Question,
    inject_needles,
    inject_topics,
)
from context_eval.strategies import REGISTRY, build
from context_eval.tokens import default_counter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("data/swe_smoke.jsonl"))
    parser.add_argument("--strategies", default="full-context,sliding-window")
    parser.add_argument("--budgets", default="16000,32000")
    parser.add_argument("--samples", type=int, default=4, help="samples to evaluate")
    parser.add_argument("--needles", type=int, default=4, help="needles per sample")
    parser.add_argument("--topics", type=int, default=0,
                        help="topic markers per sample; adds one uncued continuity question")
    parser.add_argument("--max-questions", type=int, default=0, help="cap per sample (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default=None,
                        help="process shard i of n as 'i/n' (0-based); enables "
                             "parallel workers on independent samples")
    parser.add_argument("--out", type=Path, default=Path("results/run.jsonl"))
    parser.add_argument("--skip-locomo-qa", action="store_true")
    parser.add_argument("--trace-qa", type=Path, default=None,
                        help="trace-QA cache (scripts/trace_qa.py output)")
    parser.add_argument("--needle-position", choices=["random", "mid"],
                        default="random",
                        help="mid: plant needles outside head/tail citation "
                             "previews (recall-tool recovery required)")
    parser.add_argument("--probes", default="",
                        help="long-horizon probes: comma list from "
                             "currency,next_step (currency reuses the seeded "
                             "continuity topics)")
    parser.add_argument("--recall-instruct", action="store_true",
                        help="append an explicit §-citation recall instruction "
                             "to the system prompt (ARC discoverability test)")
    return parser.parse_args(argv)


def build_questions(
    sample, needles: int, seed: int, max_q: int, skip_locomo: bool, topics: int = 0,
    trace_qa: list | None = None, needle_position: str = "random",
    probes: str = "",
):
    injected, questions = inject_needles(
        sample, n=needles, seed=seed, position=needle_position
    )
    probe_list = [p.strip() for p in probes.split(",") if p.strip()]
    seeded_topics: list[str] = []
    rng = random.Random(seed ^ 0x5EED)
    if topics:
        injected, seeded, distractors = inject_topics(injected, topics, rng)
        seeded_topics = list(seeded)
        if seeded:
            questions.append(
                Question(
                    kind="continuity",
                    question=CONTINUITY_QUESTION,
                    answer="",
                    category="continuity",
                    meta={"terms": seeded, "distractors": distractors},
                )
            )
    if "currency" in probe_list and seeded_topics:
        from context_eval.long_horizon import inject_currency

        injected, currency_qs = inject_currency(injected, seeded_topics, seed)
        questions += currency_qs
    if "next_step" in probe_list:
        from context_eval.long_horizon import inject_next_step

        injected, next_q = inject_next_step(injected, seed)
        questions.append(next_q)
    if not skip_locomo and sample.dataset == "longmemeval" and sample.meta.get("qa"):
        # official answer-check judge (see judge._score_longmemeval)
        questions += [
            Question(
                kind="longmemeval",
                question=qa["question"],
                answer=qa["answer"],
                category=qa["category_name"],
                meta={"question_id": sample.sample_id},
            )
            for qa in sample.meta["qa"]
        ]
    elif not skip_locomo and sample.meta.get("qa"):
        questions += [
            Question(
                kind="locomo",
                question=qa["question"],
                answer=qa["answer"],
                category=qa["category_name"],
                meta={"category": qa["category"]},
            )
            for qa in sample.meta["qa"]
        ]
    for qa in trace_qa or []:
        questions.append(
            Question(
                kind="trace_qa",
                question=qa["question"],
                answer=qa["answer"],
                category="trace",
            )
        )
    if max_q and len(questions) > max_q:
        rng = random.Random(seed)
        # Cap the QA pool; keep every needle (depth curves), the
        # continuity probe, and the long-horizon probes (never droppable).
        keep_kinds = ("needle", "continuity", "currency", "next_step")
        keepers = [q for q in questions if q.kind in keep_kinds]
        other = [q for q in questions if q.kind not in ("needle", "continuity")]
        keep = max(0, max_q - len(keepers))
        if len(other) > keep:
            rng.shuffle(other)
            other = other[:keep]
        questions = keepers + other
    return injected, questions


def shard_samples(samples: list, i: int, n: int) -> list:
    """Shard i of n: a partition of samples (union = all, disjoint)."""
    if not (0 <= i < n):
        raise ValueError(f"shard index {i} out of range for n={n}")
    return samples[i::n]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()

    samples = read_jsonl(args.path)
    if args.samples and len(samples) > args.samples:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.samples)
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
            samples = shard_samples(samples, i, n)
        except ValueError:
            print(f"bad --shard {args.shard!r}; expected i/n with 0 <= i < n", file=sys.stderr)
            return 2

    budgets = [int(b) for b in args.budgets.split(",")]
    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for name in strategy_names:
        if name not in REGISTRY:
            print(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}", file=sys.stderr)
            return 2

    llm = OpenRouterClient()
    counter = default_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = SYSTEM_PROMPT
    if args.recall_instruct:
        system_prompt += (
            " The history is compacted: messages shown as [§id · head:/tail:] "
            "citations are truncated previews. When a detail you need is not "
            "visible in a citation, call history_recall with that § id to "
            "fetch the full original text."
        )
    trace_qa_map = (
        json.loads(args.trace_qa.read_text()) if args.trace_qa and args.trace_qa.exists() else {}
    )

    # sidecars replace like the row file: a rerun must not leave stale
    # recordings from a previous invocation globbed in with fresh ones
    import glob as _glob

    for old in args.out.parent.glob(
        _glob.escape(args.out.stem) + ".*.recording.json"
    ):
        old.unlink()

    runs = 0
    grouped: dict[tuple[str, int], list] = defaultdict(list)
    with args.out.open("w") as out_file:  # truncate: reruns replace, never accumulate
        for sample in samples:
            injected, questions = build_questions(
                sample, args.needles, args.seed, args.max_questions, args.skip_locomo_qa,
                topics=args.topics, trace_qa=trace_qa_map.get(sample.sample_id),
                needle_position=args.needle_position, probes=args.probes,
            )
            for name in strategy_names:
                for budget in budgets:
                    result = run_sample(
                        strategy=build(name),
                        sample=injected,
                        questions=questions,
                        llm=llm,
                        counter=counter,
                        budget_tokens=budget,
                        system_prompt=system_prompt,
                    )
                    out_file.write(result.to_json() + "\n")
                    out_file.flush()
                    if result.recording:
                        safe_id = "".join(
                            c if c.isalnum() else "-" for c in result.sample_id[:24]
                        ).strip("-")
                        sidecar = args.out.with_name(
                            f"{args.out.stem}.{result.strategy}."
                            f"{result.budget_tokens}.{safe_id}.recording.json"
                        )
                        # atomic: a crash mid-write must not leave a truncated
                        # recording that later sweeps silently consume
                        tmp = sidecar.with_suffix(".recording.json.tmp")
                        tmp.write_text(json.dumps(result.recording, default=str))
                        tmp.replace(sidecar)
                    grouped[(name, budget)].append(result)
                    runs += 1
                    status = (
                        f"err={result.error}"
                        if result.error
                        else f"overflow@{result.overflowed_at}"
                        if result.overflowed_at is not None
                        else f"score={result.mean_score:.2f}"
                    )
                    print(
                        f"[{runs:3d}] {name:16} budget={budget:7d} "
                        f"{result.sample_id:40} q={len(result.answers):3d} {status}"
                    )

    print(f"\n{'strategy':16} {'budget':>7} {'runs':>4} {'acc':>6} "
          f"{'tok/q':>7} {'peak':>8} {'mean':>8} {'cmp':>4} {'ovf':>4}")
    for (name, budget), results in sorted(grouped.items()):
        answered = [r for r in results if r.overflowed_at is None and not r.error]
        overflowed = sum(1 for r in results if r.overflowed_at is not None)
        errored = sum(1 for r in results if r.error)
        acc = (
            sum(r.mean_score for r in answered) / len(answered) if answered else 0.0
        )
        tok_per_q = (
            sum(r.usage.prompt_tokens for r in answered)
            / max(1, sum(len(r.answers) for r in answered))
        )
        peak = max((r.peak_view_tokens for r in results), default=0)
        mean = sum(r.mean_view_tokens for r in results) / max(1, len(results))
        cmp_ = sum(r.compactions for r in results)
        print(
            f"{name:16} {budget:7d} {len(results):4d} {acc:6.2f} {tok_per_q:7.0f} "
            f"{peak:8d} {mean:8.0f} {cmp_:4d} {overflowed + errored:4d}"
        )
    print(f"\nwrote {runs} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
