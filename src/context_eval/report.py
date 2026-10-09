"""Results reporting: aggregates, Wilson CIs, depth curves, Pareto plot.

Consumes the RunResult JSONL written by run_eval and produces a markdown
report plus PNG plots. Two cost framings, both honest:
- pass cost: prompt tokens per answered question (answering calls only)
- conversation cost: cumulative resident-context tokens over the replay
  (sum of per-turn view tokens) + strategy overhead — the agentic-loop bill

Strategies are labelled "capped" at their budget; a budget at/above the
corpus max is labelled "uncapped" (the deployed full-context ceiling).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEPTH_RE = re.compile(r"^needle@(\d+)%$")

# At/above this budget every corpus trace fits verbatim: deployed ceiling.
UNCAPPED_BUDGET = 200_000

DEPTH_BUCKETS = (
    (0.0, 0.25, "0-25%"),
    (0.25, 0.5, "25-50%"),
    (0.5, 0.75, "50-75%"),
    (0.75, 1.01, "75-100%"),
)


def load_runs(*paths: str | Path) -> list[dict]:
    """Load RunResult dicts; on duplicate (strategy, sample, budget) keep last."""
    by_key: dict[tuple, dict] = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            run = json.loads(line)
            key = (run["strategy"], run["sample_id"], run["budget_tokens"])
            by_key[key] = run
    return list(by_key.values())


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binomial 95% CI (Wilson score)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def needle_depth(answer: dict) -> float | None:
    m = DEPTH_RE.match(answer.get("category", ""))
    return int(m.group(1)) / 100 if m else None


def _bucket(depth: float) -> str:
    for lo, hi, label in DEPTH_BUCKETS:
        if lo <= depth < hi:
            return label
    return DEPTH_BUCKETS[-1][2]


@dataclass
class Cell:
    strategy: str
    budget: int
    label: str
    tier: str
    runs: int
    overflowed: int
    errored: int
    questions: int
    answered: int
    acc: float
    acc_lo: float
    acc_hi: float
    pass_tok_per_q: float
    conv_view_tok_mean: float
    overhead_tok_mean: float
    compactions_mean: float
    tool_calls_per_q: float
    acc_penalized: float  # unanswered (overflow/errored runs) counted as 0
    abstain_rate: float  # blank predictions among answered (honest misses)
    confab_rate: float  # non-blank and wrong (confident misses)
    cost_per_q: float
    overhead_cost_mean: float
    cached_share: float
    view_rewrite_rate: float


def _tier(budget_tokens: int) -> str:
    """Deployment framing from docs/budgets.md (fixed costs sit outside budget)."""
    if budget_tokens >= UNCAPPED_BUDGET:
        return "uncapped"
    if budget_tokens >= 128_000:
        return "deployment"
    if budget_tokens >= 64_000:
        return "mid"
    return "stress"


def aggregate(runs: list[dict]) -> list[Cell]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["strategy"], run["budget_tokens"])].append(run)

    cells = []
    for (strategy, budget), group in sorted(grouped.items()):
        tier = _tier(budget)
        if strategy == "full-context" and tier == "uncapped":
            label = "full-context (uncapped)"
        elif strategy == "full-context":
            label = f"full-context (capped@{budget // 1000}k)"
        else:
            label = f"{strategy}@{budget // 1000}k"

        answers = [a for r in group for a in r["answers"]]
        scored = [a for a in answers if a.get("score_method") != "error"]
        successes = sum(1 for a in scored if a["score"] >= 0.5)
        lo, hi = wilson_interval(successes, len(scored))
        # Answer-mode split: abstention (blank prediction) is an honest
        # failure; confabulation (confident but wrong) is the silent one.
        # Headline acc conflates them -- the split is the finding.
        abstained = sum(1 for a in scored if not (a.get("prediction") or "").strip())
        wrong = len(scored) - successes - abstained
        # Overflow/errored runs answer nothing: score them as failures so
        # survival-conditioned `acc` cannot flatter overflow-prone arms
        # (arc, context-editing, full-context) at tight budgets.
        questions_per_run = max(
            (len(r["answers"]) for r in group if not r["error"] and r["overflowed_at"] is None),
            default=0,
        )
        denom = questions_per_run * len(group)

        pass_tok = sum(a["usage"]["prompt_tokens"] for a in answers)
        prompt_tok_all = sum(a["usage"]["prompt_tokens"] for a in answers)
        cached_tok_all = sum(a["usage"].get("cached_tokens", 0) for a in answers)
        answer_cost = sum(a["usage"].get("cost", 0.0) for a in answers)
        conv_view = sum(sum(t["view_tokens"] for t in r["turns"]) for r in group)
        overhead = sum(
            r["overhead_usage"]["prompt_tokens"] + r["overhead_usage"]["completion_tokens"]
            for r in group
        )
        overhead_cost = sum(r["overhead_usage"].get("cost", 0.0) for r in group)
        tools = sum(a.get("tool_calls", 0) for a in answers)
        rewrites = 0
        turn_count = 0
        for r in group:
            sig = None
            for t in r["turns"]:
                turn_count += 1
                current = t.get("view_head_sig")
                if sig is not None and current and current != sig:
                    rewrites += 1
                sig = current

        cells.append(
            Cell(
                strategy=strategy,
                budget=budget,
                label=label,
                tier=tier,
                runs=len(group),
                overflowed=sum(1 for r in group if r["overflowed_at"] is not None),
                errored=sum(1 for r in group if r["error"]),
                questions=len(answers),
                answered=len(scored),
                abstain_rate=abstained / len(scored) if scored else 0.0,
                confab_rate=wrong / len(scored) if scored else 0.0,
                acc=successes / len(scored) if scored else 0.0,
                acc_penalized=successes / denom if denom else 0.0,
                acc_lo=lo,
                acc_hi=hi,
                pass_tok_per_q=pass_tok / max(1, len(answers)),
                conv_view_tok_mean=conv_view / len(group),
                overhead_tok_mean=overhead / len(group),
                compactions_mean=sum(r["compactions"] for r in group) / len(group),
                tool_calls_per_q=tools / max(1, len(answers)),
                cost_per_q=answer_cost / max(1, len(answers)),
                overhead_cost_mean=overhead_cost / len(group),
                cached_share=cached_tok_all / prompt_tok_all if prompt_tok_all else 0.0,
                view_rewrite_rate=rewrites / turn_count if turn_count else 0.0,
            )
        )
    return cells


def depth_curves(runs: list[dict]) -> dict[str, list[tuple[str, float, int]]]:
    """Per strategy: accuracy by needle-depth bucket (bucket, acc, n)."""
    per_strategy: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for run in runs:
        for a in run["answers"]:
            depth = needle_depth(a)
            if depth is None:
                continue
            bucket = _bucket(depth)
            slot = per_strategy[run["strategy"]][bucket]
            slot[1] += 1
            if a["score"] >= 0.5:
                slot[0] += 1
    labels = [b[2] for b in DEPTH_BUCKETS]
    return {
        strategy: [
            (
                label,
                buckets[label][0] / buckets[label][1] if buckets[label][1] else 0.0,
                buckets[label][1],
            )
            for label in labels
        ]
        for strategy, buckets in per_strategy.items()
    }


def render_markdown(cells: list[Cell], curves: dict[str, list]) -> str:
    lines = [
        "# Results",
        "",
        "Accuracy = fraction of answered questions scoring >= 0.5; 95% Wilson CI.",
        "Costs are OpenRouter-reported USD (cache-aware): pass $/q = answering",
        "calls per question; overhead $ = compaction calls per run. cached =",
        "share of answering prompt tokens served from prefix cache. rewrite =",
        "acc = accuracy over answered questions; acc* = overflow-penalized",
        "(questions in overflowed/errored runs count as wrong).",
        "abst = blank-prediction rate among answered (honest miss, scores 0);",
        "confab = answered-wrong rate (confident miss). acc alone conflates them.",
        "share of turns where the strategy changed its view head (cache bust).",
        "conv view tok = cumulative resident-context tokens over the replay.",
        "",
        "| strategy | tier | runs | ovf | acc | acc* | 95% CI | abst | confab | pass tok/q | pass $/q |"
        " cached | rewrite | conv view tok | ohd $ | cmp | tools/q |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c.label} | {c.tier} | {c.runs} | {c.overflowed} | {c.acc:.2f} "
            f"| {c.acc_penalized:.2f} "
            f"| [{c.acc_lo:.2f}, {c.acc_hi:.2f}] | {c.abstain_rate:.0%} | {c.confab_rate:.0%} | {c.pass_tok_per_q:,.0f} "
            f"| ${c.cost_per_q:.4f} | {c.cached_share:.0%} | {c.view_rewrite_rate:.0%} "
            f"| {c.conv_view_tok_mean:,.0f} | ${c.overhead_cost_mean:.4f} "
            f"| {c.compactions_mean:.1f} | {c.tool_calls_per_q:.2f} |"
        )

    if curves:
        lines += [
            "",
            "## Budget tiers",
            "",
            "Budgets bound the strategy view only (fixed costs sit outside, as in",
            "production tools; see docs/budgets.md). stress = MemGPT-class core",
            "memory; deployment = Claude-Code/Cline-class history budget (window",
            "minus ~30K overhead minus reserve); uncapped = 1M-window regime.",
            "",
            "## Needle accuracy by depth",
            "",
        ]
        strategies = sorted(curves)
        header = ["strategy", *[label for label, _, _ in curves[strategies[0]]]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for s in strategies:
            row = [s]
            for _, acc, n in curves[s]:
                row.append(f"{acc:.2f} (n={n})")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def plot_pareto(cells: list[Cell], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in cells:
        x = c.pass_tok_per_q / 1000
        ax.errorbar(
            x, c.acc, yerr=[[c.acc - c.acc_lo], [c.acc_hi - c.acc]], fmt="o", capsize=3
        )
        ax.annotate(c.label, (x, c.acc), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("answering prompt tokens per question (thousands)")
    ax.set_ylabel("needle accuracy")
    ax.set_title("Accuracy vs pass cost")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_depth(curves: dict[str, list], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, points in sorted(curves.items()):
        labels = [p[0] for p in points]
        accs = [p[1] for p in points]
        ax.plot(labels, accs, marker="o", label=strategy)
    ax.set_xlabel("needle depth in trace")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Needle accuracy by depth (lost-in-the-middle view)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="RunResult JSONL file(s)")
    parser.add_argument("--out", type=Path, default=Path("results/report.md"))
    args = parser.parse_args(argv)

    runs = load_runs(*args.results)
    cells = aggregate(runs)
    curves = depth_curves(runs)

    # Sanity gate: surface pathological cells loudly instead of letting them
    # render as clean zeros. A cell with answers but 0.00 acc is the classic
    # provider-failure signature (e.g. the 429-exhaustion noise that once
    # looked like an accuracy delta); a cell with zero answered questions
    # must be explained by overflow, or something upstream is broken.
    for c in cells:
        if c.questions == 0:
            if c.overflowed + c.errored < c.runs:
                print(f"SANITY: {c.label} answered nothing on {c.runs} runs "
                      f"but only {c.overflowed + c.errored} overflowed/errored")
        elif c.acc == 0.0 and c.answered > 3:
            print(f"SANITY: {c.label} scored 0.00 on {c.answered} answered questions "
                      f"(provider failure or judge break?)")
    errs = [r for r in runs if r.get("error")]
    if errs:
        print(f"SANITY: {len(errs)}/{len(runs)} runs errored; first: {errs[0]['error'][:120]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = render_markdown(cells, curves)
    args.out.write_text(report)

    models = {r.get("model") for r in runs if r.get("model")}
    providers = {r.get("provider") for r in runs if r.get("provider")}
    print(f"{len(runs)} runs -> {args.out}")
    print(f"models: {models or '?'} | providers: {providers or '?'}")
    try:
        plot_pareto(cells, args.out.with_name("pareto.png"))
        plot_depth(curves, args.out.with_name("depth.png"))
        print(f"plots -> {args.out.with_name('pareto.png')}, {args.out.with_name('depth.png')}")
    except ImportError:
        print("matplotlib not installed; skipped plots")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
