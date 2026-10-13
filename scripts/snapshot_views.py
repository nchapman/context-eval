"""Dump each arm's assembled memory view at checkpoints through a replay.

Writes, per (sample, arm): the view the model would see at 25/50/75/100%
of the transcript. Compaction runs live (LLM); no questions are asked.
Used for qualitative memory-coherence review -- the answer scores do not
capture what the resident memory *reads like*.

Usage: uv run python scripts/snapshot_views.py --path data/longmemeval_m25.jsonl \
    --sample-ids a,b,c --strategies flat-compact,lcm --out results/longmemeval/m25/views
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from context_eval.budget import Budget
from context_eval.datasets.base import ReplayMessage
from context_eval.provider import OpenRouterClient
from context_eval.store import TranscriptStore
from context_eval.strategies import REGISTRY
from context_eval.tokens import CachedCounter, CharEstimator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--sample-ids", required=True, help="comma-separated")
    ap.add_argument("--strategies", default="flat-compact,lcm")
    ap.add_argument("--budgets", type=int, default=64000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    want = set(args.sample_ids.split(","))
    samples = [
        json.loads(line)
        for line in Path(args.path).read_text().splitlines()
        if json.loads(line)["sample_id"] in want
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = (0.25, 0.5, 0.75, 1.0)
    from context_eval.env import load_env

    load_env()
    llm = OpenRouterClient()

    for s in samples:
        for strat_name in args.strategies.split(","):
            store = TranscriptStore()
            strategy = REGISTRY[strat_name]()
            engine = type(
                "E",
                (),
                {
                    "store": store,
                    "counter": CachedCounter(CharEstimator()),
                    "budget": Budget(max_context_tokens=args.budgets),
                    "llm": llm,
                },
            )()
            from context_eval.types import Role

            role_of = {"user": Role.USER, "assistant": Role.ASSISTANT}
            msgs = [
                ReplayMessage(role=role_of[m["role"]], content=m["content"], meta={})
                for m in s["messages"]
            ]
            n = len(msgs)
            dump = []
            for frac in checkpoints:
                upto = max(1, int(n * frac))
                for m in msgs[getattr(strategy, "_dumped_upto", 0) : upto]:
                    store.append(m.role, m.content)
                    strategy.on_turn(engine)
                strategy._dumped_upto = upto
                view = strategy.assemble(engine)
                dump.append(
                    f"===== {s['sample_id']} {strat_name} @ {int(frac*100)}% "
                    f"({upto}/{n} msgs) =====\n"
                    + "\n".join(f"[{p.role.value}] {p.content}" for p in view)
                )
            stem = f"{s['sample_id']}.{strat_name}"
            (out / f"{stem}.txt").write_text("\n\n".join(dump))
            print(f"wrote {out / (stem + '.txt')}")


if __name__ == "__main__":
    main()
