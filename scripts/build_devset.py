"""Build the LongMemEval dev checkpoints (dev25, dev100).

dev25: 25 stratified tuning samples (seed 25).
dev100: dev25 + 75 fresh stratified samples (seed 100) — the seen/fresh
split exists to measure tuning overfit directly.

Usage: uv run python scripts/build_devset.py [25|100]
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path("data/longmemeval_s.jsonl")


def stratified(pool: list[dict], n: int, seed: int) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for s in pool:
        by_type[s["meta"]["qa"][0]["category_name"] or "?"].append(s)
    total = len(pool)
    raw = {t: len(v) * n / total for t, v in by_type.items()}
    alloc = {t: int(x) for t, x in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - int(raw[t]), reverse=True)[
        : n - sum(alloc.values())
    ]:
        alloc[t] += 1
    rng = random.Random(seed)
    picked: list[dict] = []
    for t in sorted(alloc):
        p = sorted(by_type[t], key=lambda s: s["sample_id"])
        picked += rng.sample(p, alloc[t])
    return picked


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "100"
    samples = [json.loads(ln) for ln in DATA.read_text().splitlines()]
    if which == "25":
        out = stratified(samples, 25, seed=25)
        path = "data/longmemeval_dev25.jsonl"
    else:
        dev25_ids = {s["sample_id"] for s in stratified(samples, 25, seed=25)}
        fresh_pool = [s for s in samples if s["sample_id"] not in dev25_ids]
        out = (
            [s for s in samples if s["sample_id"] in dev25_ids]
            + stratified(fresh_pool, 75, seed=100)
        )
        path = "data/longmemeval_dev100.jsonl"
    Path(path).write_text("\n".join(json.dumps(s) for s in out) + "\n")
    print(f"{len(out)} samples -> {path}")


if __name__ == "__main__":
    main()
