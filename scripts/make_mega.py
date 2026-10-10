"""Build multi-session mega-corpora: conversations strung together.

Long-horizon regimes need conversations longer than any single corpus
sample: this concatenates N samples from a source jsonl into one replay
with explicit session-boundary markers, seeded deterministically.

    uv run python scripts/make_mega.py --source data/swe_smoke.jsonl \
        --out data/mega_swe_pilot.jsonl --groups 1 --per-group 5

Output samples are ReplaySample-compatible (dataset prefixed "mega-").
"""

import json
import random
from pathlib import Path

from context_eval.datasets.base import ReplayMessage, ReplaySample, read_jsonl
from context_eval.types import Role

BOUNDARY = (
    "[session boundary] The previous session ended; a new session begins "
    "now. History above belongs to earlier sessions."
)


def concat_group(samples: list[ReplaySample], group_id: str) -> ReplaySample:
    messages: list[ReplayMessage] = []
    for i, s in enumerate(samples):
        if i:
            messages.append(
                ReplayMessage(role=Role.USER, content=BOUNDARY, meta={"boundary": True})
            )
        messages.extend(s.messages)
    return ReplaySample(
        dataset=f"mega-{samples[0].dataset}",
        sample_id=group_id,
        messages=messages,
        meta={
            "parts": [s.sample_id for s in samples],
            "turns": len(messages),
        },
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--per-group", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = read_jsonl(args.source)
    rng = random.Random(args.seed)
    order = list(samples)
    rng.shuffle(order)
    groups = [
        order[i * args.per_group : (i + 1) * args.per_group]
        for i in range(args.groups)
    ]
    with args.out.open("w") as f:
        for gid, parts in enumerate(groups):
            mega = concat_group(parts, f"mega-{args.source.stem}-g{gid}")
            f.write(json.dumps(mega.to_json()) + "\n")
            chars = sum(len(m.content) for m in mega.messages)
            print(
                f"{mega.sample_id}: {len(parts)} sessions, {len(mega.messages)} msgs, "
                f"~{chars // 4} tokens"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
