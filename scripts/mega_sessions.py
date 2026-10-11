"""Build order-of-magnitude-longer pi sessions by interleaving the real ones.

The three recorded sessions are cut into chunks at user-turn boundaries
(each chunk opens with a user message) and interleaved round-robin with a
per-variant rotation, cycled until the target token estimate is reached.
Result: realistic "many projects over months" mega-sessions where
workstreams go dormant for hundreds of thousands of tokens and re-enter
with new statuses -- the stress that a 13-20-tooth grid never applies.

Usage:
  uv run python scripts/mega_sessions.py --cycles 4 --out data/pi_mega.jsonl
"""

import argparse
import json
from pathlib import Path

from context_eval.tokens import CharEstimator

EST = CharEstimator()
CHUNK_TARGET = 400  # messages per chunk (approx; cut at next user turn)


def session_chunks(messages: list[dict]) -> list[list[dict]]:
    """Split at user-turn boundaries into ~CHUNK_TARGET-message chunks."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "user"] or [0]
    chunks = []
    i = 0
    while i < len(starts):
        j = i
        while j + 1 < len(starts) and starts[j + 1] - starts[i] < CHUNK_TARGET:
            j += 1
        end = starts[j + 1] if j + 1 < len(starts) else len(messages)
        chunks.append(messages[starts[i]:end])
        i = j + 1
    return chunks


def est_tokens(msgs: list[dict]) -> int:
    return EST.count_text(" " * sum(len(m.get("content") or "") for m in msgs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--out", default="data/pi_mega.jsonl")
    args = ap.parse_args()

    sessions = [json.loads(ln) for ln in open("data/pi_sessions.jsonl")]
    chunked = {s["sample_id"][-4:]: session_chunks(s["messages"]) for s in sessions}
    for k, cs in chunked.items():
        print(f"{k}: {len(cs)} chunks, ~{est_tokens([m for c in cs for m in c]) // 1000}K tokens")

    # Two variants: different rotations so workstreams re-enter in
    # different orders and at different gaps.
    variants = [("mega-a", 0), ("mega-b", 1)]
    out = Path(args.out)
    with out.open("w") as f:
        for name, rot in variants:
            chunk_seq: list[dict] = []
            keys = list(chunked)
            for _ in range(args.cycles):
                # round-robin one chunk per session, rotating the lead
                # each pass so adjacent-chunk order differs across cycles
                pos = 0
                while any(pos < len(chunked[k]) for k in keys):
                    for si in range(len(keys)):
                        k = keys[(si + rot) % len(keys)]
                        if pos < len(chunked[k]):
                            chunk_seq.extend(chunked[k][pos])
                    pos += 1
            # ensure every message role is valid and content strings only
            clean = [
                {"role": m["role"], "content": m.get("content") or ""}
                for m in chunk_seq
            ]
            rec = {
                "dataset": "pi-session",
                "sample_id": name,
                "meta": {"built_from": keys, "cycles": args.cycles, "rotation": rot},
                "messages": clean,
            }
            f.write(json.dumps(rec) + "\n")
            print(f"{name}: {len(clean)} msgs, ~{est_tokens(clean) // 1000}K est tokens")


if __name__ == "__main__":
    main()
