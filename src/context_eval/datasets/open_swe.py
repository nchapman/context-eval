"""Open-SWE-Traces (nvidia/Open-SWE-Traces) loader.

Primary datasource: real coding-agent trajectories (issue -> tool calls ->
observations -> patch). Replayed open-loop as a realistic haystack for
context-management evaluation; see docs/lit-review.md and the dataset card
(CC BY 4.0, arXiv:2606.16038).

Fetches a stratified sample via the HF datasets-server rows API (no 42GB
download): scan sequentially, estimate tokens, bucket, dedupe by instance,
select deterministically, cache as JSONL.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from context_eval.datasets.base import ReplayMessage, ReplaySample, write_jsonl
from context_eval.types import Role

DATASET = "nvidia/Open-SWE-Traces"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

# v1.0 = MiniMax-M2.5 / Qwen3.5-122B teachers (avoids DeepSeek-authored
# traces while we evaluate DeepSeek models; v1.1+ adds DeepSeek-V4-Flash).
SPLITS = ("openhands", "sweagent")

# Char-based token estimate for stratification only (~4 chars/token).
CHARS_PER_TOKEN = 4
# Estimated-token buckets spanning the budget sweep (16K/32K/128K).
BUCKET_BOUNDS = (8_000, 16_000, 32_000, 64_000, 128_000)
BUCKET_LABELS = ("<8k", "8-16k", "16-32k", "32-64k", "64-128k", ">128k")


def estimate_tokens(chars: int) -> int:
    return chars // CHARS_PER_TOKEN


def bucket_for(est_tokens: int) -> str:
    # BUCKET_LABELS has one extra entry: the overflow bucket (>= last bound).
    for bound, label in zip(BUCKET_BOUNDS, BUCKET_LABELS, strict=False):
        if est_tokens < bound:
            return label
    return BUCKET_LABELS[-1]


@dataclass(frozen=True)
class _ScannedRow:
    sample: ReplaySample
    est_tokens: int
    bucket: str


def normalize(dataset: str, row: Mapping[str, Any]) -> ReplaySample:
    """Map a raw dataset row to a ReplaySample.

    - drops the source system message (the harness injects its own)
    - coerces null content to ""
    - carries assistant tool_calls (raw) in message meta for wire reconstruction
    - records reasoning presence per message (content itself is not replayed)
    """
    messages: list[ReplayMessage] = []
    for m in row["messages"]:
        role = Role(m["role"])
        if role is Role.SYSTEM:
            continue
        meta: dict[str, Any] = {}
        if m.get("tool_calls"):
            meta["tool_calls"] = m["tool_calls"]
        if m.get("reasoning_content"):
            meta["had_reasoning"] = True
        messages.append(ReplayMessage(role=role, content=m.get("content") or "", meta=meta))

    md = row.get("metadata") or {}
    teacher = (md.get("teacher_model") or {}).get("name")
    return ReplaySample(
        dataset=dataset,
        sample_id=str(row["instance_id"]),
        messages=tuple(messages),
        meta={
            "trajectory_id": row.get("trajectory_id"),
            "repo": row.get("repo"),
            "language": row.get("language"),
            "resolved": row.get("resolved"),
            "category": md.get("category"),
            "teacher_model": teacher,
            "harness": dataset.rsplit("/", 1)[-1],
        },
    )


class OpenSWELoader:
    def __init__(
        self,
        config: str = "v1.0",
        splits: tuple[str, ...] = SPLITS,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.splits = splits
        self._client = http_client

    def fetch_page(self, split: str, offset: int) -> list[dict[str, Any]]:
        client = self._client or httpx.Client(timeout=60.0)
        try:
            resp = client.get(
                ROWS_URL,
                params={
                    "dataset": DATASET,
                    "config": self.config,
                    "split": split,
                    "offset": offset,
                    "length": PAGE_SIZE,
                },
            )
            resp.raise_for_status()
            return [r["row"] for r in resp.json()["rows"]]
        finally:
            if self._client is None:
                client.close()

    def scan(self, split: str, max_rows: int) -> Iterator[_ScannedRow]:
        seen = 0
        offset = 0
        dataset = f"{DATASET.split('/')[-1].lower()}/{self.config}/{split}"
        while seen < max_rows:
            rows = self.fetch_page(split, offset)
            if not rows:
                return
            for row in rows:
                if seen >= max_rows:
                    return
                seen += 1
                sample = normalize(dataset, row)
                est = estimate_tokens(sample.total_chars)
                yield _ScannedRow(sample=sample, est_tokens=est, bucket=bucket_for(est))
            offset += PAGE_SIZE  # advance by requested length, per API pagination


def select(
    scanned: list[_ScannedRow],
    target: int,
    seed: int = 0,
    max_per_instance: int = 1,
) -> list[_ScannedRow]:
    """Deterministic stratified selection: spread `target` across buckets,
    dedupe by instance_id, random-shuffle within bucket for coverage."""
    by_instance: dict[str, int] = {}
    buckets: dict[str, list[_ScannedRow]] = {}
    for row in scanned:
        by_instance.setdefault(row.sample.sample_id, 0)
        if by_instance[row.sample.sample_id] >= max_per_instance:
            continue
        by_instance[row.sample.sample_id] += 1
        buckets.setdefault(row.bucket, []).append(row)

    present = sorted(buckets, key=BUCKET_LABELS.index)
    per_bucket = max(1, target // max(1, len(present)))
    rng = random.Random(seed)

    selected: list[_ScannedRow] = []
    for label in present:
        pool = buckets[label][:]
        rng.shuffle(pool)
        selected.extend(pool[:per_bucket])
    return selected


def fetch_sample(
    out: Path,
    target: int = 120,
    scan_per_split: int = 2_000,
    seed: int = 0,
    splits: tuple[str, ...] = SPLITS,
    config: str = "v1.0",
    refresh: bool = False,
    http_client: httpx.Client | None = None,
) -> list[ReplaySample]:
    """Load cached sample, or scan+select+cache. Returns ReplaySamples."""
    if out.is_file() and not refresh:
        from context_eval.datasets.base import read_jsonl

        return read_jsonl(out)

    loader = OpenSWELoader(config=config, splits=splits, http_client=http_client)
    scanned: list[_ScannedRow] = []
    for split in splits:
        scanned.extend(loader.scan(split, scan_per_split))

    chosen = select(scanned, target=target, seed=seed)
    samples = [row.sample for row in chosen]
    write_jsonl(samples, out)
    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/swe_v10.jsonl"))
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--scan-per-split", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result = fetch_sample(
        out=args.out,
        target=args.target,
        scan_per_split=args.scan_per_split,
        seed=args.seed,
        refresh=args.refresh,
    )
    from collections import Counter

    print(f"{len(result)} samples -> {args.out}")
    print("buckets:", Counter(bucket_for(estimate_tokens(s.total_chars)) for s in result))
    print("harnesses:", Counter(s.meta["harness"] for s in result))
    print("teachers:", Counter(s.meta["teacher_model"] for s in result))
