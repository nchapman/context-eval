"""LoCoMo loader (snap-research/locomo, ACL 2024, Maharana et al.).

Conversational-memory axis of the data mix: 10 very long human-human
conversations (~300 turns, up to 35 sessions) with annotated QA across five
categories. Used for literature comparability (mem0/evaluation and
filesystem-is-all-you-need both report on LoCoMo).

Category mapping and judging conventions verified against the authors'
task_eval/evaluation.py: F1-based scoring for categories 1-4, abstention
detection ("no information available") for category 5 (adversarial).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from context_eval.datasets.base import (
    ReplayMessage,
    ReplaySample,
    read_jsonl,
    write_jsonl,
)
from context_eval.types import Role

LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

# Verified against task_eval/evaluation.py in the LoCoMo repo.
QA_CATEGORIES = {
    1: "multi_hop",
    2: "single_hop",
    3: "temporal_reasoning",
    4: "open_domain",
    5: "adversarial",
}

_SESSION_KEY = re.compile(r"session_(\d+)$")


def _session_keys(conv: Mapping[str, Any]) -> list[str]:
    keys = [m.group(0) for k in conv if (m := _SESSION_KEY.match(k))]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def normalize_conversation(item: Mapping[str, Any]) -> ReplaySample:
    """Map one LoCoMo conversation to a ReplaySample.

    Turns become speaker-prefixed USER-role messages (human-human chat; the
    eval model never authored these). QA annotations ride in sample meta for
    the eval layer, with category names resolved.
    """
    conv = item["conversation"]
    messages: list[ReplayMessage] = []
    for session in _session_keys(conv):
        ts = conv.get(f"{session}_date_time", "")
        for turn in conv[session]:
            messages.append(
                ReplayMessage(
                    role=Role.USER,
                    content=f"{turn['speaker']}: {turn['text']}",
                    meta={
                        "speaker": turn["speaker"],
                        "dia_id": turn.get("dia_id"),
                        "session": session,
                        "session_ts": ts,
                    },
                )
            )

    qa = []
    for q in item.get("qa", []):
        category = int(q["category"])
        entry = {
            "question": q["question"],
            "evidence": q.get("evidence") or [],
            "category": category,
            "category_name": QA_CATEGORIES[category],
        }
        if category == 5:
            # Adversarial: gold behavior is abstention (per the authors'
            # evaluation.py); adversarial_answer is the unsupported trap.
            entry["answer"] = "no information available"
            entry["adversarial_answer"] = q.get("adversarial_answer")
        else:
            entry["answer"] = str(q["answer"])
        qa.append(entry)

    return ReplaySample(
        dataset="locomo",
        sample_id=str(item["sample_id"]),
        messages=tuple(messages),
        meta={
            "qa": qa,
            "speaker_a": conv.get("speaker_a"),
            "speaker_b": conv.get("speaker_b"),
            "n_sessions": len(_session_keys(conv)),
        },
    )


def load_locomo(
    out: str | Path = Path("data/locomo.jsonl"),
    raw_path: str | Path = Path("data/locomo10.json"),
    refresh: bool = False,
    http_client: httpx.Client | None = None,
) -> list[ReplaySample]:
    """Load normalized LoCoMo samples, downloading the source if needed."""
    out = Path(out)
    raw_path = Path(raw_path)
    if out.is_file() and not refresh:
        return read_jsonl(out)

    if not raw_path.is_file():
        client = http_client or httpx.Client(timeout=120.0)
        try:
            resp = client.get(LOCOMO_URL)
            resp.raise_for_status()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(resp.content)
        finally:
            if http_client is None:
                client.close()

    data = json.loads(raw_path.read_text())
    samples = [normalize_conversation(item) for item in data]
    write_jsonl(samples, out)
    return samples


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/locomo.jsonl"))
    parser.add_argument("--raw", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result = load_locomo(out=args.out, raw_path=args.raw, refresh=args.refresh)
    print(f"{len(result)} conversations -> {args.out}")
    for s in sorted(result, key=lambda s: s.total_chars):
        cats = Counter(q["category_name"] for q in s.meta["qa"])
        print(
            f"{s.sample_id:9} turns={len(s.messages):4d} sessions={s.meta['n_sessions']:3d} "
            f"chars={s.total_chars:7d} qa={len(s.meta['qa']):3d} cats={dict(cats)}"
        )
