"""LongMemEval-S loader (xiaowu0162/longmemeval-cleaned, ICLR 2025, Wu et al.).

External conversational-memory axis of the data mix: 500 questions across
six types (single-session-user/assistant/preference, multi-session,
temporal-reasoning, knowledge-update), each with its own compiled chat
history (~40-50 timestamped sessions, ~115K tokens with a real tokenizer).

Question wiring reuses the locomo kind end to end (F1 scoring via
locomo_score in judge.py): every LongMemEval type falls into the F1
branch; 'temporal-reasoning' maps onto the existing temporal handling.
The trade-off -- results report as kind=locomo -- is recorded per run by
the dataset field ('longmemeval') and category names keep the original
type strings for per-ability breakdowns.

Timestamps ride inline: each session's first user turn is prefixed with
'[YYYY/MM/DD HH:MM]' so temporal questions are answerable from the
transcript without inflating message counts with marker turns.

Output ordering is stratified (round-robin over question types) so
--samples N yields a balanced subset at any N.
"""

from __future__ import annotations

import json
from pathlib import Path

from context_eval.datasets.base import ReplayMessage, ReplaySample, write_jsonl
from context_eval.types import Role


def convert(raw_path: str | Path, out_path: str | Path) -> dict:
    """Convert longmemeval_s_cleaned.json to our replay jsonl."""
    data = json.loads(Path(raw_path).read_text())
    by_type: dict[str, list] = {}
    for q in data:
        by_type.setdefault(q["question_type"], []).append(q)
    type_counts = {t: len(v) for t, v in by_type.items()}
    stratified: list = []
    pools = [by_type[t] for t in sorted(by_type)]
    while any(pools):
        for pool in pools:
            if pool:
                stratified.append(pool.pop(0))

    samples = []
    for q in stratified:
        messages: list[ReplayMessage] = []
        for session, date in zip(
            q["haystack_sessions"], q["haystack_dates"], strict=True
        ):
            stamped = False
            for m in session:
                role = Role.USER if m["role"] == "user" else Role.ASSISTANT
                content = m["content"]
                if not stamped and role is Role.USER:
                    content = f"[{date}] {content}"
                    stamped = True
                messages.append(
                    ReplayMessage(role=role, content=content, meta={})
                )
        samples.append(
            ReplaySample(
                dataset="longmemeval",
                sample_id=q["question_id"],
                messages=tuple(messages),
                meta={
                    "question_date": q["question_date"],
                    "answer_session_ids": q["answer_session_ids"],
                    "qa": [
                        {
                            "question": q["question"],
                            "answer": q["answer"],
                            "category": 0,
                            "category_name": q["question_type"],
                        }
                    ],
                },
            )
        )
    write_jsonl(samples, out_path)
    n_msgs = sum(len(s.messages) for s in samples)
    return {
        "questions": len(samples),
        "types": type_counts,
        "messages_total": n_msgs,
        "mean_messages": n_msgs / len(samples),
    }


if __name__ == "__main__":
    import sys

    stats = convert(
        sys.argv[1] if len(sys.argv) > 1 else "data/longmemeval_s_cleaned.json",
        sys.argv[2] if len(sys.argv) > 2 else "data/longmemeval_s.jsonl",
    )
    print(json.dumps(stats, indent=1))
