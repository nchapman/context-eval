"""Normalized replay-sample format shared by all dataset loaders.

The harness consumes ``ReplaySample`` objects: an ordered message list to
feed into the transcript store (open-loop replay), plus dataset metadata for
stratification and reporting. Question generation (needles, trace-derived QA)
happens downstream and is dataset-agnostic.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_eval.types import Role


@dataclass(frozen=True)
class ReplayMessage:
    role: Role
    content: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplaySample:
    dataset: str  # e.g. "open-swe-traces/v1.0/openhands"
    sample_id: str
    messages: tuple[ReplayMessage, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset": self.dataset,
                "sample_id": self.sample_id,
                "meta": dict(self.meta),
                "messages": [
                    {"role": m.role.value, "content": m.content, "meta": dict(m.meta)}
                    for m in self.messages
                ],
            }
        )

    @staticmethod
    def from_json(raw: str) -> ReplaySample:
        d = json.loads(raw)
        return ReplaySample(
            dataset=d["dataset"],
            sample_id=d["sample_id"],
            meta=dict(d.get("meta") or {}),
            messages=tuple(
                ReplayMessage(
                    role=Role(m["role"]),
                    content=m["content"] or "",
                    meta=dict(m.get("meta") or {}),
                )
                for m in d["messages"]
            ),
        )


def write_jsonl(samples: Sequence[ReplaySample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            f.write(s.to_json() + "\n")


def read_jsonl(path: Path) -> list[ReplaySample]:
    with path.open() as f:
        return [ReplaySample.from_json(line) for line in f if line.strip()]
