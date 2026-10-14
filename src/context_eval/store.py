"""Append-only, ID-addressable transcript store.

Shared infrastructure (not a strategy): every arm that keeps a verbatim
archive with stable IDs (retrieval, the sawtooth ledger, ARC, editing,
representation, RLM-style recursion — see docs/lit-review.md §2) builds
on this; the "addressable store" is what makes pointers, citations, and
search possible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from context_eval.types import Message, Role


class TranscriptStore:
    """In-memory append-only archive. Message IDs are stable and monotonic."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._by_id: dict[str, Message] = {}

    def append(
        self,
        role: Role,
        content: str,
        meta: Mapping[str, Any] | None = None,
    ) -> Message:
        msg = Message(
            id=f"m{len(self._messages) + 1:06d}",
            role=role,
            content=content,
            meta=dict(meta) if meta else {},
        )
        if msg.id in self._by_id:
            raise ValueError(f"duplicate message id: {msg.id}")
        self._messages.append(msg)
        self._by_id[msg.id] = msg
        return msg

    def get(self, message_id: str) -> Message | None:
        return self._by_id.get(message_id)

    def require(self, message_id: str) -> Message:
        msg = self.get(message_id)
        if msg is None:
            raise KeyError(f"unknown message id: {message_id}")
        return msg

    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def tail(self, n: int) -> tuple[Message, ...]:
        if n <= 0:
            return ()
        return tuple(self._messages[-n:])

    def __len__(self) -> int:
        return len(self._messages)
