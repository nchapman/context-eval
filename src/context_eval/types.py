"""Core domain types shared by the harness, store, provider, and strategies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Roles in the append-only transcript."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # tool observation recorded verbatim in the archive


@dataclass(frozen=True)
class Message:
    """An immutable, ID-addressable transcript entry.

    Every message ever seen lives in the store verbatim; strategies project
    from this archive into a bounded active view.
    """

    id: str
    role: Role
    content: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> PromptMessage:
        return PromptMessage(role=self.role, content=self.content, meta=dict(self.meta))


@dataclass(frozen=True)
class PromptMessage:
    """A message in the assembled active view sent to the model."""

    role: Role
    content: str
    # Carried through from the archive (e.g. tool_calls) so the driver can
    # reconstruct the wire; synthesized entries (summaries) use their own.
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass
class Usage:
    """Token/call telemetry. One instance accumulates across a whole run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Subset of completion_tokens burned on reasoning (reasoning models).
    reasoning_tokens: int = 0
    # Subset of prompt_tokens served from provider prefix cache (discounted).
    cached_tokens: int = 0
    # Actual USD charged by OpenRouter for these calls (cache-aware).
    cost: float = 0.0
    llm_calls: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens
        self.cost += other.cost
        self.llm_calls += other.llm_calls

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
