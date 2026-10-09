"""Strategy interface: the pluggable context-management contract.

Every strategy implements the same three operations over the same shared
infrastructure (store + counter + budget + optional LLM for compaction):

- ``assemble``: project the archive into the bounded active view for this call
- ``on_turn``: react to a transcript append (trigger compaction, etc.)
- ``tools``: engine-intercepted recall/search tools exposed to the model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from context_eval.budget import Budget
from context_eval.provider import LLMClient, ToolSchema
from context_eval.store import TranscriptStore
from context_eval.tokens import TokenCounter
from context_eval.types import PromptMessage, Usage


@dataclass
class StrategyStats:
    """Efficiency telemetry attributed to a strategy's own overhead."""

    # LLM work the strategy itself did (summarization, verification, ...).
    overhead_usage: Usage = field(default_factory=Usage)
    compactions: int = 0
    recall_tool_calls: int = 0
    # Compactions whose completion returned no content (reasoning burned the
    # max_tokens cap, provider failure, ...). Zero is the invariant; nonzero
    # means the arm is running with blank summaries and every downstream
    # metric for it is suspect.
    empty_summaries: int = 0
    # Peak tokens in any assembled active view (harness records per turn).
    peak_context_tokens: int = 0


@dataclass
class Engine:
    """What the harness hands to a strategy on every hook."""

    store: TranscriptStore
    counter: TokenCounter
    budget: Budget  # already net of system prompt + tool-schema overhead
    llm: LLMClient | None = None  # strategies that compact set this


@dataclass(frozen=True)
class EngineTool:
    """An engine-intercepted tool: executed by the strategy, never by an LLM."""

    schema: ToolSchema
    execute: Callable[[Mapping[str, Any]], str]


class Strategy(ABC):
    name: str = "strategy"

    def on_turn(self, engine: Engine) -> None:  # noqa: B027 - optional hook
        """Called after each transcript append. Default: nothing."""

    @abstractmethod
    def assemble(self, engine: Engine) -> list[PromptMessage]:
        """Build the active view (no system prompt; harness adds it)."""

    def tools(self, engine: Engine) -> list[EngineTool]:
        """Recall/search tools exposed to the model. Default: none."""
        return []

    def stats(self) -> StrategyStats:
        return StrategyStats()
