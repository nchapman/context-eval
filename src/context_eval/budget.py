"""Context budget definition and enforcement.

The harness enforces a fixed budget per LLM call across every strategy — the
only fair comparison (ARC's protocol). Strategies compute their active view;
the harness hard-fails a run if the assembled view overflows, and records the
violation. Full-context is expected to overflow at small budgets: that is the
anchor's own failure mode and part of the data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    # Hard cap on prompt tokens per LLM call, including system prompt and
    # tool-schema overhead. Strategies receive a Budget already net of that
    # fixed overhead so the cap they see applies to their active view.
    max_context_tokens: int
    max_output_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


class BudgetExceededError(RuntimeError):
    """Raised when an assembled view would overflow the fixed budget."""

    def __init__(self, label: str, tokens: int, budget: int) -> None:
        super().__init__(
            f"{label}: assembled view is {tokens} tokens, over budget of {budget}"
        )
        self.label = label
        self.tokens = tokens
        self.budget = budget


def enforce(tokens: int, budget: Budget, label: str) -> None:
    if tokens > budget.max_context_tokens:
        raise BudgetExceededError(label, tokens, budget.max_context_tokens)
