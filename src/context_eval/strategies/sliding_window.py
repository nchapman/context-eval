"""0b — Sliding window (anchor, lower bound).

Keep the newest messages that fit the budget, dropping oldest first. Lossy,
no recovery. Token-budget-driven (rather than message-count) so it sits on
the same budget axis as every other strategy in the sweep.
"""

from __future__ import annotations

from context_eval.strategies.base import Engine, Strategy
from context_eval.types import PromptMessage


class SlidingWindow(Strategy):
    name = "sliding-window"

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        budget = engine.budget.max_context_tokens
        kept: list[PromptMessage] = []
        used = 0
        # Walk newest -> oldest; stop at the first message that does not fit.
        for msg in reversed(engine.store.messages()):
            cost = engine.counter.count_message(msg)
            if used + cost > budget:
                break
            used += cost
            kept.append(msg.to_prompt())
        kept.reverse()
        return kept
