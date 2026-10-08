"""0a — Full context (anchor, upper bound).

Send everything verbatim. No compaction, no recovery path. Expected to win
raw accuracy at generous budgets and to overflow at tight ones — both are
data points, not bugs: the harness records the budget where the anchor dies.
"""

from __future__ import annotations

from context_eval.strategies.base import Engine, Strategy
from context_eval.types import PromptMessage


class FullContext(Strategy):
    name = "full-context"

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        return [m.to_prompt() for m in engine.store.messages()]
