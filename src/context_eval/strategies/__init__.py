"""Strategy implementations (candidates from docs/candidates.md)."""

from context_eval.strategies.base import Engine, EngineTool, Strategy, StrategyStats
from context_eval.strategies.full_context import FullContext
from context_eval.strategies.sliding_window import SlidingWindow

REGISTRY = {
    "full-context": FullContext,
    "sliding-window": SlidingWindow,
}


def build(name: str) -> Strategy:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}") from None


__all__ = [
    "Engine",
    "EngineTool",
    "Strategy",
    "StrategyStats",
    "FullContext",
    "SlidingWindow",
    "REGISTRY",
    "build",
]
