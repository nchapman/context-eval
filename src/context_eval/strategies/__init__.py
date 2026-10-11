"""Strategy implementations (candidates from docs/candidates.md)."""

from context_eval.strategies.arc import ARCCitations
from context_eval.strategies.base import Engine, EngineTool, Strategy, StrategyStats
from context_eval.strategies.conclusions import ConclusionsMemory
from context_eval.strategies.context_editing import ContextEditing
from context_eval.strategies.flat_compact import FlatCompact
from context_eval.strategies.full_context import FullContext
from context_eval.strategies.lcm import LCMDagger
from context_eval.strategies.representation import RepresentationMemory
from context_eval.strategies.retrieval import RetrievalOnly
from context_eval.strategies.sawtooth import (
    SawtoothLedger,
    SawtoothLedger2,
    SawtoothLedger3,
    SawtoothLedger4,
    SawtoothNone,
    SawtoothRewrite,
    SawtoothSegments,
)
from context_eval.strategies.sliding_window import SlidingWindow

REGISTRY = {
    "full-context": FullContext,
    "sliding-window": SlidingWindow,
    "retrieval": RetrievalOnly,
    "context-editing": ContextEditing,
    "flat-compact": FlatCompact,
    "arc": ARCCitations,
    "lcm": LCMDagger,
    "representation": RepresentationMemory,
    "conclusions": ConclusionsMemory,
    "sawtooth-none": SawtoothNone,
    "sawtooth-rewrite": SawtoothRewrite,
    "sawtooth-segments": SawtoothSegments,
    "sawtooth-ledger": SawtoothLedger,
    "sawtooth-ledger2": SawtoothLedger2,
    "sawtooth-ledger3": SawtoothLedger3,
    "sawtooth-ledger4": SawtoothLedger4,
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
    "ARCCitations",
    "LCMDagger",
    "RepresentationMemory",
    "REGISTRY",
    "build",
]
