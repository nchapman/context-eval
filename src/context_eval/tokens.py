"""Token counting for budget enforcement and thresholds.

Exact counting uses DeepSeek's published tokenizer.json (same family as the
eval models); the char estimator is a consistent fallback that preserves
cross-strategy fairness (every strategy is measured by the same counter).
OpenRouter's per-call `usage` is ground truth and can validate the local
counter on live runs.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from context_eval.types import Message, PromptMessage

TOKENIZER_PATH = Path(
    os.environ.get("CONTEXT_EVAL_TOKENIZER", "data/deepseek_tokenizer.json")
)

# Rough chat-framing overhead per message (role tags, delimiters). Applied
# uniformly to every strategy, so it never biases comparisons.
# Calibration (scripts/calibrate_overhead.py, 24 live calls, deepseek
# fp8): true tokens = 0.928 * (chars/4) + 1.6/message — i.e. CHARS_PER_TOKEN=4
# overestimates content by ~7.7% and MESSAGE_OVERHEAD_TOKENS overcharges by
# ~2.4/message. Both errors are conservative (views slightly smaller than
# nominal) and symmetric across arms, so they cannot reorder comparisons;
# the constants are kept unchanged to preserve budget comparability with
# every recorded run.
MESSAGE_OVERHEAD_TOKENS = 4

CHARS_PER_TOKEN = 4


class TokenCounter(ABC):
    @abstractmethod
    def count_text(self, text: str) -> int: ...

    def count_message(self, message: PromptMessage | Message) -> int:
        """True wire cost of one message: content + overhead + tool-call
        arguments from meta. All strategy-internal budget arithmetic must go
        through this (not count_text(content)) so triggers match the
        driver's measurement exactly."""
        total = self.count_text(message.content) + MESSAGE_OVERHEAD_TOKENS
        calls = getattr(message, "meta", {}).get("tool_calls")
        if calls:
            total += self.count_text(json.dumps(calls, separators=(",", ":")))
        return total

    def count_messages(self, messages: Sequence[PromptMessage | Message]) -> int:
        return sum(self.count_message(m) for m in messages)


class CharEstimator(TokenCounter):
    """Deterministic fallback: ~4 chars per token, no dependencies."""

    def count_text(self, text: str) -> int:
        return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


class TokenizerCounter(TokenCounter):
    """Exact counting via a HF `tokenizers` JSON file (e.g. DeepSeek's)."""

    def __init__(self, path: Path | str = TOKENIZER_PATH) -> None:
        from tokenizers import Tokenizer  # deferred: optional at import time

        self._tokenizer = Tokenizer.from_file(str(path))

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def default_counter() -> TokenCounter:
    """Exact tokenizer when available, consistent estimator otherwise."""
    if TOKENIZER_PATH.is_file():
        try:
            return TokenizerCounter(TOKENIZER_PATH)
        except Exception:
            return CharEstimator()
    return CharEstimator()


class CachedCounter(TokenCounter):
    """Memoizes count_text by exact string.

    Replay re-assembles views every turn; message content is immutable, so
    caching by the string itself makes per-turn measurement linear in new
    content instead of the whole history.
    """

    def __init__(self, inner: TokenCounter) -> None:
        self._inner = inner
        self._cache: dict[str, int] = {}

    def count_text(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is None:
            cached = self._inner.count_text(text)
            self._cache[text] = cached
        return cached
