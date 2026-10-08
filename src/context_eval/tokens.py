"""Token counting for budget enforcement and thresholds.

Exact counting uses DeepSeek's published tokenizer.json (same family as the
eval models); the char estimator is a consistent fallback that preserves
cross-strategy fairness (every strategy is measured by the same counter).
OpenRouter's per-call `usage` is ground truth and can validate the local
counter on live runs.
"""

from __future__ import annotations

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
MESSAGE_OVERHEAD_TOKENS = 4

CHARS_PER_TOKEN = 4


class TokenCounter(ABC):
    @abstractmethod
    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Sequence[PromptMessage | Message]) -> int:
        return sum(self.count_text(m.content) + MESSAGE_OVERHEAD_TOKENS for m in messages)


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
