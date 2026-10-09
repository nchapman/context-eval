"""Retrieval-only strategy: recent window + verbatim archive search.

The "why not just RAG the transcript?" baseline. The resident view is a
plain sliding window; everything older is reachable only through search
over the verbatim archive. No compaction, no summaries, no resident
previews -- the LLM never runs during replay.

Ranking is a Python port of the pi-extensions recall design (battle-tested
in production): hand-rolled BM25 over line-aligned chunks with
identifier-aware tokenization (``parseHeader`` matches "parse header"),
multiplied by an exponential recency decay measured in archive positions
from the frontier (floored, so old content fades but never vanishes).
Citation: ../pi-extensions/extensions/recall.ts.

Factorially this completes the strategy grid:

| strategy       | resident view          | recall tool      |
|----------------|------------------------|------------------|
| sliding-window | recent tail            | none             |
| retrieval      | recent tail            | search + recall  |
| arc            | tail + previews        | recall           |
| lcm            | tail + summaries (DAG) | search + recall  |

retrieval vs sliding-window isolates the value of on-demand recall;
retrieval vs lcm isolates the value of a resident structured view.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Mapping
from typing import Any

from context_eval.provider import ToolSchema
from context_eval.strategies.base import Engine, EngineTool, StrategyStats
from context_eval.strategies.sliding_window import SlidingWindow

BM25_K1 = 1.2
BM25_B = 0.75
CHUNK_CHARS = 3_000
# Age (in archive messages from the frontier) at which a chunk's score has
# halved; floored so distinctive old content still surfaces. The production
# analogue uses wall-clock hours (half-life 4h, floor 0.25).
HALF_LIFE_MESSAGES = 500.0
RECENCY_FLOOR = 0.25


def tokenize(text: str) -> list[str]:
    """Identifier-aware terms: parseHeader / parse_header / "parse header" all
    produce the same tokens (mirrors pi-extensions recall)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"([A-Za-z])(\d)", r"\1 \2", spaced)
    spaced = re.sub(r"(\d)([A-Za-z])", r"\1 \2", spaced)
    spaced = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", spaced)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split into <= max_chars pieces, preferring line boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    offset = 0
    while offset < len(text):
        window = text[offset : offset + max_chars]
        cut = window.rfind("\n")
        if offset + max_chars >= len(text) or cut <= 0:
            chunks.append(window)
            offset += max_chars
        else:
            chunks.append(window[: cut + 1])
            offset += cut + 1
    return chunks


class Bm25Index:
    """Decayed BM25 over message chunks; deterministic, no dependencies.

    Scoring runs on SQLite FTS5 (the same engine LCM's history_search uses;
    bm25 with k1=1.2, b=0.75) over identifier-tokenized chunk text, with a
    two-stage re-rank: FTS5 fetches top candidates, Python applies the
    memory-horizon decay. Environments without FTS5 fall back to a pure
    Python scorer implementing the identical policy. Engine pattern per
    alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search (candidate fetch,
    then fuse); a sqlite-vec hybrid arm is a natural future extension.
    """

    def __init__(
        self,
        half_life: float = HALF_LIFE_MESSAGES,
        floor: float = RECENCY_FLOOR,
        use_fts: bool = True,
    ) -> None:
        self.half_life = half_life
        self.floor = floor
        self._chunks: list[tuple[str, dict[str, int], int]] = []  # (msg_id, tf, pos)
        self._texts: dict[str, str] = {}
        self._df: dict[str, int] = {}
        self._total_len = 0
        self._conn: sqlite3.Connection | None = None
        if use_fts:
            try:
                conn = sqlite3.connect(":memory:")
                conn.execute(
                    "CREATE VIRTUAL TABLE chunks USING "
                    "fts5(msg_id UNINDEXED, pos UNINDEXED, content)"
                )
                self._conn = conn
            except sqlite3.Error:
                self._conn = None

    def add(self, message_id: str, content: str) -> None:
        pos = len(self._chunks)
        for piece in chunk_text(content):
            tokens = tokenize(piece)
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
            self._total_len += len(tokens)
            self._chunks.append((message_id, tf, pos))
            self._texts[message_id] = content
            if self._conn is not None:
                self._conn.execute(
                    "INSERT INTO chunks (msg_id, pos, content) VALUES (?, ?, ?)",
                    (message_id, pos, " ".join(tokens)),
                )

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        terms = set(tokenize(query))
        if not terms or not self._chunks:
            return []
        if self._conn is not None:
            ranked = self._fts_candidates(terms)
        else:
            ranked = self._python_scores(terms)
        frontier = self._chunks[-1][2]
        scored: list[tuple[float, int, str]] = []
        for raw, pos, msg_id in ranked:
            age = frontier - pos
            recency = max(self.floor, 0.5 ** (age / self.half_life))
            scored.append((raw * recency, pos, msg_id))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        return [(msg_id, self._snippet(msg_id, terms)) for _, _, msg_id in scored[:limit]]

    def _fts_candidates(
        self, terms: set[str], fetch: int = 50
    ) -> list[tuple[float, int, str]]:
        """Raw bm25 scores from FTS5 (descending); -rank is the score."""
        match = " OR ".join(f'"{t}"' for t in sorted(terms))
        try:
            cursor = self._conn.execute(
                "SELECT msg_id, pos, rank FROM chunks "
                "WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
                (match, fetch),
            )
            return [(-row[2], row[1], row[0]) for row in cursor]
        except sqlite3.Error:
            return self._python_scores(terms)

    def _python_scores(self, terms: set[str]) -> list[tuple[float, int, str]]:
        """Fallback scorer: identical Okapi BM25 over the same chunks."""
        n = len(self._chunks)
        avg = self._total_len / n
        scored: list[tuple[float, int, str]] = []
        for msg_id, tf, pos in self._chunks:
            score = 0.0
            length = sum(tf.values())
            for t in terms:
                f = tf.get(t)
                if not f:
                    continue
                df = self._df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += (
                    idf * f * (BM25_K1 + 1)
                ) / (
                    f + BM25_K1 * (1 - BM25_B + BM25_B * length / avg)
                )
            if score > 0:
                scored.append((score, pos, msg_id))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        return scored

    def _snippet(self, message_id: str, terms: set[str]) -> str:
        text = self._texts.get(message_id, "")
        joined = " ".join(text.split())
        lower = joined.lower()
        # Center on the rarest query term that appears: distinctive terms
        # ("480") locate the relevant passage; common ones ("line") do not.
        by_df = sorted(terms, key=lambda t: (self._df.get(t, 0), t))
        best = None
        for t in by_df:
            i = lower.find(t)
            if i >= 0:
                best = i
                break
        start = max(0, (best or 0) - 120)
        return ("…" if start > 0 else "") + joined[start : start + 280]


class RetrievalOnly(SlidingWindow):
    name = "retrieval"

    def __init__(self) -> None:
        self._index = Bm25Index()
        self._search_calls = 0
        self._recall_calls = 0
        self._indexed_upto = 0

    def on_turn(self, engine: Engine) -> None:
        messages = engine.store.messages()
        for msg in messages[self._indexed_upto :]:
            self._index.add(msg.id, msg.content)
        self._indexed_upto = len(messages)

    def tools(self, engine: Engine) -> list[EngineTool]:
        search = ToolSchema(
            name="search_history",
            description=(
                "Search the verbatim full history of this session (including "
                "everything outside the visible recent window). Returns "
                "matching message ids with snippets; use recall_history to "
                "fetch full content by id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search terms"}
                },
                "required": ["query"],
            },
        )
        recall = ToolSchema(
            name="recall_history",
            description="Return the full original content of a message by id.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "message id, e.g. m000014"}
                },
                "required": ["id"],
            },
        )

        def do_search(args: Mapping[str, Any]) -> str:
            self._search_calls += 1
            results = self._index.search(str(args.get("query", "")), limit=5)
            if not results:
                return "no matches"
            return "\n".join(f"{mid}: {snippet}" for mid, snippet in results)

        def do_recall(args: Mapping[str, Any]) -> str:
            self._recall_calls += 1
            try:
                msg = engine.store.require(str(args.get("id", "")))
            except KeyError:
                return "no such message id"
            return msg.content

        return [
            EngineTool(schema=search, execute=do_search),
            EngineTool(schema=recall, execute=do_recall),
        ]

    def stats(self) -> StrategyStats:
        return StrategyStats(recall_tool_calls=self._search_calls + self._recall_calls)
