"""#2 — LCM: hierarchical summary DAG + recall tools (clean-room, arXiv:2605.04050).

Effectively lossless: the active view is a bounded projection (older history
replaced by depth-graded summaries), originals stay verbatim in the store
and remain recoverable via search/recall tools.

Clean-room mechanics implemented here:
- two thresholds on the assembled view: ``tau_soft`` (compact between turns)
  and ``tau_hard`` (compact before continuing); in this synchronous harness
  both fire in ``on_turn`` — the async/blocking distinction is recorded as
  trigger metadata for the writeup
- compaction summarizes the oldest unsummarized chunk into a depth-1 node;
  when four siblings accumulate at a depth they consolidate into a parent
  (D1 session -> D2 phase -> D3+ durable), preserving [§id] references
- escalation ladder: if summarization fails, degrade deterministically to
  pointer nodes (segment omitted; recall tools still work)
- tools: ``history_search`` (FTS5/BM25 over the verbatim archive, with a
  token-overlap fallback) and ``history_recall`` (full content by id)

Divergences from the paper (documented for the writeup): fixed sibling
arity 4 instead of parameter sweeps; chunk size as a fraction of budget;
per-model prompt scaling omitted.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass

from context_eval.provider import ToolSchema
from context_eval.strategies.base import Engine, EngineTool, Strategy, StrategyStats
from context_eval.types import PromptMessage, Role, Usage

TAU_SOFT = 0.70  # fraction of budget: begin compacting between turns
TAU_HARD = 0.90  # fraction of budget: compact before continuing
CHUNK_FRAC = 0.25  # target chunk size as a fraction of budget
SIBLINGS_TO_MERGE = 4
# Cap on summary generation: reasoning tokens draw from the same budget,
# and observed per-call reasoning runs p99~256 / max~395 tokens on this
# model -- a tight cap truncates summaries mid-sentence, silently
# degrading the DAG. View size is enforced at assembly, not here.
# max_tokens is a REASONING+content cap on deepseek-flash: big summarization
# segments reason for 2000+ tokens and 2048 truncates content to nothing
# (measured: reasoning_tok 2039/2048, 18 chars emitted, flat ran summary-less
# through the whole suite). 4096 gives reasoning headroom above terse content,
# matching the recap call's budget.
# Experiment switch: CONTEXT_EVAL_SUMMARY_MAX_TOKENS overrides the cap
# (e.g. 8192 for the reasoning-separation legs) so cap sensitivity can be
# measured without code changes; flat/representation import this constant.
SUMMARY_MAX_TOKENS = int(os.environ.get("CONTEXT_EVAL_SUMMARY_MAX_TOKENS", "4096"))

# Calibrated compaction-call operating point (results/smoke): reasoning
# disabled + word budget stated in the prompt + matched backstop cap.
# 8/8 cells complete, budget-honoring, 100% of available inventory markers
# retained; reasoning-enabled blanks even at 8192 (appetite, not cap size)
# and bills 1.1K-8.2K discarded thinking tokens per call.
# CONTEXT_EVAL_SUMMARIZER_REASONING=on selects the reasoning-enabled
# ablation path (needs require_content; still blanks on the tail).
SUMMARIZER_REASONING_OFF = (
    os.environ.get("CONTEXT_EVAL_SUMMARIZER_REASONING", "off") != "on"
)
SUMMARY_WORDS = 150
SUMMARY_BACKSTOP_TOKENS = 1024


def summary_call_kwargs() -> dict:
    """Call parameters shared by every compaction call site."""
    if SUMMARIZER_REASONING_OFF:
        return {
            "reasoning": {"enabled": False},
            "max_tokens": SUMMARY_BACKSTOP_TOKENS,
            "require_content": True,
        }
    return {"max_tokens": SUMMARY_MAX_TOKENS, "require_content": True}

# Experiment switch: CONTEXT_EVAL_TOPICS_LINES=0 assembles the pre-P1
# prompts (no inventory instruction) so control/treatment legs can run
# against the same code with only the prompt differing.
_TOPICS_IN_PROMPTS = os.environ.get("CONTEXT_EVAL_TOPICS_LINES", "1") != "0"

_D1_PROMPT = (
    f"""Summarize this segment of an agent session (coding, chat, or mixed) for
future reference.
Preserve exactly: the task/goal, key decisions and why, file paths touched,
commands run and their outcomes, errors and resolutions, and every specific
value (ids, tokens, numbers, names, versions).
Mark where each fact came from as [§<id>]. Keep the whole summary under
~{SUMMARY_WORDS} words, including any final topic-inventory line.
"""
    + (
        "End with a TOPICS line: every named project, topic, codename, or\n"
        "investigation mentioned in this segment, as `name|status` (status = open,\n"
        "resolved, paused, in progress, ...). Never omit a named topic.\n"
        if _TOPICS_IN_PROMPTS
        else ""
    )
    + "\nSEGMENT:\n{segment}"
)

_DN_PROMPT = (
    """These are summaries of consecutive earlier segments of one agent session.
Consolidate them into one higher-level summary. Keep every specific value
(ids, tokens, numbers, names, versions) and their [§<id>] references; drop
narrative. Keep the summary under ~150 words.
"""
    + (
        "Your TOPICS line is the union of the children's TOPICS lines: deduplicate\n"
        "by name, keep the latest status, never drop a named topic.\n"
        if _TOPICS_IN_PROMPTS
        else ""
    )
    + "\nSUMMARIES:\n{summaries}"
)


@dataclass
class _Node:
    depth: int  # 1 = leaf summary, 2+ = consolidations
    summary: str
    covers: tuple[str, ...]  # message ids covered (transitively)
    trigger: str = "soft"


class LCMDagger(Strategy):
    name = "lcm"

    def __init__(
        self,
        tau_soft: float = TAU_SOFT,
        tau_hard: float = TAU_HARD,
        chunk_frac: float = CHUNK_FRAC,
    ) -> None:
        self.tau_soft = tau_soft
        self.tau_hard = tau_hard
        self.chunk_frac = chunk_frac
        self._nodes: list[_Node] = []
        self._frontier = 0  # store messages below this index are summarized
        self._overhead = Usage()
        self._search_calls = 0
        self._recall_calls = 0
        self._escalations = 0
        self._empty_summaries = 0
        self._last_trigger = "none"
        self._fts = _FtsIndex()

    # --- compaction ---------------------------------------------------------

    def on_turn(self, engine: Engine) -> None:
        self._fts.index(engine.store.messages()[self._fts.indexed_upto :])
        budget = engine.budget.max_context_tokens
        view_tokens = engine.counter.count_messages(self._raw_view(engine))
        if view_tokens <= int(budget * self.tau_soft):
            self._last_trigger = "none"
            return

        self._last_trigger = (
            "hard" if view_tokens > int(budget * self.tau_hard) else "soft"
        )
        chunk_target = max(1, int(budget * self.chunk_frac))
        guard = 0
        while (
            engine.counter.count_messages(self._raw_view(engine)) > int(budget * self.tau_soft)
            and self._frontier < len(engine.store.messages())
        ):
            guard += 1
            if guard > 64:  # pathological churn guard
                break
            self._compact_oldest_chunk(engine, chunk_target)

    def _compact_oldest_chunk(self, engine: Engine, chunk_target: int) -> None:
        messages = engine.store.messages()
        counter = engine.counter
        chunk: list = []
        tokens = 0
        start = self._frontier
        for m in messages[start:]:
            cost = counter.count_message(m)
            if chunk and tokens + cost > max(chunk_target, cost):
                break
            chunk.append(m)
            tokens += cost
            if tokens >= chunk_target:
                break
        if not chunk:
            return

        summary = self._summarize_chunk(engine, chunk)
        if summary is None:
            # Escalation ladder: deterministic pointer node, still recallable
            first, last = chunk[0].id, chunk[-1].id
            summary = (
                f"[segment §{first}..§{last} omitted deterministically; "
                f"use history_search/history_recall to recover]"
            )
            self._escalations += 1

        self._nodes.append(
            _Node(
                depth=1,
                summary=summary,
                covers=tuple(m.id for m in chunk),
                trigger=self._last_trigger,
            )
        )
        self._frontier = start + len(chunk)
        self._consolidate(engine)

    def _summarize_chunk(self, engine: Engine, chunk) -> str | None:
        if engine.llm is None:
            return None
        # Full content: the chunk is already bounded by chunk_target, and
        # truncating here would hide deep content (and injected needles)
        # from the summary while flat-compact's summarizer sees everything.
        segment = "\n".join(f"[§{m.id} {m.role.value}] {m.content}" for m in chunk)
        try:
            completion = engine.llm.complete(
                [{"role": "user", "content": _D1_PROMPT.format(segment=segment)}],
                **summary_call_kwargs(),
            )
        except Exception:
            return None
        self._overhead.add(completion.usage)
        text = completion.text.strip()
        if not text and completion.finish_reason == "length":
            self._empty_summaries += 1
        return text or None

    def _consolidate(self, engine: Engine) -> None:
        """Merge the oldest SIBLINGS_TO_MERGE siblings at any depth."""
        merged = True
        while merged:
            merged = False
            by_depth: dict[int, list[int]] = {}
            for idx, node in enumerate(self._nodes):
                if node is not None:
                    by_depth.setdefault(node.depth, []).append(idx)
            for depth in sorted(by_depth):
                live = by_depth[depth]
                if len(live) < SIBLINGS_TO_MERGE:
                    continue
                group = [self._nodes[i] for i in live[:SIBLINGS_TO_MERGE]]
                parent = self._consolidate_group(engine, group)
                if parent is None:
                    continue  # summarizer unavailable: leave children as-is
                for i in live[:SIBLINGS_TO_MERGE]:
                    self._nodes[i] = None
                self._nodes.append(parent)
                merged = True
                break

    def _consolidate_group(self, engine: Engine, group: list[_Node]) -> _Node | None:
        if engine.llm is None:
            return None
        summaries = "\n---\n".join(n.summary for n in group)
        try:
            completion = engine.llm.complete(
                [{"role": "user", "content": _DN_PROMPT.format(summaries=summaries)}],
                **summary_call_kwargs(),
            )
        except Exception:
            return None
        self._overhead.add(completion.usage)
        text = completion.text.strip()
        if not text:
            return None
        covers = tuple(mid for n in group for mid in n.covers)
        return _Node(depth=group[0].depth + 1, summary=text, covers=covers)

    # --- view ----------------------------------------------------------------

    def _raw_view(self, engine: Engine) -> list[PromptMessage]:
        view: list[PromptMessage] = []
        live = [n for n in self._nodes if n is not None]
        # Render in coverage order (covers[0] is chronological): consolidated
        # parents are appended late but cover the oldest ids.
        for node in sorted(live, key=lambda n: n.covers[0]):
            view.append(
                PromptMessage(
                    role=Role.USER,
                    content=(
                        f"[Earlier context — summarized (D{node.depth}, "
                        f"§{node.covers[0]}..§{node.covers[-1]})]\n{node.summary}"
                    ),
                )
            )
        if view:
            view.append(PromptMessage(role=Role.USER, content="[Recent context — verbatim]"))
        for m in engine.store.messages()[self._frontier :]:
            view.append(m.to_prompt())
        return view

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        return self._raw_view(engine)

    # --- recall tools ----------------------------------------------------------

    def tools(self, engine: Engine) -> list[EngineTool]:
        search = ToolSchema(
            name="history_search",
            description=(
                "Search the verbatim archive of this session (full history, "
                "including compacted segments). Returns matching message ids "
                "with snippets; use history_recall to fetch full content."
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
            name="history_recall",
            description="Return the full original content of a message by id.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "message id, e.g. m000014"}
                },
                "required": ["id"],
            },
        )

        def do_search(args):
            self._search_calls += 1
            query = str(args.get("query", ""))
            self._fts.index(engine.store.messages()[self._fts.indexed_upto :])
            hits = self._fts.search(query, limit=5)
            if not hits:
                return "no matches"
            return "\n".join(f"[§{mid}] {snip}" for mid, snip in hits)

        def do_recall(args):
            self._recall_calls += 1
            mid = str(args.get("id", "")).strip().lstrip("§")
            msg = engine.store.get(mid)
            if msg is None:
                return f"error: unknown id {mid!r}"
            return f"[§{mid} {msg.role.value}]\n{msg.content}"

        return [
            EngineTool(schema=search, execute=do_search),
            EngineTool(schema=recall, execute=do_recall),
        ]

    def stats(self) -> StrategyStats:
        stats = StrategyStats()
        stats.overhead_usage = self._overhead
        stats.compactions = sum(
            1 for n in self._nodes if n is not None
        ) + self._escalations
        stats.recall_tool_calls = self._search_calls + self._recall_calls
        stats.empty_summaries = self._empty_summaries
        return stats


class _FtsIndex:
    """Incremental FTS5 index over the archive; token-overlap fallback."""

    def __init__(self) -> None:
        self.indexed_upto = 0
        self._conn: sqlite3.Connection | None = None
        self._fallback: list[tuple[str, str]] = []
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE VIRTUAL TABLE msgs USING fts5(id UNINDEXED, content)")
            self._conn = conn
        except sqlite3.Error:
            self._conn = None

    def index(self, messages) -> None:
        rows = [(m.id, m.content) for m in messages]
        if self._conn is not None:
            self._conn.executemany("INSERT INTO msgs (id, content) VALUES (?, ?)", rows)
        else:
            self._fallback.extend(rows)
        self.indexed_upto += len(rows)

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        if not query.strip():
            return []
        if self._conn is not None:
            fts_query = self._fts_query(query)
            if fts_query:
                try:
                    cursor = self._conn.execute(
                        "SELECT id, snippet(msgs, 1, '>', '<', '…', 12) "
                        "FROM msgs WHERE msgs MATCH ? ORDER BY rank LIMIT ?",
                        (fts_query, limit),
                    )
                    return [(row[0], row[1]) for row in cursor]
                except sqlite3.Error:
                    pass  # malformed MATCH syntax: fall through to overlap
        return self._overlap_search(query, limit)

    @staticmethod
    def _fts_query(query: str) -> str:
        """Quoted OR-joined terms; empty when nothing indexable (skips FTS)."""
        terms = [t for t in re.findall(r"[\w]+", query) if len(t) > 1][:8]
        return " OR ".join(f'"{t}"' for t in terms)

    def _overlap_search(self, query: str, limit: int) -> list[tuple[str, str]]:
        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for mid, content in self._fallback:
            tokens = set(re.findall(r"\w+", content.lower()))
            overlap = len(terms & tokens)
            if overlap:
                scored.append((-overlap, mid, content[:200]))
        scored.sort()
        return [(mid, snip) for _, mid, snip in scored[:limit]]
