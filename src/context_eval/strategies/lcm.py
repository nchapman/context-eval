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
commands run and their outcomes, errors and resolutions, every specific
value (ids, tokens, numbers, names, versions), who the user is and
prefers (field or role, sustained interests shown across turns, explicit
likes/dislikes and constraints), and the date of every dated event (use
the [YYYY/MM/DD] session stamps verbatim).
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
(ids, tokens, numbers, names, versions) and their [§<id>] references; keep
each dated event's date; drop narrative. Write as dated bullets
(`- name — date — value [§id]`), not prose. Keep the summary under
~{words} words.
"""
    + (
        "Your TOPICS line is the union of the children's TOPICS lines: deduplicate\n"
        "by name, keep the latest status, never drop a named topic.\n"
        if _TOPICS_IN_PROMPTS
        else ""
    )
    + "\nSUMMARIES:\n{summaries}"
)


def _dn_words(depth: int) -> int:
    """Depth-scaled consolidation cap: roots at 400+ words parse as identity
    soup and violate the stated cap anyway (m25 qualitative review: D4
    nodes rendered 400-800 words, unioning conflicting personas). But caps
    below the D1 leaf budget (150) force fact-text loss the repair net
    cannot restore (fix-validation measured lcm 0.78->0.70 at 120/100/90);
    compression is legitimate only where children are already summaries.
    """
    return {2: 150, 3: 120}.get(depth, 100)


def _repair_consolidation(text: str, group: list[_Node]) -> str:
    """Deterministic lossless-consolidation net: children's [§ids] and
    TOPICS union that the LLM dropped get appended, not lost (m25: D4
    roots silently dropped TOPICS and orphaned the 5th model kit in a
    sibling; nothing checked the merge contract)."""
    child_ids = sorted(
        {i for n in group for i in re.findall(r"§([A-Za-z0-9_-]+)", n.summary)}
    )
    kept_ids = set(re.findall(r"§([A-Za-z0-9_-]+)", text))
    missing = [i for i in child_ids if i not in kept_ids]
    if missing:
        text += "\nRefs kept: " + ", ".join(f"§{i}" for i in missing)
    if _TOPICS_IN_PROMPTS and "TOPICS:" not in text:
        topics: dict[str, str] = {}
        for n in group:
            m = re.search(r"TOPICS:\s*(.+)", n.summary)
            if not m:
                continue
            for item in m.group(1).split(";"):
                name, _, status = item.strip().partition("|")
                if name:
                    topics[name] = status or "open"
        if topics:
            text += "\nTOPICS: " + "; ".join(
                f"{k}|{v}" for k, v in topics.items()
            )
    return text


_DATE_PREFIX = re.compile(r"^\[(\d{4}/\d{2}/\d{2})\]")


def _session_date(store, mid: str) -> str:
    """Resolve a message's session date from the loader's `[YYYY/MM/DD]`
    opener stamp (datasets/longmemeval.py stamps the first user message of
    each session). Walks backwards; returns "" when the corpus has no
    dates."""
    msgs = store.messages()
    try:
        i = next(k for k, m in enumerate(msgs) if m.id == mid)
    except StopIteration:
        return ""
    for m in reversed(msgs[: i + 1]):
        hit = _DATE_PREFIX.match(m.content)
        if hit:
            return hit.group(1)
    return ""


def render_search_results(results, store) -> str:
    """Render search hits with role and session-date attribution:
    `m000169 [user] [2023/05/22]: ...`.

    Role: without the header a hit like '...managed to raise $250 through
    sponsors...' is unattributable -- could be anyone discussed in the
    transcript -- and the answer model discards it (dev25 charity total:
    the fact was visible in the #1 snippet and never integrated).
    Date: temporal questions need the session timestamp, which lives only
    on the session opener message; annotating every hit with its resolved
    session date makes 'how many days ago' arithmetic possible without a
    second retrieval (dev100 forensics: 7/14 fresh temporal misses died
    on exactly this)."""
    lines = []
    for mid, snippet in results:
        role = "?"
        try:
            role = store.require(mid).role.value
        except KeyError:
            pass
        date = _session_date(store, mid)
        when = f" [{date}]" if date else ""
        lines.append(f"{mid} [{role}]{when}: {snippet}")
    return "\n".join(lines)


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

        # Boundary context for the next chunk: the tail of this one is
        # prepended as context so topics spanning a frontier keep their
        # anchor (context only -- messages are not re-archived).
        self._context_tail = chunk[-3:]

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
        # Boundary overlap: the previous chunk's last few messages are
        # context (CONTEXT:), not content to re-summarize as new.
        ctx = "".join(
            f"[§{m.id} {m.role.value}] {m.content}\n"
            for m in getattr(self, "_context_tail", ())
        )
        segment = ctx + ("CONTEXT ENDS --\n" if ctx else "") + "\n".join(
            f"[§{m.id} {m.role.value}] {m.content}" for m in chunk
        )
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
        depth = group[0].depth + 1
        try:
            completion = engine.llm.complete(
                [{
                    "role": "user",
                    "content": _DN_PROMPT.format(
                        summaries=summaries, words=_dn_words(depth)
                    ),
                }],
                **summary_call_kwargs(),
            )
        except Exception:
            return None
        self._overhead.add(completion.usage)
        text = completion.text.strip()
        if not text:
            return None
        text = _repair_consolidation(text, group)
        covers = tuple(mid for n in group for mid in n.covers)
        return _Node(depth=depth, summary=text, covers=covers)

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
            return render_search_results(hits, engine.store)

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
    """Incremental FTS5 index over the archive; token-overlap fallback.

    Snippets are start-anchored (~250 chars), not match-centered: messages
    lead with the subject ("I recently participated in a charity walk and
    raised $250..."), so centering on the match terms amputates exactly the
    attribution the answer model needs to USE a hit (dev25: the $250 hit
    ranked #1 with the amount visible, but the subject was outside the old
    90-char window and the model never integrated it). When the best match
    is deep in a long message, the match window is appended after the
    opening so neither context nor the hit is lost."""

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
        results: list[tuple[str, str]] = []
        # Exact-phrase boost: when the query appears verbatim in a message,
        # that message outranks token-matches. Without this, a probe or
        # artifact like `[handoff] Next:` can be buried below real content
        # that merely shares vocabulary (measured on a pi session whose
        # project keeps HANDOFF.md files: 9 colliding messages, top-5
        # truncation, the phrase match never surfaced).
        ql = re.sub(r"\s+", " ", query.strip()).lower()
        if len(ql) >= 6:
            if self._conn is not None:
                rows = self._conn.execute(
                    "SELECT id, content FROM msgs WHERE instr(lower(content), ?) "
                    "LIMIT ?",
                    (ql, limit),
                ).fetchall()
                results = [(r[0], self._phrase_snippet(r[1], ql)) for r in rows]
            else:
                ql_norm = ql
                for mid, content in self._fallback:
                    if ql_norm in content.lower():
                        results.append((mid, self._phrase_snippet(content, ql_norm)))
                        if len(results) >= limit:
                            break
        if len(results) >= limit:
            return results
        if self._conn is not None:
            fts_query = self._fts_query(query)
            if fts_query:
                try:
                    cursor = self._conn.execute(
                        "SELECT id, content FROM msgs WHERE msgs MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (fts_query, limit * 2),
                    )
                    seen = {mid for mid, _ in results}
                    for mid, content in cursor:
                        if mid not in seen:
                            results.append((mid, self._snippet(content, query)))
                            if len(results) >= limit:
                                break
                    return results
                except sqlite3.Error:
                    pass  # malformed MATCH syntax: fall through to overlap
        return results + self._overlap_search(query, limit)

    SNIPPET_OPENING = 250  # start-anchored context chars
    SNIPPET_MATCH = 110  # appended window when the match is deep

    @classmethod
    def _snippet(cls, content: str, query: str) -> str:
        c = " ".join(content.split())
        opening = (
            c if len(c) <= cls.SNIPPET_OPENING
            else c[: cls.SNIPPET_OPENING].rsplit(" ", 1)[0] + "…"
        )
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        # find the first strong match term beyond the opening window
        low = c.lower()
        deep = next(
            (
                low.find(t, cls.SNIPPET_OPENING - 50)
                for t in sorted(terms, key=len, reverse=True)
                if low.find(t, cls.SNIPPET_OPENING - 50) != -1
            ),
            None,
        )
        if deep is not None and deep >= cls.SNIPPET_OPENING:
            start = max(0, deep - cls.SNIPPET_MATCH // 3)
            window = c[start : start + cls.SNIPPET_MATCH]
            return f"{opening} …[later] {window}"
        return opening

    @staticmethod
    def _phrase_snippet(content: str, phrase: str) -> str:
        # kept for phrase-boost marking: start-anchored, phrase highlighted
        opening = _FtsIndex._snippet(content, phrase)
        mark = phrase
        return opening.replace(mark, f">{mark}<", 1)

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
