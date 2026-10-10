"""Conclusions memory: append-only observation store + mechanical inventory.

The Honcho-v3 shape made testable: extraction calls *append* typed
conclusions (never recompress — the failure mode of whole-doc memory
rewrites, measured at 39-68% inventory erosion), the topic inventory is
assembled deterministically in code (zero churn, always current — the R3
instruction win without an LLM), details stay searchable (need-driven
recovery, R5), and the verbatim tail rides at the effective band.

View: [inventory (always resident)] + [bounded recent conclusions] +
[verbatim tail]. Loss events are extraction calls only — small, bounded,
calibrated (reasoning-off, stated bullet budget, require-content).
"""

from __future__ import annotations

import re

from context_eval.provider import ToolSchema
from context_eval.strategies.base import Engine, EngineTool, Strategy, StrategyStats
from context_eval.strategies.lcm import _FtsIndex, summary_call_kwargs
from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS
from context_eval.types import PromptMessage, Role, Usage

TAU = 0.90  # extract when the assembled view exceeds this share of budget
KEEP_TAIL_FRAC = 0.25  # verbatim tail, as a fraction of budget
RESIDENT_TOKENS = 2048  # cap on the conclusions block (inventory excluded)
MAX_SPAN_CHARS = int(__import__("os").environ.get("CONTEXT_EVAL_SPAN_CHARS", "64000"))

_EXTRACT_PROMPT = """You extract durable conclusions from one segment of an \
agent session (coding, chat, or mixed).
Output ONLY a bullet list, one conclusion per line, each exactly:
- TOPIC | status | the conclusion [§<ids>]
Rules: TOPIC is the named project/codename/decision this belongs to \
(single short name). status is one of open, resolved, paused, in progress. \
State conclusions; never quote the transcript. Include every specific \
value (ids, numbers, versions, paths, names). A topic reappearing in a \
later segment is a new line — statuses supersede by recency. Keep the \
whole output under ~30 bullets / ~150 words.
SEGMENT:
{segment}"""

_BULLET = re.compile(r"^[-*]\s*(.+)$")
# topic | status | text [§ids] — tolerant: missing status/ids still parse
_ROW = re.compile(
    r"^(?P<topic>[^|]{1,60}?)\s*\|\s*"
    r"(?P<status>open|resolved|paused|in progress)\s*\|\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_IDS = re.compile(r"§([A-Za-z0-9_-]+)")


class _Record:
    __slots__ = ("seq", "topic", "status", "text", "ids", "raw")

    def __init__(self, seq: int, line: str) -> None:
        self.seq = seq
        self.raw = line
        m = _ROW.match(line.strip())
        if m:
            self.topic = m.group("topic").strip()
            self.status = m.group("status").lower()
            rest = m.group("rest")
        else:
            self.topic, self.status, rest = "", "open", line.strip()
        self.ids = tuple(i.lstrip("§") for i in _IDS.findall(rest))
        self.text = _IDS.sub("", rest).strip()

    def tokens(self, counter) -> int:
        return counter.count_text(self.raw)


class ConclusionsMemory(Strategy):
    name = "conclusions"

    def __init__(self, tau: float = TAU, keep_tail_frac: float = KEEP_TAIL_FRAC) -> None:
        self._tau = tau
        self._keep_tail_frac = keep_tail_frac
        self._records: list[_Record] = []
        self._concluded_upto = 0  # store messages below this index are extracted
        self._index = _FtsIndex()  # verbatim archive (parity with flat/lcm)
        self._compactions = 0
        self._empty = 0
        self._search_calls = 0
        self._overhead = Usage()

    # -- write path ----------------------------------------------------------

    def on_turn(self, engine: Engine) -> None:
        counter = engine.counter
        budget = engine.budget.max_context_tokens
        messages = engine.store.messages()
        if len(messages) > self._index.indexed_upto:
            self._index.index(messages[self._index.indexed_upto :])
        # Trigger on the projected view (unconcluded messages, tail
        # included) — matching how deployed agents compact on projected
        # context. Raw-total triggers thrash one call per turn once the
        # transcript passes budget (the hang this fixed).
        projected = sum(counter.count_message(m) for m in messages[self._concluded_upto :])
        if projected <= self._tau * budget:
            return
        tail_start = self._tail_start(engine)
        span = messages[self._concluded_upto : tail_start]
        if not span:
            return  # tail itself over budget: nothing new to conclude
        segment = ""
        for m in span:
            segment += f"§{m.id} [{m.role.value}]\n{m.content}\n\n"
        for chunk_start in range(0, len(segment), MAX_SPAN_CHARS):
            chunk = segment[chunk_start : chunk_start + MAX_SPAN_CHARS]
            self._extract(engine, chunk)
        self._concluded_upto = tail_start

    def _extract(self, engine: Engine, segment: str) -> None:
        self._compactions += 1
        try:
            completion = engine.llm.complete(
                [{"role": "user", "content": _EXTRACT_PROMPT.format(segment=segment)}],
                **summary_call_kwargs(),
            )
        except Exception:
            return  # fail-open: span stays verbatim until the next trigger
        self._overhead.add(completion.usage)
        text = (completion.text or "").strip()
        if not text:
            self._empty += 1
            return
        for line in text.splitlines():
            m = _BULLET.match(line.strip())
            if not m:
                continue
            self._records.append(_Record(len(self._records), m.group(1)))

    # -- inventory (deterministic, zero LLM) ----------------------------------

    def _inventory(self) -> str:
        """Mechanical TOPICS line: dedupe by name, latest status wins."""
        latest: dict[str, tuple[int, str]] = {}
        for r in self._records:
            if not r.topic:
                continue
            key = r.topic.casefold()
            if key not in latest or r.seq > latest[key][0]:
                latest[key] = (r.seq, r.status)
        if not latest:
            return ""
        ordered = sorted(latest.items(), key=lambda kv: -kv[1][0])
        return "TOPICS: " + "; ".join(f"{t}|{s}" for t, (_, s) in ordered)

    # -- view ------------------------------------------------------------------

    def _tail_start(self, engine: Engine) -> int:
        budget = max(1, int(engine.budget.max_context_tokens * self._keep_tail_frac))
        messages = engine.store.messages()
        used, start = 0, len(messages)
        for i in range(len(messages) - 1, -1, -1):
            cost = engine.counter.count_message(messages[i])
            if used + cost > budget:
                break
            used += cost
            start = i
        return start

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        counter = engine.counter
        messages = engine.store.messages()
        tail_start = self._tail_start(engine)
        inventory = self._inventory()

        # Resident bullets: newest first under the token cap; older ones
        # stay searchable but leave the view.
        resident: list[_Record] = []
        used = 0
        for r in reversed(self._records):
            cost = r.tokens(counter) + 8
            if used + cost > RESIDENT_TOKENS and resident:
                break
            resident.append(r)
            used += cost
        resident.reverse()

        lines = ["[Session conclusions — append-only; conclusions_search finds older ones]"]
        if inventory:
            lines.append(inventory)
        lines += [f"- {r.raw}" for r in resident]
        block = "\n".join(lines)

        view: list[PromptMessage] = [
            PromptMessage(role=Role.TOOL, content=block)
        ]
        # Enforce the budget: shed oldest resident bullets if needed.
        budget = engine.budget.max_context_tokens
        head = counter.count_text(block) + MESSAGE_OVERHEAD_TOKENS
        tail_cost = sum(counter.count_message(m) for m in messages[tail_start:])
        while resident and head + tail_cost > budget:
            r = resident.pop(0)
            block = block.replace(f"- {r.raw}\n", "", 1)
            head = counter.count_text(block) + MESSAGE_OVERHEAD_TOKENS
        view[0] = PromptMessage(role=Role.TOOL, content=block)
        view += [m.to_prompt() for m in messages[tail_start:]]
        return view

    # -- tools -------------------------------------------------------------------

    def tools(self, engine: Engine) -> list[EngineTool]:
        schema = ToolSchema(
            name="conclusions_search",
            description=(
                "Search all extracted session conclusions (including ones "
                "no longer in view) by keyword overlap. Returns the top 5 "
                "as [n] TOPIC|status|conclusion lines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "keywords to match"}
                },
                "required": ["query"],
            },
        )

        def execute(args):
            self._search_calls += 1
            q = {t for t in re.findall(r"[a-z0-9]+", str(args.get("query", "")).lower())}
            if not q:
                return "error: empty query"
            scored = sorted(
                ((len(q & set(re.findall(r"[a-z0-9]+", r.raw.lower()))), -r.seq, r)
                 for r in self._records),
                reverse=True,
            )
            hits = [r for score, _, r in scored[:5] if score > 0]
            if not hits:
                return "no matching conclusions"
            return "\n".join(f"[{i}] {r.raw}" for i, r in enumerate(hits))

        # Verbatim archive parity: without history_search the model
        # hammer-searches thin conclusions, exhausts the tool budget, and
        # answers blank (measured: ~25% blank answers on the smoke).
        archive = ToolSchema(
            name="history_search",
            description=(
                "Search the verbatim archive of this session (including "
                "compacted segments). Returns matching message ids with "
                "snippets; history_recall fetches full content by id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search terms"}
                },
                "required": ["query"],
            },
        )

        def do_search(args):
            results = self._index.search(str(args.get("query", "")), limit=5)
            if not results:
                return "no matches"
            return "\n".join(f"{mid}: {snippet}" for mid, snippet in results)

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

        def do_recall(args):
            try:
                return engine.store.require(str(args.get("id", ""))).content
            except KeyError:
                return "no such message id"

        return [
            EngineTool(schema=schema, execute=execute),
            EngineTool(schema=archive, execute=do_search),
            EngineTool(schema=recall, execute=do_recall),
        ]

    def stats(self) -> StrategyStats:
        return StrategyStats(
            overhead_usage=self._overhead,
            compactions=self._compactions,
            empty_summaries=self._empty,
            recall_tool_calls=self._search_calls,
        )
