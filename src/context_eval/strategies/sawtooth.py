"""Sawtooth schedule with interchangeable backs — isolates the back of the tooth.

The R9 framing made experimental: the sawtooth (fill at full fidelity,
delta-only prefill, one deep cut) is the *schedule* and is held identical
across four arms; the only variable is what the cut leaves resident.

Held constant for every mode:
- trigger: fire when BACK_TOKENS + unconcluded messages would exceed the
  budget (the back is priced at its CAP, not its current size, so tooth
  boundaries are bit-identical across modes on the same transcript)
- head: the last HEAD_TOKENS of messages, verbatim, message-aligned
- archive: everything cut is indexed verbatim; every mode ships the same
  history_search / history_recall (cued recall is a control, not a variable)
- back budget: BACK_TOKENS cap on the resident memory block

Varies — the back's content semantics:
- none      no back at all (the floor: head + archive only)
- rewrite   ONE incremental fold per tooth, prior back + span -> new back
            (flat-compact semantics under the fixed cadence)
- segments  per-chunk summaries appended once, never re-derived; the most
            recent ones that fit the cap stay resident (append-only,
            unstructured — isolates append-onlyness from typed structure)
- ledger    typed TOPIC|status|fact observations appended once plus the
            mechanical latest-wins inventory (conclusions semantics)
"""

from __future__ import annotations

from context_eval.provider import ToolSchema
from context_eval.strategies.base import Engine, EngineTool, Strategy, StrategyStats
from context_eval.strategies.conclusions import (
    _BULLET,
    _EXTRACT_PROMPT,
    _FtsIndex,
    _Record,
)
from context_eval.strategies.flat_compact import _D1_PROMPT
from context_eval.strategies.lcm import summary_call_kwargs
from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS
from context_eval.types import PromptMessage, Role, Usage

HEAD_TOKENS = 12_000  # verbatim head, identical across modes
BACK_TOKENS = 4_096  # resident memory block cap, identical across modes
MAX_SPAN_CHARS = 64_000  # per-call extraction/summary chunk (chars)

_HEADER = "[Session memory — append-only archive is searchable via history_search]"


class SawtoothBack(Strategy):
    """One class, four registered names; ``mode`` is the only difference."""

    def __init__(self, mode: str, name: str) -> None:
        assert mode in ("none", "rewrite", "segments", "ledger")
        self.mode = mode
        self.name = name
        self._concluded_upto = 0  # messages below this are archived + backed
        self._teeth = 0
        self._empty = 0
        self._overhead = Usage()
        self._index = _FtsIndex()
        # mode state
        self._back_text = ""  # rewrite: the folded doc
        self._segment_summaries: list[str] = []  # segments: appended chunks
        self._records: list[_Record] = []  # ledger: appended observations

    # -- schedule (mode-independent) ------------------------------------------

    def _head_start(self, engine: Engine) -> int:
        """First store index of the verbatim head (message-aligned)."""
        messages = engine.store.messages()
        used, start = 0, len(messages)
        for i in range(len(messages) - 1, self._concluded_upto - 1, -1):
            cost = engine.counter.count_message(messages[i])
            if used + cost > HEAD_TOKENS:
                break
            used += cost
            start = i
        return start

    def on_turn(self, engine: Engine) -> None:
        messages = engine.store.messages()
        if len(messages) > self._index.indexed_upto:
            self._index.index(messages[self._index.indexed_upto :])
        # Price the back at its cap so tooth boundaries cannot depend on
        # back content — cadence identity across modes is the experiment.
        projected = BACK_TOKENS + MESSAGE_OVERHEAD_TOKENS + sum(
            engine.counter.count_message(m) for m in messages[self._concluded_upto :]
        )
        if projected <= engine.budget.max_context_tokens:
            return
        head_start = self._head_start(engine)
        span = messages[self._concluded_upto : head_start]
        if not span:
            return  # head alone fills the budget: nothing to cut
        self._cut(engine, span)
        self._concluded_upto = head_start
        self._teeth += 1

    def _segment_text(self, span) -> str:
        parts = []
        for m in span:
            parts.append(f"§{m.id} [{m.role.value}]\n{m.content}\n\n")
        return "".join(parts)

    def _chunks(self, span) -> list[str]:
        text = self._segment_text(span)
        return [text[i : i + MAX_SPAN_CHARS] for i in range(0, len(text), MAX_SPAN_CHARS)]

    def _complete(self, engine: Engine, prompt: str) -> str:
        completion = engine.llm.complete(
            [{"role": "user", "content": prompt}], **summary_call_kwargs()
        )
        if completion.usage:
            self._overhead.add(completion.usage)
        text = (completion.text or "").strip()
        if not text and completion.finish_reason == "length":
            self._empty += 1
        return text

    # -- the cut: build the back per mode --------------------------------------

    def _cut(self, engine: Engine, span) -> None:
        if self.mode == "none":
            return
        if self.mode == "rewrite":
            parts = []
            if self._back_text:
                parts.append(f"PREVIOUS SUMMARY:\n{self._back_text}")
                parts.append("NEW HISTORY SINCE:")
            parts.append(self._segment_text(span))
            prompt = _D1_PROMPT.format(segment="\n".join(parts))
            folded = self._complete(engine, prompt)
            if folded:
                self._back_text = folded
            return
        if self.mode == "segments":
            for chunk in self._chunks(span):
                summary = self._complete(engine, _D1_PROMPT.format(segment=chunk))
                if summary:
                    self._segment_summaries.append(summary)
            return
        # ledger
        for chunk in self._chunks(span):
            text = self._complete(engine, _EXTRACT_PROMPT.format(segment=chunk))
            if not text:
                continue
            for line in text.splitlines():
                m = _BULLET.match(line.strip())
                if m:
                    self._records.append(_Record(len(self._records), m.group(1)))

    # -- inventory (ledger only; deterministic, zero LLM) -----------------------

    def _inventory(self) -> str:
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

    # -- view --------------------------------------------------------------------

    def _back_block(self, engine: Engine) -> str:
        counter = engine.counter
        if self.mode == "none":
            return ""
        if self.mode == "rewrite":
            return f"{_HEADER}\n{self._back_text}" if self._back_text else ""
        if self.mode == "segments":
            lines: list[str] = []
            used = 0
            for s in reversed(self._segment_summaries):
                cost = counter.count_text(s) + 1
                if used + cost > BACK_TOKENS and lines:
                    break
                lines.append(s)
                used += cost
            if not lines:
                return ""
            return f"{_HEADER}\n" + "\n---\n".join(reversed(lines))
        # ledger: inventory always resident; newest bullets fill the rest
        parts: list[str] = []
        inventory = self._inventory()
        if inventory:
            parts.append(inventory)
        used = counter.count_text(inventory) if inventory else 0
        resident: list[str] = []
        for r in reversed(self._records):
            cost = r.tokens(counter) + 8
            if used + cost > BACK_TOKENS and resident:
                break
            resident.append(f"- {r.raw}")
            used += cost
        if not parts and not resident:
            return ""
        return f"{_HEADER}\n" + "\n".join([*parts, *reversed(resident)])

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        view: list[PromptMessage] = []
        block = self._back_block(engine)
        if block:
            view.append(PromptMessage(role=Role.TOOL, content=block))
        view.extend(
            m.to_prompt() for m in engine.store.messages()[self._concluded_upto :]
        )
        return view

    # -- tools: identical for every mode (the back is the only variable) ---------

    def tools(self, engine: Engine) -> list[EngineTool]:
        archive = ToolSchema(
            name="history_search",
            description=(
                "Search the verbatim archive of this session (including "
                "compacted segments). Returns matching message ids with "
                "snippets; history_recall fetches full content by id."
            ),
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "search terms"}},
                "required": ["query"],
            },
        )
        recall = ToolSchema(
            name="history_recall",
            description="Return the full original content of a message by id.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "message id, e.g. m000014"
                    }
                },
                "required": ["id"],
            },
        )

        def do_search(args):
            results = self._index.search(str(args.get("query", "")), limit=5)
            if not results:
                return "no matches"
            return "\n".join(f"{mid}: {snippet}" for mid, snippet in results)

        def do_recall(args):
            try:
                return engine.store.require(str(args.get("id", ""))).content
            except KeyError:
                return "no such message id"

        return [
            EngineTool(schema=archive, execute=do_search),
            EngineTool(schema=recall, execute=do_recall),
        ]

    def stats(self) -> StrategyStats:
        return StrategyStats(
            overhead_usage=self._overhead,
            compactions=self._teeth,
            empty_summaries=self._empty,
            recall_tool_calls=0,
        )


class SawtoothNone(SawtoothBack):
    def __init__(self) -> None:
        super().__init__("none", "sawtooth-none")


class SawtoothRewrite(SawtoothBack):
    def __init__(self) -> None:
        super().__init__("rewrite", "sawtooth-rewrite")


class SawtoothSegments(SawtoothBack):
    def __init__(self) -> None:
        super().__init__("segments", "sawtooth-segments")


class SawtoothLedger(SawtoothBack):
    def __init__(self) -> None:
        super().__init__("ledger", "sawtooth-ledger")
