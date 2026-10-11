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
    _IDS,
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

# Tuned extraction (v2): naming continuity against the running inventory,
# per-topic budget, and action-weighted phrasing. Targets the qualitative
# defects measured on the v1 ledger: 119 topic keys for ~4 real projects,
# 194/342 records on one topic, resolved-heavy history crowding out live
# state, 38-42 tokens/record.
_EXTRACT_PROMPT_V2 = """You extract durable observations from one segment of an \
agent session (coding, chat, or mixed), appending to an existing ledger.
Output ONLY a bullet list, one observation per line, each exactly:
- TOPIC | status | the observation [§<ids>]
Rules:
- TOPIC must be a project/codename/decision, one short canonical name. \
Reuse a KNOWN TOPIC name verbatim when the observation belongs to it; \
create a new name only for genuinely new work.
- status is one of open, resolved, paused, in progress.
- At most ~5 lines per topic in this segment: keep the observations that \
would most change what the agent does next; fold related details into one \
line. Prefer open/in-progress state and decisions over historical \
measurements; a resolved topic needs at most its outcome.
- State observations in ~20 words; every specific value that gates future \
work (ids, numbers, versions, paths) stays. Never quote the \
transcript.
- A one-line topic/status mention by the user is itself a durable \
observation: record it as its own bullet even when nothing else happened.
- A topic reappearing later is a new line; statuses supersede by recency.
{known_topics}SEGMENT:
{segment}"""

# v3 nudge, from the ledger2 mechanism check on 5a1e: naming continuity
# worked (147 distinct topics -> 4) but over-consolidated real workstreams
# (~5 true projects) into two umbrella keys, and the ~20-word rule did not
# compress (40 tok/record measured). Distinct-workstream naming + a hard
# 15-word cap.
_EXTRACT_PROMPT_V3 = _EXTRACT_PROMPT_V2.replace(
    "create a new name only for genuinely new work.",
    "distinct workstreams keep their own names (a port, a quant tier, a "
    "benchmark is its own topic); a new name is wrong only if the line "
    "would already make sense under a KNOWN TOPIC.",
).replace("in ~20 words", "in at most 15 words")

_KNOWN_TOPICS = """KNOWN TOPICS so far (reuse these exact names where they fit):
{topics}

"""

_HEADER = "[Session memory — append-only archive is searchable via history_search]"


class SawtoothBack(Strategy):
    """One class, four registered names; ``mode`` is the only difference."""

    def __init__(self, mode: str, name: str) -> None:
        assert mode in ("none", "rewrite", "segments", "ledger", "ledger2", "ledger3")
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
            if self.mode in ("ledger2", "ledger3"):
                inventory = self._inventory()
                known = _KNOWN_TOPICS.format(
                    topics=", ".join(t for t, _ in sorted(
                        self._topic_latest().items(), key=lambda kv: -kv[1][0]
                    )[:40])
                ) if inventory else ""
                prompt = (_EXTRACT_PROMPT_V3 if self.mode == "ledger3"
                          else _EXTRACT_PROMPT_V2).format(
                    segment=chunk, known_topics=known
                )
            else:
                prompt = _EXTRACT_PROMPT.format(segment=chunk)
            text = self._complete(engine, prompt)
            if not text:
                continue
            for line in text.splitlines():
                m = _BULLET.match(line.strip())
                if m:
                    self._records.append(_Record(len(self._records), m.group(1)))

    # -- inventory (ledger modes; deterministic, zero LLM) -----------------------

    def _topic_latest(self) -> dict[str, tuple[int, str]]:
        latest: dict[str, tuple[int, str]] = {}
        for r in self._records:
            if not r.topic:
                continue
            key = r.topic.casefold()
            if key not in latest or r.seq > latest[key][0]:
                latest[key] = (r.seq, r.status)
        return latest

    def _inventory(self) -> str:
        latest = self._topic_latest()
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
        # ledger: inventory always resident; bullets fill the rest — topic-
        # balanced for v2 (round-robin across topics, newest first, so a
        # prolific topic cannot evict every other), global-recency for v1
        parts: list[str] = []
        inventory = self._inventory()
        if inventory:
            parts.append(inventory)
        used = counter.count_text(inventory) if inventory else 0
        if self.mode in ("ledger2", "ledger3"):
            by_topic: dict[str, list[_Record]] = {}
            for r in self._records:
                by_topic.setdefault(r.topic.casefold(), []).append(r)
            rounds: list[_Record] = []
            waves = [list(v) for v in by_topic.values()]
            for wave in waves:
                wave.reverse()  # newest first within topic
            while any(waves):
                for wave in waves:
                    if wave:
                        rounds.append(wave.pop(0))
            chosen: list[_Record] = []
            for r in rounds:
                disp = _Record(r.seq, _IDS.sub("", r.raw).strip())
                cost = disp.tokens(counter) + 8
                if used + cost > BACK_TOKENS and chosen:
                    break
                chosen.append(disp)
                used += cost
            if not parts and not chosen:
                return ""
            return f"{_HEADER}\n" + "\n".join([*parts, *(f"- {r.raw}" for r in chosen)])
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


class SawtoothLedger2(SawtoothBack):
    def __init__(self) -> None:
        super().__init__("ledger2", "sawtooth-ledger2")


class SawtoothLedger3(SawtoothBack):
    def __init__(self) -> None:
        super().__init__("ledger3", "sawtooth-ledger3")
