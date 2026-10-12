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
- ledger5   resident memory per docs/resident-memory-theory.md: a
            workstream/topic tree with one activation law, a rung ladder,
            an exposed forgetting surface, and a collection-time recording
            schema that makes every dial re-renderable offline
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

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

# v4: status is a deliberate act. v1-v3 let every line vote on topic
# status by recency; a janitorial note ("stale comment remains") could
# out-vote "tq4 ships as production tier" three lines up. Notes now carry
# no status; only explicit state-change lines move the inventory. Resident
# block is topic-grouped (reads like a changelog, not 30 interleaved
# rows); near-duplicate restatements are suppressed at append.
# Add long-line exemptions for the embedded prompts (line-per-rule format
# mirrors the deriver prompts this ports; wrapping would churn the diff).
# ruff: noqa: E501
_EXTRACT_PROMPT_V4 = """You maintain the ledger of an agent session (coding, chat, or mixed), appending observations from one segment.
Output ONLY a bullet list, one per line, in exactly one of two forms:
- TOPIC | note | the observation [§<ids>]
- TOPIC | <status> | what changed and why [§<ids>]
Rules:
- The state-change form (second form) is reserved for deliberate state changes (status: open, resolved, paused, in progress). Emit it when a topic's state actually changed in this segment — a decision landed, work shipped, something blocked or set aside — and say what and why. Everything else is the first form: findings, measurements, next steps, agreements. Never stamp a status on a note.
- TOPIC is one short canonical name per workstream; reuse KNOWN TOPICS verbatim (a port, a tier, a benchmark is its own topic). Never mark an umbrella project resolved because a sub-stream completed.
- At most ~5 lines per topic in this segment; keep what would most change what the agent does next; fold related details into one line.
- EXCEPTION, never skipped and exempt from the budget: a one-line topic/status mention by the user (e.g. "the admin revamp is 'corvid', paused") is ALWAYS its own line — TOPIC is the codename; state-change form when a status is stated — even when nothing else happened with the topic.
- At most 15 words; keep every value that gates future work (ids, numbers, versions, paths). Never quote the transcript.
- A topic reappearing in a later segment is a new line.
{known_topics}SEGMENT:
{segment}"""

_V4_ROW = re.compile(
    r"^(?P<topic>[^|]{1,60}?)\s*\|\s*"
    r"(?P<kind>[^|]{1,20})\s*\|\s*"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
# citation matcher local to v4: eats comma lists ('[§a, 12, b]') whole;
# the shared _IDS is frozen with the v1-v3 arms and must not change
_IDS_V4 = re.compile(r"§[A-Za-z0-9_-]+(?:\s*,\s*[A-Za-z0-9_-]+)*")


def _heartbeat(mode: str, tooth: int, added: int, empty: int) -> None:
    """One stderr line per extraction call. Long replays otherwise print
    nothing until the sample completes (an hour+ at mega depth), hiding
    the documented silent-failure signature: 429 exhaustion / blank
    completions produce a zero record an hour later instead of a
    visible stall now."""
    import sys

    print(
        f"[{mode}] tooth {tooth}: +{added} lines"
        + (f" ({empty} EMPTY)" if empty else ""),
        file=sys.stderr,
        flush=True,
    )

_KNOWN_TOPICS = """KNOWN TOPICS so far (reuse these exact names where they fit):
{topics}

"""


class _Ledger4Line:
    __slots__ = ("seq", "topic", "kind", "text", "tokens_cache")

    def __init__(self, seq: int, line: str) -> None:
        self.seq = seq
        m = _V4_ROW.match(line.strip().lstrip("-*").strip())
        if m:
            # extractors sometimes bold the name ('- **kvarn** | open | ...');
            # keep the asterisks out of the topic key or '**' aliases fork
            # every bolded line into a phantom topic
            self.topic = m.group("topic").strip().strip("*_").strip()
            self.kind = m.group("kind").strip().lower()
            rest = m.group("rest")
        else:
            self.topic, self.kind, rest = "", "note", line.strip()
        if self.kind not in ("note", "open", "resolved", "paused", "in progress"):
            # off-vocabulary status word ('shipped', 'done', ...): keep the
            # observation as a note rather than dropping the whole line
            rest = f"{self.kind}: {rest}"
            self.kind = "note"
        # v4-local id matcher: also eats comma lists ('[§a, b, c]') that the
        # shared _IDS (frozen with v1-v3) leaves half-stripped
        self.text = _IDS_V4.sub("", rest)
        self.text = re.sub(r"\[[\s,]*\]", "", self.text)  # emptied id groups
        self.text = re.sub(r"\s{2,}", " ", self.text).strip(" ,;")
        self.tokens_cache = None

    @property
    def is_state_change(self) -> bool:
        return self.kind in ("open", "resolved", "paused", "in progress")

    def status(self) -> str:
        return self.kind

    def tokens(self, counter) -> int:
        if self.tokens_cache is None:
            self.tokens_cache = counter.count_text(self.text)
        return self.tokens_cache

    def wordset(self) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", self.text.lower()))

_HEADER = "[Session memory — append-only archive is searchable via history_search]"
_HEADER_V4 = ("[Session memory — an append-only index of observed topics. "
              "NOT exhaustive: history_search the verbatim archive before "
              "concluding anything never happened]")


class SawtoothBack(Strategy):
    """One class, seven registered names; ``mode`` is the only difference."""

    def __init__(self, mode: str, name: str) -> None:
        assert mode in (
            "none", "rewrite", "segments", "ledger", "ledger2", "ledger3", "ledger4", "ledger5",
        )
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
        self._lines: list[_Ledger4Line] = []  # ledger4: typed lines
        self._topic_canon: dict[str, str] = {}  # ledger4: alias -> canonical
        # ledger5 state (resident memory; see docs/resident-memory-theory.md)
        self._dials5 = Ledger5Dials()
        self._lines5: list[_Ledger5Line] = []
        self._uses5: dict[str, list[int]] = {}  # topic key -> use teeth
        self._ws_canon5: dict[str, str] = {}
        self._topic_canon5: dict[str, str] = {}
        self._chunks5: list[dict] = []
        self._archived_tokens5 = 0
        self._rec5: dict = {
            "version": 1,
            "strategy": "sawtooth-ledger5",
            "dials": asdict(self._dials5),
            "back_tokens_cap": BACK_TOKENS,
            "teeth": [],
            "lines": [],
            "uses": [],
            "searches": [],
            "chunks": [],
            "renders": [],
            "rungs": [],
        }

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
                # the fold must fit the price the trigger pays for it
                # (BACK_TOKENS); a runaway fold would overflow the view
                cap = BACK_TOKENS - engine.counter.count_text(_HEADER) - 2
                while engine.counter.count_text(folded) > cap and len(folded) > 100:
                    folded = folded[: int(len(folded) * 0.9)].rsplit(".", 1)[0]
                self._back_text = folded
            return
        if self.mode == "segments":
            for chunk in self._chunks(span):
                summary = self._complete(engine, _D1_PROMPT.format(segment=chunk))
                if summary:
                    self._segment_summaries.append(summary)
            return
        # ledger family
        if self.mode == "ledger5":
            for chunk in self._chunks(span):
                self._v5_extract_chunk(engine, chunk)
            self._v5_audit(engine)
            self._rec5_tooth(engine, span)
            return
        if self.mode == "ledger4":
            for chunk in self._chunks(span):
                known = self._known_topics_v4()
                text = self._complete(
                    engine, _EXTRACT_PROMPT_V4.format(segment=chunk, known_topics=known)
                )
                added, empty = 0, 0
                if not text:
                    empty = 1
                for line in (text or "").splitlines():
                    m = _BULLET.match(line.strip())
                    if not m:
                        continue
                    new = _Ledger4Line(len(self._lines), m.group(1))
                    if not new.topic:
                        continue
                    # near-duplicate suppression: same topic, content
                    # already ~covered by a recent line -> skip the append.
                    # A status transition is never suppressed (recency and
                    # wording carry no status signal -- suppressing a flip
                    # with recycled wording would strand the inventory in
                    # the stale status); a same-status state change may
                    # still dedupe against its own restatement.
                    words = new.wordset()
                    if words and any(
                        len(words & old.wordset()) / max(1, len(words)) >= 0.8
                        and old.topic == new.topic
                        and (
                            not new.is_state_change
                            or (old.is_state_change and old.kind == new.kind)
                        )
                        for old in self._lines[-60:]
                    ):
                        continue
                    new.topic = self._canonical_topic(new.topic)
                    self._lines.append(new)
                    added += 1
                _heartbeat(self.mode, self._teeth, added, empty)
            return
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

    def _known_topics_v4(self, cap: int = 100) -> str:
        """Vocabulary block passed to each v4 extraction call. Ranked by
        last ACTIVITY (any line), not last state change -- a resolved
        workstream must not sink below recent noise and lose its name,
        or the extractor coins variants and the inventory forks (166
        keys for ~30 workstreams at 115 teeth under the old top-40,
        last-state-change ranking). Cap 100 costs ~500 prompt tokens
        against a ~16K chunk: cheap insurance for naming continuity.
        """
        if not self._lines:
            return ""
        last: dict[str, int] = {}
        display: dict[str, str] = {}
        for ln in self._lines:
            key = ln.topic.casefold()
            last[key] = max(last.get(key, 0), ln.seq)
            display.setdefault(key, ln.topic)
        topics = ", ".join(
            display[k] for k in sorted(last, key=lambda k: -last[k])[:cap]
        )
        return _KNOWN_TOPICS.format(topics=topics)

    def _canonical_topic(self, name: str) -> str:
        """Normalize topic names to one canonical key: case variants and
        formatting artifacts fold to the first-seen spelling. Word-set
        extension folding was tried and rejected: no threshold separates
        an alias ('kvarn port' after 'kvarn') from a sibling sub-stream
        ('kvarn gate'), and wrongly absorbing the latter is worse than
        staying split. Naming continuity is the vocabulary's job.
        """
        key = name.casefold()
        if key not in self._topic_canon:
            self._topic_canon[key] = name
        return self._topic_canon[key]

    def _topic_state4(self) -> dict[str, tuple[int, str]]:
        """Per-topic status: only explicit state-change lines move it; a
        topic seen only in notes defaults to open."""
        state: dict[str, tuple[int, str]] = {}
        for ln in self._lines:
            key = ln.topic.casefold()
            if ln.is_state_change:
                state[key] = (ln.seq, ln.kind)
            elif key not in state:
                state[key] = (ln.seq, "open")
        return state

    def _inventory4(self) -> str:
        state = self._topic_state4()
        if not state:
            return ""
        ordered = sorted(state.items(), key=lambda kv: -kv[1][0])
        return "TOPICS: " + "; ".join(f"{t}|{s}" for t, (_, s) in ordered)

    # -- view --------------------------------------------------------------------

    def _back_block(self, engine: Engine) -> str:
        counter = engine.counter
        if self.mode == "none":
            return ""
        if self.mode == "ledger5":
            return self._back_block5(engine)
        if self.mode == "ledger4":
            return self._back_block4(engine)
        if self.mode == "rewrite":
            return f"{_HEADER}\n{self._back_text}" if self._back_text else ""
        if self.mode == "segments":
            lines: list[str] = []
            used = engine.counter.count_text(_HEADER) + 2
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
        # guard: the TOPICS line is uncapped in v1-v3 and overflows the
        # priced back beyond ~340 topics (live: mega-b died at turn 38717).
        # Cap entries at ~1.5K tokens; below that (every recorded grid
        # regime) the block is byte-identical to before.
        entries = inventory.split("; ")
        while len(entries) > 2 and counter.count_text("; ".join(entries)) > 1536:
            entries.pop()
        inventory = "; ".join(entries)
        if inventory:
            parts.append(inventory)
        used = counter.count_text(_HEADER) + 2 + (
            counter.count_text(inventory) if inventory else 0
        )
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

    def _back_block4(self, engine: Engine) -> str:
        """Topic-grouped resident block: TOPICS line, then per topic a
        status header and its newest notes. Topics ordered by recency of
        their newest line; lines filled round-robin across topics so no
        single workstream evicts the rest."""
        counter = engine.counter
        state = self._topic_state4()
        if not state:
            return ""
        by_topic: dict[str, list[_Ledger4Line]] = {}
        for ln in self._lines:
            by_topic.setdefault(ln.topic.casefold(), []).append(ln)
        recency = sorted(
            by_topic, key=lambda t: max(ln.seq for ln in by_topic[t]), reverse=True
        )
        waves = [list(reversed(by_topic[t])) for t in recency]  # newest first
        chosen: list[_Ledger4Line] = []
        # guard: a runaway TOPICS line must not outgrow the priced back
        # (the trigger assumes the block fits BACK_TOKENS; overflow kills
        # the run). Drop oldest entries past ~1.5K.
        entries = self._inventory4().split("; ")
        while len(entries) > 2 and counter.count_text("; ".join(entries)) > 1536:
            entries.pop()
        inventory = "; ".join(entries)
        # Exact accounting: the trigger prices this block at BACK_TOKENS,
        # so the ASSEMBLED text must fit it -- approximations (flat
        # per-line slack) let 139 group headers outgrow the slack at mega
        # depth and overflowed a live run (block 4499 vs cap 4096).
        used = (
            counter.count_text(_HEADER_V4)
            + 2
            + counter.count_text(inventory)
            + 2
        )
        header_cost: dict[str, int] = {}
        chosen_topics: set[str] = set()
        while any(waves):
            progressed = False
            for wave in waves:
                if not wave:
                    continue
                ln = wave.pop(0)
                t = ln.topic.casefold()
                cost = ln.tokens(counter) + 3  # '- ' prefix + newline
                if t not in chosen_topics:
                    if t not in header_cost:
                        header_cost[t] = counter.count_text(
                            f"## {by_topic[t][0].topic} — {state[t][1]}"
                        ) + 3  # newline + blank-line join
                    cost += header_cost[t]
                if used + cost > BACK_TOKENS:
                    wave.clear()
                    continue
                chosen.append(ln)
                chosen_topics.add(t)
                used += cost
                progressed = True
            if not progressed:
                break
        chosen_ids = {id(ln) for ln in chosen}
        groups: list[str] = [inventory]
        for t in recency:
            lines = [ln for ln in by_topic[t] if id(ln) in chosen_ids]
            if not lines:
                continue
            name = by_topic[t][0].topic
            groups.append(
                f"## {name} — {state[t][1]}\n"
                + "\n".join(f"- {ln.text}" for ln in lines)
            )
        return f"{_HEADER_V4}\n\n" + "\n\n".join(groups)

    # -- ledger5 hooks (no-ops for every other mode) --------------------------

    def _note_search(self, query: str, results) -> None:
        """Search hook: record the search log and apply re-activation uses.

        Quoted-only semantics (the dial default): a search re-activates a
        topic only when the query NAMES it -- an intentional search for
        the topic, not an incidental FTS fan hit on shared words. """
        if self.mode != "ledger5":
            return
        tooth = self._teeth + 1
        matched = self._v5_match_topics(query)
        self._rec5["searches"].append({
            "tooth": tooth, "query": query,
            "results": [mid for mid, _ in results],
            "matched": sorted(matched),
        })
        mode = self._dials5.search_reactivates
        if mode == "off":
            return
        keys = set(matched)
        if mode == "any-hit":
            result_ids = {mid for mid, _ in results}
            for ln in self._lines5:
                if set(ln.ids) & result_ids:
                    keys.add(f"{ln.umbrella}/{ln.topic}".casefold())
        for key in keys:
            self._uses5.setdefault(key, []).append(tooth)
            self._rec5["uses"].append({"tooth": tooth, "key": key, "source": "search"})

    def _v5_match_topics(self, query: str) -> set[str]:
        qtok = {w for w in re.split(r"\W+", query.lower()) if len(w) > 2}
        if not qtok:
            return set()
        out: set[str] = set()
        for ln in self._lines5:
            toks = {
                w
                for w in re.split(r"\W+", f"{ln.umbrella} {ln.topic}".lower())
                if len(w) > 2
            }
            if toks & qtok:
                out.add(f"{ln.umbrella}/{ln.topic}".casefold())
        return out

    def _v5_extract_chunk(self, engine: Engine, chunk: str, audited: bool = False) -> None:
        chunk_ids = set(re.findall(r"§([A-Za-z0-9_-]+) \[", chunk))
        tooth = self._teeth + 1  # the tooth being born (post-cut number)
        known = self._known_topics_v5()
        text = self._complete(
            engine, _EXTRACT_PROMPT_V5.format(segment=chunk, known_topics=known)
        )
        added, empty, covered = 0, 0, False
        if not text:
            empty = 1
        for line in (text or "").splitlines():
            row = _bullet_or_row(line)
            if row is None:
                continue
            new = _Ledger5Line(len(self._lines5), row)
            if not new.umbrella or not new.topic:
                continue
            # coverage accrues BEFORE dedupe: an extraction that only
            # restates known facts still SAW the chunk -- marking it
            # uncovered would audit (and re-extract) every quiet chunk
            if set(new.ids) & chunk_ids:
                covered = True
            # near-duplicate suppression (ledger4 rules): same topic, ~same
            # content -> skip; a status transition is never suppressed.
            words = new.wordset()
            if words and any(
                len(words & old.wordset()) / max(1, len(words)) >= 0.8
                and f"{old.umbrella}/{old.topic}".casefold()
                == f"{new.umbrella}/{new.topic}".casefold()
                and (
                    not new.is_state_change
                    or (old.is_state_change and old.kind == new.kind)
                )
                for old in self._lines5[-60:]
            ):
                continue
            new.umbrella = self._canon_ws5(new.umbrella)
            new.topic = self._canon_topic5(new.umbrella, new.topic)
            new.tooth = tooth
            self._lines5.append(new)
            key = f"{new.umbrella}/{new.topic}".casefold()
            self._uses5.setdefault(key, []).append(tooth)
            self._rec5["lines"].append({
                "seq": new.seq, "ws": new.umbrella, "topic": new.topic,
                "kind": new.kind, "text": new.text, "ids": list(new.ids),
                "tooth": tooth, "audited": audited,
            })
            self._rec5["uses"].append({"tooth": tooth, "key": key, "source": "segment"})
            added += 1
        _heartbeat("ledger5", self._teeth, added, empty)
        entry = {
            "tooth": tooth, "ids": sorted(chunk_ids),
            "covered": covered or not chunk_ids, "audited": audited,
        }
        self._chunks5.append(entry)
        self._rec5["chunks"].append(entry)

    def _v5_audit(self, engine: Engine) -> None:
        """Append-only re-encoding audit (theory §3.7): archive chunks with
        zero ledger coverage for REAUDIT_GAP teeth get ONE re-extraction,
        cached forever after. The chunk text is rebuilt from the store by
        id -- a pure function of the segment -- so sweeps replay the same
        appends without provider calls. The GAP itself, however, is a
        collection-time dial: the recorded appends already carry their
        audit-time birth teeth and prompt state, so a different gap
        cannot be counterfactually re-run offline (rerender rejects it)."""
        gap = self._dials5.reaudit_gap
        current = self._teeth + 1
        for ch in self._chunks5:
            if ch["covered"] or ch["audited"]:
                continue
            if current - ch["tooth"] < gap:
                continue
            ch["audited"] = True  # set first: audit once, whatever happens
            wanted = set(ch["ids"])
            parts = [
                f"§{m.id} [{m.role.value}]\n{m.content}\n\n"
                for m in engine.store.messages()
                if m.id in wanted
            ]
            if parts:
                self._v5_extract_chunk(engine, "".join(parts), audited=True)

    def _rec5_tooth(self, engine: Engine, span) -> None:
        self._archived_tokens5 += sum(
            engine.counter.count_message(m) for m in span
        )
        tooth = self._teeth + 1
        self._rec5["teeth"].append({
            "tooth": tooth,
            "ts": datetime.now(UTC).isoformat(),
            "archived_tokens": self._archived_tokens5,
            "cut_messages": len(span),
            "chunks": sum(1 for c in self._chunks5 if c["tooth"] == tooth),
        })

    def _canon_ws5(self, name: str) -> str:
        key = name.casefold()
        if key not in self._ws_canon5:
            self._ws_canon5[key] = name
        return self._ws_canon5[key]

    def _canon_topic5(self, ws: str, topic: str) -> str:
        key = f"{ws}/{topic}".casefold()
        if key not in self._topic_canon5:
            self._topic_canon5[key] = topic
        return self._topic_canon5[key]

    def _known_topics_v5(self, ws_cap: int = 30, topic_cap: int = 100) -> str:
        if not self._lines5:
            return ""
        last: dict[str, int] = {}
        disp: dict[str, str] = {}
        ws_last: dict[str, int] = {}
        ws_disp: dict[str, str] = {}
        for ln in self._lines5:
            tkey = f"{ln.umbrella}/{ln.topic}".casefold()
            last[tkey] = max(last.get(tkey, 0), ln.seq)
            disp.setdefault(tkey, f"{ln.umbrella}/{ln.topic}")
            wkey = ln.umbrella.casefold()
            ws_last[wkey] = max(ws_last.get(wkey, 0), ln.seq)
            ws_disp.setdefault(wkey, ln.umbrella)
        ws = ", ".join(
            ws_disp[k] for k in sorted(ws_last, key=lambda k: -ws_last[k])[:ws_cap]
        )
        topics = ", ".join(
            disp[k] for k in sorted(last, key=lambda k: -last[k])[:topic_cap]
        )
        return f"WORKSTREAMS: {ws}\nKNOWN TOPICS (workstream/topic): {topics}\n"

    def _back_block5(self, engine: Engine) -> str:
        cap = min(self._dials5.resident_budget, BACK_TOKENS)
        block, rungs = render_ledger5(
            self._lines5, self._uses5, self._teeth + 1, self._dials5,
            engine.counter, cap,
        )
        last = self._rec5["renders"][-1] if self._rec5["renders"] else None
        if not last or last["block"] != block or last["render_tooth"] != self._teeth + 1:
            self._rec5["renders"].append({
                "completed_teeth": self._teeth,
                "render_tooth": self._teeth + 1,
                "uses_upto": len(self._rec5["uses"]),  # exact-use snapshot
                "block": block,
            })
            self._rec5["rungs"].extend(rungs)
        return block

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
            self._note_search(str(args.get("query", "")), results)
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

    def recording(self) -> dict | None:
        """Collection-time recording for offline dial sweeps (ledger5 only).

        The recording is the sweep substrate: re-rendering it under new
        dials must reproduce what a live run at those dials would have
        shown, up to the recorded exogenous stream (the counterfactual
        caveat in the theory doc). Everything the render touches is in
        here: lines with birth teeth, source-tagged uses, search logs,
        per-tooth token positions, and rendered blocks for verification.
        """
        if self.mode != "ledger5":
            return None
        return self._rec5


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


class SawtoothLedger4(SawtoothBack):
    def __init__(self) -> None:
        super().__init__("ledger4", "sawtooth-ledger4")


# -- ledger5: resident memory (docs/resident-memory-theory.md) ----------------
#
# Implementation mapping (theory -> this arm), deviations here per project
# convention (and mirrored in docs/lit-review.md):
# - items are TOPICS; the rung ladder is kept-detail-line count: 0 lines =
#   L2 (name+status; name-only when PIN_STATE_STATUS is off), 1 = L1,
#   2+ = L0. No separate gists are generated: the ledger's <=15-word
#   lines are already compressed, so "gist" = the newest line alone.
# - d(class) uses the topic's NEWEST line's kind: a topic currently driven
#   by decisions behaves decision-durable. State-class durability is
#   carried by D_STATE (smallest exponent) and the status pin together.
# - uses: one per extracted line (the segment IS the exogenous stream)
#   plus name-matching searches. "Quoted-only" is operationalized as
#   INTENTIONAL NAME-MATCH: the query names the topic (Matthew-effect
#   guard intact). The literal spec reading -- "only the result the
#   agent actually quotes back" -- is unrecordable here (models don't
#   cite ids in answers) and diverges in both directions; the recorded
#   search log (query, result ids, matched topics) reconstructs the
#   name-match arm exactly, and the off / any-hit arms are synthesized
#   from it. The renderer never feeds the ledger -- the invariant, not
#   a dial.
# - search itself is raw FTS, limit 5: no alias expansion, no
#   ledger-weight/recency re-ranking of results (theory §3.6 fan
#   management is aspirational here; depth-robustness of search makes
#   this bounded).
# - supersession ("replaces §id") and tree-repair ("moved under")
#   channels are NOT implemented: near-duplicate suppression is a
#   birth-time filter, and audits only re-extract UNCOVERED chunks -- a
#   covered-but-misfiled topic is never repaired. Newest-first rendering
#   approximates "newest non-superseded fact"; a stale line can occupy
#   budget that supersession would free.
# - index-overflow triage extends the sealed-collapse rule to ACTIVE
#   workstreams: if the reserved zero-fill index exceeds cap, the
#   lowest-mass active workstream is force-sealed (enumeration is
#   preserved, "overflow never" holds, but a recently-used workstream
#   can be labeled sealed). Composition 'max' = max member activation
#   (the weight-in-one-locus reading of §3.4). pin_next_step pins one
#   fact per UNSEALED workstream (spec says "open"); topics with no
#   state-change line default to status open.
# - sealing/staleness are render-time classifications, recomputed pure
#   from (last use, statuses) every render: no lifecycle state to corrupt,
#   and a re-mentioned workstream unseals itself for free. The sealed
#   tier renders NAME + LAST STATUS (the workstream-level L2 rung):
#   names-only made cued-status probes answer 'sealed' instead of the
#   status (mega-a qual finding).
# - CLOCK=teeth+virtual-idle and the USE_SOURCES diagnostic arms are
#   recorded (timestamps, source-tagged uses) but computed offline in the
#   sweep harness, not live.
# - RESIDENT_BUDGET is clamped to BACK_TOKENS: the trigger prices the
#   back at its cap, so a larger dial would break the overflow invariant.
_EXTRACT_PROMPT_V5 = """You maintain the ledger of an agent session (coding, chat, or mixed), appending observations from one segment.
Every line belongs to a WORKSTREAM (the durable project it serves, a handful per session) and a TOPIC (the specific sub-stream inside it: a gate, a port, a benchmark, a phase).
Output ONLY a bullet list, one per line, in exactly one of three forms:
- WORKSTREAM/topic | note | the observation [§<ids>]
- WORKSTREAM/topic | decision | what was chosen or agreed, and why [§<ids>]
- WORKSTREAM/topic | <status> | what changed and why [§<ids>]
Rules:
- WORKSTREAM comes from the WORKSTREAMS vocabulary where one fits, verbatim. Coin a new one ONLY when the segment starts a genuinely new project. Never coin one per file, task, or gate.
- TOPIC is one short canonical name per sub-stream; reuse KNOWN TOPICS verbatim where they fit.
- note = findings, measurements, next steps. decision = a deliberate choice, commitment, or agreement that future work depends on. The status form (open, resolved, paused, in progress) is reserved for deliberate state changes: a decision landed, work shipped, something blocked or set aside — and says what and why. Never stamp a status on a note or decision.
- At most ~5 lines per topic in this segment; keep what would most change what the agent does next; fold related details into one line.
- EXCEPTION, never skipped and exempt from the budget: a one-line topic/status mention (any speaker, any surface -- chat, tool output, even a trailing "[status] x: paused" annotation) is ALWAYS its own line under a workstream named for that codename -- quoted nicknames ('corvid') and tracker ids (NOVA-11, FIRE-1147) alike. It is NEVER a note under another workstream, even when the rest of the segment is about something else. Several distinct codenames in one segment: one line each, with the state-change form when a status is stated.
- At most 15 words; keep every value that gates future work (ids, numbers, versions, paths). Never quote the transcript.
- A topic reappearing in a later segment is a new line.
{known_topics}Segment:
{segment}"""

_V5_KINDS = ("note", "decision", "open", "resolved", "paused", "in progress")
_V5_STATE_KINDS = ("open", "resolved", "paused", "in progress")

_V5_ROW = re.compile(
    r"^(?P<ws>[^/|]{1,40}?)/(?P<topic>[^|]{1,60}?)\s*\|\s*"
    r"(?P<kind>[^|]{1,20})\s*\|\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_V5_FLAT = re.compile(
    r"^(?P<name>[^|]{1,60}?)\s*\|\s*(?P<kind>[^|]{1,20})\s*\|\s*(?P<rest>.*)$",
    re.IGNORECASE,
)


def _bullet_or_row(line: str) -> str | None:
    """Line text if it is a bullet OR a bare ledger row.

    The model occasionally drops the leading '- ' marker and emits bare
    'ws/topic | kind | text [§ids]' rows (observed live-shaped in the
    extraction bench, e.g. 16-line responses with zero bullets). The
    bullet gate alone silently discards such chunks; accept the bare row
    form too — _Ledger5Line already tolerates markers and off-vocabulary
    kinds.
    """
    s = line.strip()
    m = _BULLET.match(s)
    if m:
        return m.group(1)
    bare = s.lstrip("-*").strip()
    if s and ("|" in bare) and (_V5_ROW.match(bare) or _V5_FLAT.match(bare)):
        return bare
    return None


class _Ledger5Line:
    __slots__ = ("seq", "umbrella", "topic", "kind", "text", "ids", "tooth", "tokens_cache")

    def __init__(self, seq: int, line: str) -> None:
        self.seq = seq
        self.tooth = 0
        self.ids: list[str] = []
        self.tokens_cache = None
        raw = line.strip().lstrip("-*").strip()
        m = _V5_ROW.match(raw)
        if m:
            ws, topic = m.group("ws"), m.group("topic")
            kind, rest = m.group("kind"), m.group("rest")
        else:
            f = _V5_FLAT.match(raw)
            if not f:  # malformed: no parseable row at all
                self.umbrella = self.topic = self.kind = ""
                self.text = ""
                return
            ws = topic = f.group("name")  # no slash: workstream is its own topic
            kind, rest = f.group("kind"), f.group("rest")
        self.umbrella = ws.strip().strip("*_").strip()
        self.topic = topic.strip().strip("*_").strip()
        self.kind = kind.strip().lower()
        if self.kind not in _V5_KINDS:
            # off-vocabulary status word ('shipped', 'done', ...): keep the
            # observation as a note rather than dropping the whole line
            rest = f"{self.kind}: {rest}"
            self.kind = "note"
        cited = _IDS_V4.search(rest)
        if cited:
            self.ids = [w.strip() for w in cited.group(0)[1:].split(",")]
        self.text = _IDS_V4.sub("", rest)
        self.text = re.sub(r"\[[\s,]*\]", "", self.text)  # emptied id groups
        self.text = re.sub(r"\s{2,}", " ", self.text).strip(" ,;")

    @classmethod
    def from_record(cls, d: dict) -> _Ledger5Line:
        """Rebuild a line from its recording dict -- the offline sweep path."""
        ln = cls(d["seq"], f"{d['ws']}/{d['topic']} | {d['kind']} | {d['text']}")
        ln.tooth = d.get("tooth", 0)
        ln.ids = list(d.get("ids", []))
        return ln

    @property
    def is_state_change(self) -> bool:
        return self.kind in _V5_STATE_KINDS

    def wordset(self) -> set[str]:
        return {w for w in re.split(r"\W+", self.text.lower()) if len(w) > 2}

    def tokens(self, counter) -> int:
        if self.tokens_cache is None:
            self.tokens_cache = counter.count_text(self.text)
        return self.tokens_cache


@dataclass(frozen=True)
class Ledger5Dials:
    """The theory's dial table (docs/resident-memory-theory.md §5).

    Defaults are stated priors, not fitted values; the sweep harness
    fits them offline from recorded ledgers. Diagnostic values
    (activation_form='exp', search_reactivates='off'/'any-hit') exist to
    demonstrate failure modes, never as tuning directions.

    Sweepability (rerender_from_recording enforces this): the eleven
    render-time dials (d_note, d_decision, d_state, activation_form,
    recency_floor_full, pin_state_status, pin_next_step,
    resident_budget, composition_detail, seal_after_teeth,
    stale_status_teeth) sweep offline. search_reactivates arms are
    rebuilt offline from the recorded search log. reaudit_gap, clock,
    and use_sources are COLLECTION-TIME: the recording bakes them in
    (audits already ran under the recorded gap; aging is tooth-integer;
    use streams are segment/search), so overriding them offline raises
    rather than silently rendering an unchanged block.
    """

    d_note: float = 0.50
    d_decision: float = 0.35
    d_state: float = 0.20
    activation_form: str = "power"  # power | exp (diagnostic)
    recency_floor_full: int = 8  # teeth; L0 detail needs a use within K
    pin_state_status: bool = True
    pin_next_step: bool = True  # product prior, not theory-derived
    resident_budget: int = 4_096  # clamped to BACK_TOKENS at render
    composition_detail: str = "max"  # max | mass
    seal_after_teeth: int = 30
    stale_status_teeth: int = 15
    reaudit_gap: int = 3
    use_sources: str = "exogenous"  # diagnostic arms computed offline
    search_reactivates: str = "quoted"  # quoted | any-hit | off
    clock: str = "teeth"  # teeth+virtual-idle computed offline


@dataclass
class _T5Topic:
    key: str
    name: str
    facts: list
    status: str
    blocks_seal: bool
    act: float
    last_use: int
    marker_upgrade: int = 0  # tokens: faded marker minus plain line


@dataclass
class _T5Ws:
    key: str
    name: str
    topics: list
    status: str
    last_use: int
    sealed: bool
    mass: float
    top: float


def _l5_d_for(kind: str, dials: Ledger5Dials) -> float:
    if kind in _V5_STATE_KINDS:
        return dials.d_state
    if kind == "decision":
        return dials.d_decision
    return dials.d_note


def _l5_activation(use_teeth: list[int], tooth: int, d: float, form: str) -> float:
    """The law: ln sum over uses of (teeth since use)^(-d), clamped t>=1.

    A fresh single-use item sits at exactly ln 1 = 0; an aged single use
    decays as -d*ln(t) -- logarithmically, never to a floor.
    """
    total = 0.0
    for u in use_teeth:
        t = tooth - u
        if t < 1:
            t = 1
        total += math.exp(-t / 8.0) if form == "exp" else t ** (-d)
    return math.log(total) if total > 0.0 else float("-inf")


_HEADER_V5 = (
    "[Session memory — an append-only index of workstreams. NOT exhaustive: "
    "history_search the verbatim archive before concluding anything never "
    "happened. Topics marked 'faded' have more detail in the archive — "
    "search the topic name to restore it]"
)


def _l5_topic_line(t: _T5Topic, dials: Ledger5Dials, *, has_detail: bool) -> str:
    status = f": {t.status}" if (dials.pin_state_status or has_detail) else ""
    faded = f' · faded — search "{t.name}"' if not has_detail else ""
    return f"{t.name}{status}{faded}"


def _l5_sealed_line(sealed: list, counter, cap: int) -> str:
    """The sealed tier is the workstream-level L2 rung: names PLUS last
    status, detail surrendered. Names-only was tried live (mega-a qual):
    cued-status probes ('where does amphora stand?') got answered with
    the lifecycle word 'sealed' instead of the topic's status -- the
    index must never answer a status question with memory-tier
    vocabulary it can cheaply avoid."""
    names = [f"{w.name} ({w.status})" for w in sealed]  # mass-desc: trim end
    budget = max(40, min(400, cap // 4))
    line = "sealed: " + ", ".join(names)
    while len(names) > 2 and counter.count_text(line) > budget:
        names.pop()
        line = "sealed: " + ", ".join(names) + f" (+{len(sealed) - len(names)} earlier)"
    return line


def _l5_emit(active, sealed, kept, dials, counter, cap) -> str:
    parts = [_HEADER_V5]
    for w in active:
        lines = [f"## {w.name} — {w.status}"]
        for t in w.topics:
            k = sorted(kept.get(t.key, ()), key=lambda f: f.seq, reverse=True)
            lines.append(_l5_topic_line(t, dials, has_detail=bool(k)))
            lines.extend(f"- {f.text}" for f in k)
        parts.append("\n".join(lines))
    if sealed:
        parts.append(_l5_sealed_line(sealed, counter, cap))
    return "\n\n".join(parts)


def _l5_index(lines, uses, tooth, dials):
    """Group facts into topics and workstreams; compute the law.

    Lines born at or after `tooth` are excluded, and the boundary is
    deliberately one tooth tighter than the uses filter: extraction
    stamps a line with the POST-increment tooth (T) while renders happen
    at _teeth + 1 (T+1), so a line born at tooth R first appears in the
    render at R+1 -- while question-time searches legitimately stamp
    uses AT the render tooth. (The render is a view of the ledger AS IT
    WAS at that tooth; the live arm always renders at the newest tooth,
    where the filter is a no-op.)"""
    groups: dict[str, list] = {}
    names: dict[str, tuple[str, str]] = {}
    for ln in lines:
        if not ln.umbrella or not ln.topic:
            continue
        if ln.tooth >= tooth:
            continue  # not born yet -- no clairvoyance in sweeps
        key = f"{ln.umbrella}/{ln.topic}".casefold()
        groups.setdefault(key, []).append(ln)
        names.setdefault(key, (ln.umbrella, ln.topic))
    topics: dict[str, _T5Topic] = {}
    for key, facts in groups.items():
        facts.sort(key=lambda f: f.seq)
        status = next(
            (f.kind for f in reversed(facts) if f.is_state_change), "open"
        )
        ulist = [u for u in (uses.get(key) or []) if u <= tooth] or [
            f.tooth for f in facts if f.tooth
        ]
        if not ulist:
            ulist = [1]
        d = _l5_d_for(facts[-1].kind, dials)
        act = _l5_activation(ulist, tooth, d, dials.activation_form)
        last_use = max(ulist)
        stale = (
            status in ("open", "in progress")
            and (tooth - last_use) > dials.stale_status_teeth
        )
        topics[key] = _T5Topic(
            key=key, name=names[key][1], facts=facts,
            status="stalled" if stale else status,
            blocks_seal=(status in ("open", "in progress") and not stale),
            act=act, last_use=last_use,
        )
    ws_map: dict[str, list] = {}
    for key, t in topics.items():
        ws_map.setdefault(names[key][0].casefold(), []).append(t)
    wss = []
    for ws_key, ts in ws_map.items():
        st_fact = max(
            (f for t in ts for f in t.facts if f.is_state_change),
            key=lambda f: f.seq, default=None,
        )
        status = st_fact.kind if st_fact else "open"
        last_use = max(t.last_use for t in ts)
        stale = (
            status in ("open", "in progress")
            and (tooth - last_use) > dials.stale_status_teeth
        )
        sealed = (tooth - last_use) > dials.seal_after_teeth and not any(
            t.blocks_seal for t in ts
        )
        wss.append(_T5Ws(
            key=ws_key, name=names[ts[0].key][0], topics=ts,
            status="stalled" if stale else status, last_use=last_use,
            sealed=sealed, mass=sum(t.act for t in ts),
            top=max(t.act for t in ts),
        ))
    return topics, wss


def render_ledger5(lines, uses, tooth, dials, counter, cap):
    """The view is a rendering, not a document: a pure function of
    (ledger, uses, tooth, budget) -- the exact function both the live arm
    and the offline dial sweeps call.

    View-assembly policy (theory §3.4, invariant): index reservation first
    (every unsealed workstream header + every topic name+status line),
    then topic-balanced detail fill (rounds across workstreams,
    activation-ordered within, newest facts first), exact accounting
    throughout -- the block can never exceed cap.
    Returns (block_text, rung_rows).
    """
    topics, wss = _l5_index(lines, uses, tooth, dials)
    if not topics:
        return "", []
    comp = (lambda w: w.mass) if dials.composition_detail == "mass" else (lambda w: w.top)
    active = [w for w in wss if not w.sealed]
    sealed = [w for w in wss if w.sealed]
    active.sort(key=lambda w: (comp(w), w.last_use, w.name), reverse=True)
    sealed.sort(key=lambda w: (w.mass, w.name), reverse=True)
    for w in wss:
        w.topics.sort(key=lambda t: (t.act, t.last_use, t.name), reverse=True)
        for t in w.topics:
            # what the topic line costs with detail vs faded: the true
            # first-fact delta in BOTH pin_state_status settings (with the
            # pin off, a detailed line gains ': status' -- plain-name
            # accounting undercounted exactly there)
            t.marker_upgrade = counter.count_text(
                _l5_topic_line(t, dials, has_detail=False)
            ) - counter.count_text(
                _l5_topic_line(t, dials, has_detail=True)
            )
    # -- reservation + triage: the zero-fill block must fit
    while True:
        base = _l5_emit(active, sealed, {}, dials, counter, cap)
        if counter.count_text(base) <= cap or len(active) <= 1:
            break
        victim = min(active, key=lambda w: (w.mass, w.name))
        victim.sealed = True  # rung rows must match the emitted block
        active.remove(victim)
        sealed.append(victim)
        sealed.sort(key=lambda w: (w.mass, w.name), reverse=True)
    # -- incremental fill with exact arithmetic
    used = counter.count_text(base)
    kept: dict[str, list] = {}
    keptseq: dict[str, set] = {}
    added: list = []  # global add order, for overflow rollback

    def fact_cost(f) -> int:
        return counter.count_text(f"- {f.text}") + 1

    def delta(t: _T5Topic, f) -> int:
        d = fact_cost(f)
        if not kept.get(t.key):
            d -= t.marker_upgrade  # the faded marker goes away
        return d

    def add(t: _T5Topic, f) -> bool:
        nonlocal used
        d = delta(t, f)
        if used + d > cap:
            return False
        kept.setdefault(t.key, []).append(f)
        keptseq.setdefault(t.key, set()).add(f.seq)
        added.append((t, f))
        used += d
        return True

    if dials.pin_next_step:
        for w in active:
            cands = [
                f for t in w.topics for f in t.facts
                if f.kind in ("note", "decision")
            ]
            if cands:
                f = max(cands, key=lambda f: f.seq)
                t = next(t for t in w.topics if any(x is f for x in t.facts))
                if f.seq not in keptseq.get(t.key, set()):
                    add(t, f)  # a pin that does not fit is dropped, not forced
    while True:
        progressed = False
        for w in active:
            for t in w.topics:
                if kept.get(t.key) and (tooth - t.last_use) > dials.recency_floor_full:
                    continue  # L0 needs a use within the recency floor
                nxt = next(
                    (f for f in reversed(t.facts)
                     if f.seq not in keptseq.get(t.key, set())),
                    None,
                )
                if nxt is None:
                    continue
                if add(t, nxt):
                    progressed = True
        if not progressed:
            break
    block = _l5_emit(active, sealed, kept, dials, counter, cap)
    # rollback safety net: if arithmetic ever drifts from the emitted
    # text, drop the globally-last-added facts until it fits (converges:
    # every drop frees at least one token).
    while counter.count_text(block) > cap and added:
        t, f = added.pop()
        kept[t.key] = [x for x in kept[t.key] if x.seq != f.seq]
        keptseq[t.key].discard(f.seq)
        block = _l5_emit(active, sealed, kept, dials, counter, cap)
    if counter.count_text(block) > cap:
        # degenerate regime: cap below header + minimal index (never at
        # BACK_TOKENS). Elide the whole index rather than exceed the
        # priced back -- the overflow invariant is absolute (R11).
        block = (
            f"{_HEADER_V5}\n\n"
            f"+{len(wss)} workstreams — index elided at this budget; "
            "history_search the archive"
        )
    while counter.count_text(block) > cap:  # absolute last resort
        block = block[: max(0, int(len(block) * 0.9))]
    rungs = [
        {
            "tooth": tooth, "ws": w.name, "topic": t.name, "status": t.status,
            "kept": len(kept.get(t.key, ())), "sealed": w.sealed,
            "act": round(t.act, 3),
        }
        for w in wss
        for t in w.topics
    ]
    return block, rungs


def _l5_visible_uses(events: list, tooth: int) -> list:
    """Uses visible to a render at `tooth`: segment uses must predate the
    tooth (they are born inside compactions, which a render at tooth R
    has not yet seen -- only R-1 of them have happened), while search
    uses may stamp at the render tooth itself (question-time)."""
    return [
        u for u in events
        if (u["tooth"] < tooth if u.get("source") == "segment" else u["tooth"] <= tooth)
    ]


def _l5_arm_uses(rec: dict, dials, recorded) -> list | None:
    """Rebuild the use-event stream for a different search_reactivates
    arm from the recorded search log (queries, result ids, matched
    topics) and line ids. Returns None when the recorded stream already
    matches the dial -- the common case, where per-render uses_upto
    snapshots stay valid."""
    if dials.search_reactivates == recorded.search_reactivates:
        return None
    base = [u for u in rec["uses"] if u["source"] != "search"]
    if dials.search_reactivates == "off":
        return base
    if dials.search_reactivates == "any-hit":
        lines = [_Ledger5Line.from_record(d) for d in rec["lines"]]
        for s in rec["searches"]:
            rids = set(s["results"])
            keys = set(s["matched"])
            # lines existing at the search moment: birth tooth strictly
            # before the search's stamp (a tooth-T search sees <= T-1)
            for ln in lines:
                if ln.tooth < s["tooth"] and set(ln.ids) & rids:
                    keys.add(f"{ln.umbrella}/{ln.topic}".casefold())
            base.extend(
                {"tooth": s["tooth"], "key": k, "source": "search"} for k in keys
            )
        return base
    raise ValueError(
        f"unknown search_reactivates arm: {dials.search_reactivates!r}"
    )


def replay_use_events(rec: dict, tooth: int | None = None,
                      dials: Ledger5Dials | None = None) -> tuple[int, list]:
    """The exact (tooth, use_events) a replay consumes -- the single
    source of truth shared by rerender_from_recording and analysis code
    (rung age annotation), so derived ages always match the emitted
    block. Same validation and arm semantics as rerender."""
    dials = dials or Ledger5Dials(**rec["dials"])
    recorded = Ledger5Dials(**rec["dials"])
    for fld in ("reaudit_gap", "clock", "use_sources"):
        if getattr(dials, fld) != getattr(recorded, fld):
            raise ValueError(
                f"{fld} is a collection-time dial (recorded "
                f"{getattr(recorded, fld)!r}): the recording bakes it in, "
                "it cannot be swept offline"
            )
    arm_events = _l5_arm_uses(rec, dials, recorded)
    last_render = rec["renders"][-1] if rec["renders"] else None
    if tooth is None:
        tooth = last_render["render_tooth"] if last_render else 1
        if arm_events is None:
            # pre-snapshot recordings (before uses_upto existed) fall back
            # to all uses -- the old, slightly-looser semantics
            upto = last_render.get("uses_upto", len(rec["uses"])) if last_render else 0
            use_events = rec["uses"][:upto]
        else:
            use_events = _l5_visible_uses(arm_events, tooth)
    else:
        # a render at tooth R happens with _teeth = R-1, so the R-th
        # compaction's segment uses/lines did not exist yet; only
        # question-time SEARCH uses legitimately stamp at the render
        # tooth. Prefer the render's own uses_upto snapshot when one was
        # recorded at this tooth (exact, incl. mid-tooth search state);
        # interpolate source-aware otherwise (and always, for synthesized
        # search arms -- snapshot indices refer to the RECORDED stream).
        if arm_events is None:
            snap = next(
                (r["uses_upto"] for r in reversed(rec["renders"])
                 if r["render_tooth"] == tooth and "uses_upto" in r),
                None,
            )
            use_events = (
                rec["uses"][:snap] if snap is not None
                else _l5_visible_uses(rec["uses"], tooth)
            )
        else:
            use_events = _l5_visible_uses(arm_events, tooth)
    return tooth, use_events


def rerender_from_recording(
    rec: dict, counter, tooth: int | None = None, dials: Ledger5Dials | None = None
):
    """Offline dial-sweep entry point: re-render a recorded ledger.

    `dials` overrides the recorded configuration (pass
    dataclasses.replace(Ledger5Dials(**rec["dials"]), ...) for partial
    changes) -- the recording itself is never mutated, so a sweep loop
    over many dial values shares one recording safely. The eleven
    render-time dials sweep for real; search_reactivates arms
    (off / any-hit / quoted-as-recorded) are rebuilt from the recorded
    search log. reaudit_gap, clock and use_sources are collection-time
    -- the recording bakes them in -- and overriding them raises rather
    than silently rendering an unchanged block that a sweep would then
    misreport as "no effect". Byte-exactness is per LAST render at a
    given tooth (superseded same-tooth renders are not addressable by
    tooth alone); synthesized search arms trade that exactness for the
    source-aware tooth filter (mid-tooth ordering precision is a
    recorded-arm luxury).

    Roundtrip contract: with tooth=None (the default), this reproduces
    the last recorded render byte-for-byte -- uses are clipped to the
    render's uses_upto snapshot, so question-time searches that landed
    AFTER that render cannot leak into it (tested). With an explicit
    tooth, ALL recorded uses at or before that tooth count: that is the
    counterfactual sweep path. Search-arm variants (any-hit, off) are
    derivable from rec['searches'] + line ids.

    The counter MUST be the one the live run used (run_eval always uses
    default_counter()) -- a different estimator prices the fill
    differently and the re-render legitimately diverges (verified the
    hard way on the mega-a recording: CharEstimator admitted ~20% more
    detail lines than the DeepSeek tokenizer did).
    """
    dials = dials or Ledger5Dials(**rec["dials"])
    tooth, use_events = replay_use_events(rec, tooth, dials)
    lines = [_Ledger5Line.from_record(d) for d in rec["lines"]]
    uses: dict[str, list[int]] = {}
    for u in use_events:
        uses.setdefault(u["key"], []).append(u["tooth"])
    cap = min(dials.resident_budget, rec.get("back_tokens_cap", BACK_TOKENS))
    return render_ledger5(lines, uses, tooth, dials, counter, cap)


class SawtoothLedger5(SawtoothBack):
    def __init__(self) -> None:
        super().__init__("ledger5", "sawtooth-ledger5")


