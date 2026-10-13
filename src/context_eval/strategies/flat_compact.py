"""Flat single-shot compaction: the deployed default baseline.

This is what Claude Code /compact and auto-compact, Cline, Codex, and
OpenHands' LLMSummarizingCondenser all do: when the view exceeds a hard
threshold, ONE summarization call folds everything older than a kept tail
into a single running summary; the agent continues from ``[summary] +
tail``. Repeated triggers re-summarize summary+history -- the
summary-of-summary compounding regime that Context Compaction Theory
(arXiv:2608.01326) identifies as the shared failure mode of deployed
agents ("after 2 compactions, all reasoning is gone").

Comparisons this arm anchors:

- flat-compact vs lcm: what the hierarchical summary DAG buys over the
  single-shot default (and what it costs in compaction LLM calls)
- flat-compact vs sliding-window: what any summarization buys over plain
  eviction

For harness comparability the strategy exposes the same archive
search/recall tools as lcm and retrieval (Claude Code itself ships no
search tool with /compact -- noted for framing, not replicated).
"""

from __future__ import annotations

from context_eval.provider import ToolSchema
from context_eval.strategies.base import Engine, EngineTool, Strategy, StrategyStats
from context_eval.strategies.lcm import (
    _FtsIndex,
    render_search_results,
    summary_call_kwargs,
)
from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS
from context_eval.types import PromptMessage, Role, Usage

FLAT_TAU = 0.90  # compact when the assembled view exceeds this share of budget
KEEP_TAIL_FRAC = 0.25  # tail kept verbatim, as a fraction of budget

# The deployed /compact prompt is a sectioned template (Claude Code ships
# Primary Request / Key Concepts / Files / Errors ...); OpenHands' condenser
# is a plain instruction. Dev100 forensics: the prose cap inherited from
# lcm's D1 (150 words) discarded one-off per-event numbers (charity $600 /
# $5,000 / $250 lost to vagueness) -- an EVENTS section co-locates
# name/date/amount so multi-session aggregation survives the fold.
_FLAT_SUMMARY_WORDS = 300
# The 1024-token shared backstop truncated M-scale folds mid-bullet (the
# USER run-on alone could exceed it), silently dropping whole sections
# every fold -- the qualitative review found summaries ending mid-word
# with no FACTS/STATE ever emitted.
_FLAT_BACKSTOP_TOKENS = 2560
_FLAT_PROMPT = f"""Summarize this agent session history (chat or coding) for future
reference as a flat, sectioned summary. Do this in two steps:
STEP 1 - from NEW HISTORY SINCE only, list every dated event and every
specific value (names, numbers, dates, ids).
STEP 2 - merge that list into the PREVIOUS SUMMARY's sections, carrying
prior bullets forward. Never skip step 1: new history MUST enter the
summary even when the PREVIOUS SUMMARY already covers the topic.

USER: at most 5 lines - role, location, sustained traits, CURRENT values
only. Do NOT add new interests or topics here; route each new topic to
FACTS. When a value changes, replace the old one ('previously X' once);
never keep both, never list one entity twice.
EVENTS: one bullet per dated event, as `name - date - amount/outcome`.
Every event with a date, number, or count gets its own bullet; never
merge events, never drop amounts. Resolve relative dates to absolute
YYYY/MM/DD using the segment's [YYYY/MM/DD] stamps.
FACTS: every other specific value (ids, tokens, numbers, names,
versions, file paths, commands and outcomes, errors and resolutions).
STATE: the current task/goal and open threads, one or two lines.

Keep the whole summary under ~{_FLAT_SUMMARY_WORDS} words.
Mark where each fact came from as [<id>].

SEGMENT:
{{segment}}"""


class FlatCompact(Strategy):
    name = "flat-compact"

    def __init__(
        self,
        tau: float = FLAT_TAU,
        keep_tail_frac: float = KEEP_TAIL_FRAC,
    ) -> None:
        self._tau = tau
        self._keep_tail_frac = keep_tail_frac
        self._summary: str = ""
        self._summarized_upto = 0  # store messages folded into the summary
        self._compactions = 0
        self._snapshots: list[tuple[int, str]] = []
        self._empty = 0
        self._overhead = Usage()
        self._index = _FtsIndex()

    # -- view ---------------------------------------------------------------

    def _tail(self, engine: Engine) -> tuple[list, int]:
        """Kept-verbatim tail plus the store index where it starts."""
        budget = max(1, int(engine.budget.max_context_tokens * self._keep_tail_frac))
        messages = engine.store.messages()
        kept = []
        used = 0
        start = len(messages)
        for i in range(len(messages) - 1, self._summarized_upto - 1, -1):
            cost = engine.counter.count_message(messages[i])
            if used + cost > budget:
                break
            used += cost
            start = i
            kept.append(messages[i])
        kept.reverse()
        return kept, start

    def _raw_view_tokens(self, engine: Engine) -> int:
        """Projected context without compaction: summary + everything since
        the last compaction. The trigger fires on THIS, matching how
        deployed agents compact on projected context, not the residual."""
        messages = engine.store.messages()
        summary = engine.counter.count_text(self._summary) if self._summary else 0
        return summary + MESSAGE_OVERHEAD_TOKENS + sum(
            engine.counter.count_message(m)
            for m in messages[self._summarized_upto :]
        )

    def on_turn(self, engine: Engine) -> None:
        messages = engine.store.messages()
        if len(messages) > self._index.indexed_upto:
            self._index.index(messages[self._index.indexed_upto :])
        tail, tail_start = self._tail(engine)
        if self._raw_view_tokens(engine) <= self._tau * engine.budget.max_context_tokens:
            return
        if engine.llm is None or tail_start <= self._summarized_upto:
            return  # nothing compactable yet

        # Fold everything before the tail into the running summary: the
        # segment is prior unsummarized messages, prefixed by the previous
        # summary (summary-of-summary compounding on later triggers).
        segment_parts: list[str] = []
        if self._summary:
            segment_parts.append(f"PREVIOUS SUMMARY:\n{self._summary}")
            segment_parts.append("NEW HISTORY SINCE:")
        for msg in messages[self._summarized_upto : tail_start]:
            segment_parts.append(f"[§{msg.id}] {msg.role.value}: {msg.content}")
        prompt = self._fold_prompt("\n".join(segment_parts))

        completion = engine.llm.complete(
            [{"role": "user", "content": prompt}],
            **{**summary_call_kwargs(), "max_tokens": _FLAT_BACKSTOP_TOKENS},
        )
        if completion.usage:
            self._overhead.add(completion.usage)
        if not completion.text.strip() and completion.finish_reason == "length":
            self._empty += 1
        self._summary = completion.text.strip() or self._summary
        self._summarized_upto = tail_start
        self._compactions += 1
        if self._compactions % 10 == 1 or self._compactions % 10 == 0:
            self._snapshots.append((self._compactions, self._summary))

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        # Deployed semantics: verbatim everything since the last compaction
        # (plus the running summary). The keep-tail cap decides only what
        # gets FOLDED at compaction time -- between compactions the view
        # grows freely until the next trigger, exactly like /compact.
        view: list[PromptMessage] = []
        if self._summary:
            view.append(
                PromptMessage(
                    role=Role.USER,
                    content=f"[Session summary — compacted {self._compactions}x]\n{self._summary}",
                )
            )
        view.append(PromptMessage(role=Role.USER, content="[Recent context — verbatim]"))
        view.extend(
            m.to_prompt() for m in engine.store.messages()[self._summarized_upto :]
        )
        return view

    def _fold_prompt(self, segment: str) -> str:
        """The compaction instruction; v2 variants override this hook."""
        return _FLAT_PROMPT.format(segment=segment)

    def recording(self) -> dict | None:
        """Summary artifact forensics: final summary plus every 10th fold
        (watch cue decay across folds without paying for full history)."""
        if not self._summary:
            return None
        return {
            "final_summary": self._summary,
            "compactions": self._compactions,
            "snapshots": self._snapshots,
        }

    # -- recall tools (parity with lcm/retrieval) ----------------------------

    def tools(self, engine: Engine) -> list[EngineTool]:
        search = ToolSchema(
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
            return render_search_results(results, engine.store)

        def do_recall(args):
            try:
                return engine.store.require(str(args.get("id", ""))).content
            except KeyError:
                return "no such message id"

        return [
            EngineTool(schema=search, execute=do_search),
            EngineTool(schema=recall, execute=do_recall),
        ]

    def stats(self) -> StrategyStats:
        return StrategyStats(
            overhead_usage=self._overhead,
            compactions=self._compactions,
            empty_summaries=self._empty,
        )
