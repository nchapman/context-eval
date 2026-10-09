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
from context_eval.strategies.lcm import _D1_PROMPT, SUMMARY_MAX_TOKENS, _FtsIndex
from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS
from context_eval.types import PromptMessage, Role, Usage

FLAT_TAU = 0.90  # compact when the assembled view exceeds this share of budget
KEEP_TAIL_FRAC = 0.25  # tail kept verbatim, as a fraction of budget


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
            cost = (
                engine.counter.count_text(messages[i].content) + MESSAGE_OVERHEAD_TOKENS
            )
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
            engine.counter.count_text(m.content) + MESSAGE_OVERHEAD_TOKENS
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
        prompt = _D1_PROMPT.format(segment="\n".join(segment_parts))

        completion = engine.llm.complete(
            [{"role": "user", "content": prompt}], max_tokens=SUMMARY_MAX_TOKENS
        )
        if completion.usage:
            self._overhead.add(completion.usage)
        self._summary = completion.text.strip() or self._summary
        self._summarized_upto = tail_start
        self._compactions += 1

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
            return "\n".join(f"{mid}: {snippet}" for mid, snippet in results)

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
        return StrategyStats(overhead_usage=self._overhead, compactions=self._compactions)
