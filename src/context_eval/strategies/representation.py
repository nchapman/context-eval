"""Incremental representation memory: the Honcho-mechanism port.

Honcho (Plastic Labs v3) routes messages to a background deriver that
extracts conclusions and merges them into per-entity representations --
documents updated incrementally, never re-derived from the growing
transcript. This arm ports that MECHANISM at fixed model and fixed
accounting; Honcho's async scheduling, multi-tier dialectic, and dreamer
consolidation are deployment machinery, not the claim under test
(differences are noted in docs/lit-review.md).

The structural property under test: each update reads CURRENT MEMORY +
messages since the last update only -- input size is O(memory + delta),
independent of history length. The fold-over-fold regime where flat's
re-derivation truncates (measured 42-100% empty folds at 4096 max_tokens)
cannot occur here by construction. The inventory of open threads lives
in the memory as maintained state, not as a per-fold output -- the exact
capability the P1' prompting experiment showed fold-based summaries
cannot carry.

Anchors:
- representation vs flat-compact: incremental merge vs summary-of-summary
- representation vs lcm: one maintained doc vs hierarchical consolidation
"""

from __future__ import annotations

from context_eval.provider import ToolSchema
from context_eval.strategies.base import Engine, EngineTool, Strategy, StrategyStats
from context_eval.strategies.flat_compact import FLAT_TAU, KEEP_TAIL_FRAC
from context_eval.strategies.lcm import SUMMARY_MAX_TOKENS, _FtsIndex
from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS
from context_eval.types import PromptMessage, Role, Usage

MEMORY_TARGET_TOKENS = 2048  # resident cap the update prompt enforces

_MEMORY_PROMPT = """You maintain the long-term memory of an agent's coding session.
CURRENT MEMORY is below, then NEW MESSAGES since the last update.
Emit the complete updated memory, merged:
- preserve every still-relevant fact and every specific value (ids,
  tokens, numbers, names, versions, file paths) exactly
- update entities and statuses; open new threads as they appear
- drop only what is obsolete or superseded
Structure as sections: Goal & decisions | Entities & values | Open
threads (`name|status`). State conclusions; never quote transcripts.
Keep the whole memory under {target} tokens. Be terse.

CURRENT MEMORY:
{memory}

NEW MESSAGES:
{delta}"""


class RepresentationMemory(Strategy):
    name = "representation"

    def __init__(
        self,
        tau: float = FLAT_TAU,
        keep_tail_frac: float = KEEP_TAIL_FRAC,
    ) -> None:
        self._tau = tau
        self._keep_tail_frac = keep_tail_frac
        self._memory: str = ""
        self._folded_upto = 0  # store messages merged into the memory
        self._updates = 0
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
        for i in range(len(messages) - 1, self._folded_upto - 1, -1):
            cost = engine.counter.count_message(messages[i])
            if used + cost > budget:
                break
            used += cost
            start = i
            kept.append(messages[i])
        kept.reverse()
        return kept, start

    def _raw_view_tokens(self, engine: Engine) -> int:
        """Projected context without an update: memory + everything since
        the last fold -- same trigger semantics as flat-compact."""
        messages = engine.store.messages()
        memory = engine.counter.count_text(self._memory) if self._memory else 0
        return memory + MESSAGE_OVERHEAD_TOKENS + sum(
            engine.counter.count_message(m)
            for m in messages[self._folded_upto :]
        )

    def on_turn(self, engine: Engine) -> None:
        messages = engine.store.messages()
        if len(messages) > self._index.indexed_upto:
            self._index.index(messages[self._index.indexed_upto :])
        tail, tail_start = self._tail(engine)
        if self._raw_view_tokens(engine) <= self._tau * engine.budget.max_context_tokens:
            return
        if engine.llm is None or tail_start <= self._folded_upto:
            return  # nothing new to fold yet

        # Merge only the new delta into the memory: the prompt never
        # contains history older than the memory itself already covers.
        delta = "\n".join(
            f"[§{m.id}] {m.role.value}: {m.content}"
            for m in messages[self._folded_upto : tail_start]
        )
        prompt = _MEMORY_PROMPT.format(
            target=MEMORY_TARGET_TOKENS,
            memory=self._memory or "(none yet)",
            delta=delta,
        )
        completion = engine.llm.complete(
            [{"role": "user", "content": prompt}], max_tokens=SUMMARY_MAX_TOKENS
        )
        if completion.usage:
            self._overhead.add(completion.usage)
        if not completion.text.strip() and completion.finish_reason == "length":
            self._empty += 1
        # An empty update must not wipe the memory (same guard class as
        # flat's `or self._summary`).
        if completion.text.strip():
            self._memory = completion.text.strip()
        self._folded_upto = tail_start
        self._updates += 1

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        view: list[PromptMessage] = []
        if self._memory:
            view.append(
                PromptMessage(
                    role=Role.USER,
                    content=f"[Agent memory — updated {self._updates}x]\n{self._memory}",
                )
            )
        view.append(PromptMessage(role=Role.USER, content="[Recent context — verbatim]"))
        view.extend(
            m.to_prompt() for m in engine.store.messages()[self._folded_upto :]
        )
        return view

    # -- recall tools (parity with flat/lcm/retrieval) ------------------------

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
        return StrategyStats(
            overhead_usage=self._overhead,
            compactions=self._updates,
            empty_summaries=self._empty,
        )
