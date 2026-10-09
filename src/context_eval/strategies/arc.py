"""#3 — ARC: Addressable Recall Compaction (clean-room, arXiv:2607.25066).

Deterministic, strictly lossless. Every tool observation lives verbatim in
the ID-addressable store (already our TranscriptStore). The active view
compacts older/longer observations into ``§id`` citations (head/tail preview
+ metadata); observations below the length threshold rho stay verbatim. The
``_recall`` tool restores full content on demand; citation handles persist
across later compactions. No LLM participates in compaction.

Divergences from the paper (documented for the writeup): we rank compaction
candidates by size (largest first, oldest tiebreak) to minimize citation
count, and recompute the compacted set at every assemble (stateless).
"""

from __future__ import annotations

from context_eval.provider import ToolSchema
from context_eval.strategies.base import Engine, EngineTool, Strategy, StrategyStats
from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS
from context_eval.types import PromptMessage, Role

HEAD_CHARS = 160
TAIL_CHARS = 120
# Paper's rho: observations at or below this token count stay verbatim.
RHO_TOKENS = 64

_RECALL_NAME = "_recall"
_RECALL_DESCRIPTION = (
    "Retrieve the full original content of a compacted tool observation. "
    "Pass the id exactly as shown in a § citation (e.g. m000014)."
)


def _preview(text: str) -> str:
    return " ".join(text.split())


def make_citation(message_id: str, content: str, tokens: int) -> str:
    flat = _preview(content)
    head = flat[:HEAD_CHARS]
    tail = flat[-TAIL_CHARS:] if len(flat) > HEAD_CHARS + TAIL_CHARS else ""
    parts = [
        f"[§{message_id} tool output — compacted, {tokens} tokens total]",
        f"head: {head}",
    ]
    if tail:
        parts.append(f"tail: {tail}")
    parts.append(f'full content: use the {_RECALL_NAME} tool with id "{message_id}"')
    return "\n".join(parts)


class ARCCitations(Strategy):
    name = "arc"

    def __init__(self, rho_tokens: int = RHO_TOKENS) -> None:
        self.rho_tokens = rho_tokens
        self._cited_now: set[str] = set()
        self._recall_calls = 0

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        messages = engine.store.messages()
        budget = engine.budget.max_context_tokens
        counter = engine.counter

        # Start from the verbatim view and its true size.
        entries: list[tuple[int, PromptMessage]] = []
        total = 0
        for idx, m in enumerate(messages):
            cost = counter.count_text(m.content) + MESSAGE_OVERHEAD_TOKENS
            entries.append((idx, m.to_prompt()))
            total += cost

        # Compaction candidates: tool observations above rho, not already
        # tiny. Rank by size (largest first), oldest as tiebreak.
        candidates = sorted(
            (
                (counter.count_text(m.content) + MESSAGE_OVERHEAD_TOKENS, -i, m)
                for i, m in enumerate(messages)
                if m.role is Role.TOOL
                and counter.count_text(m.content) > self.rho_tokens
            ),
            reverse=True,
        )
        cited: dict[int, str] = {}
        for cost, neg_idx, m in candidates:
            if total <= budget:
                break
            i = -neg_idx
            obs_tokens = counter.count_text(m.content)
            citation = make_citation(m.id, m.content, obs_tokens)
            cite_tokens = counter.count_text(citation) + MESSAGE_OVERHEAD_TOKENS
            if cite_tokens >= cost:
                continue  # citing would grow the view; leave verbatim
            cited[i] = citation
            total -= cost - cite_tokens

        self._cited_now = {messages[i].id for i in cited}
        view: list[PromptMessage] = []
        for i, (_idx, prompt) in enumerate(entries):
            if i in cited:
                view.append(PromptMessage(role=prompt.role, content=cited[i]))
            else:
                view.append(prompt)
        return view

    def tools(self, engine: Engine) -> list[EngineTool]:
        schema = ToolSchema(
            name=_RECALL_NAME,
            description=_RECALL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The § id from a citation, e.g. m000014",
                    }
                },
                "required": ["id"],
            },
        )

        def execute(args):
            self._recall_calls += 1
            needle_id = str(args.get("id", "")).strip().lstrip("§")
            msg = engine.store.get(needle_id)
            if msg is None:
                return f"error: unknown observation id {needle_id!r}"
            return f"[§{needle_id} full content]\n{msg.content}"

        return [EngineTool(schema=schema, execute=execute)]

    def stats(self) -> StrategyStats:
        stats = StrategyStats()
        stats.compactions = len(self._cited_now)
        stats.recall_tool_calls = self._recall_calls
        return stats
