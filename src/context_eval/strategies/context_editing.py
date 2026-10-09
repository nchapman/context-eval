"""Deterministic context editing: clear old tool-result payloads.

Anthropic's ``clear_tool_uses`` context-editing strategy (Claude Agent SDK
/ server-side context management): keep the conversation skeleton --
assistant turns, user turns, tool calls -- and clear the payloads of old
tool results once the view exceeds a threshold. Cheap, deterministic, no
LLM anywhere. Widely deployed as the "safest lightest-touch" default.

Comparisons this arm anchors:

- editing vs sliding-window: value of keeping the cleared skeleton
  (structure, pairing integrity, head of each result) over dropping turns
- editing vs arc: value of LLM-generated previews + citations over plain
  head-stubs
"""

from __future__ import annotations

from context_eval.strategies.base import Engine, Strategy
from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS
from context_eval.types import PromptMessage, Role

STUB_CHARS = 120  # head of a cleared result kept as a stub
EDIT_TAU = 0.90  # start clearing when the verbatim view exceeds this


class ContextEditing(Strategy):
    name = "context-editing"

    def assemble(self, engine: Engine) -> list[PromptMessage]:
        budget = engine.budget.max_context_tokens
        messages = engine.store.messages()

        # Verbatim cost of everything; if under budget, nothing to edit.
        total = sum(
            engine.counter.count_text(m.content) + MESSAGE_OVERHEAD_TOKENS
            for m in messages
        )
        if total <= budget:
            return [m.to_prompt() for m in messages]

        # Over budget: clear tool-result payloads oldest-first, keeping a
        # stub, until the edited view fits. Everything else stays verbatim.
        cleared: dict[str, str] = {}
        edited_total = total
        for msg in messages:
            if edited_total <= budget:
                break
            if msg.role is not Role.TOOL:
                continue
            full = engine.counter.count_text(msg.content)
            stub = f"[cleared] {msg.content[:STUB_CHARS]}…"
            cleared[msg.id] = stub
            edited_total -= full - engine.counter.count_text(stub)

        return [
            PromptMessage(
                role=msg.role, content=cleared.get(msg.id, msg.content)
            )
            for msg in messages
        ]
