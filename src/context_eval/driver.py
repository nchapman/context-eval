"""Harness driver: open-loop replay + question answering under a fixed budget.

Per replayed turn: append to the store, run the strategy's ``on_turn`` hook
(may compact), and measure the assembled view (occupancy + budget check).
No task LLM calls during replay — the recorded trajectory IS the history.

At question time: assemble a fresh view, prepend the system prompt, expose
the strategy's recall tools, and run a bounded tool loop. Every call's usage
is recorded. Overflow at replay time ends the run (the anchor failure mode).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from context_eval.budget import Budget
from context_eval.datasets.base import ReplaySample
from context_eval.judge import Score, score_question
from context_eval.provider import (
    LLMClient,
    ToolSchema,
    assistant_tool_message,
    parse_tool_arguments,
    tool_result_message,
)
from context_eval.questions import Question
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine, Strategy
from context_eval.tokens import CachedCounter, TokenCounter
from context_eval.types import Message, PromptMessage, Role, Usage

SYSTEM_PROMPT = (
    "You are answering questions about a previously recorded software "
    "engineering session (an agent working in a repository, or a recorded "
    "conversation). The messages below are that history, possibly compacted. "
    "If details you need appear missing, use the provided recall tools to "
    "retrieve them from the full archive. Answer concisely; when a question "
    "asks for a specific value, reply with the exact value."
)

MAX_TOOL_STEPS = 6


@dataclass
class TurnRecord:
    index: int
    message_id: str
    view_tokens: int
    overflow: bool
    # Hash of the head of the assembled view: unchanged between turns means
    # the prompt prefix is stable (provider-cache friendly); a change means
    # the strategy rewrote its view (cache bust from that point on).
    view_head_sig: str = ""


@dataclass
class AnswerRecord:
    question: str
    gold: str
    kind: str
    category: str
    prediction: str
    score: float
    score_method: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: int = 0
    model: str = ""
    provider: str = ""
    error: str | None = None  # per-question failure (scored 0, run continues)
    meta: dict = field(default_factory=dict)  # question meta (terms, distractors)


@dataclass
class RunResult:
    strategy: str
    dataset: str
    sample_id: str
    budget_tokens: int
    turns: list[TurnRecord] = field(default_factory=list)
    answers: list[AnswerRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)  # answering calls
    overhead_usage: Usage = field(default_factory=Usage)  # strategy compaction
    compactions: int = 0
    peak_view_tokens: int = 0
    overflowed_at: int | None = None
    model: str = ""
    provider: str = ""
    error: str | None = None
    empty_summaries: int = 0  # compactions that produced no content (see StrategyStats)

    @property
    def mean_view_tokens(self) -> float:
        return sum(t.view_tokens for t in self.turns) / max(1, len(self.turns))

    @property
    def mean_score(self) -> float:
        return sum(a.score for a in self.answers) / max(1, len(self.answers))

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def view_to_wire(view: Sequence[PromptMessage | Message]) -> list[dict[str, Any]]:
    """Convert an assembled view to wire messages.

    Assistant messages carrying tool_calls pair with the TOOL messages that
    follow them (by call order). Unpaired tool observations render as user
    messages with an observation prefix — same convention for every strategy.
    """
    wire: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []

    for m in view:
        tool_calls = m.meta.get("tool_calls") if m.meta else None
        if m.role is Role.ASSISTANT and tool_calls:
            calls = [
                {
                    "id": tc.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", "tool"),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
            pending_call_ids = [c["id"] for c in calls]
            wire.append(
                {"role": "assistant", "content": m.content or None, "tool_calls": calls}
            )
        elif m.role is Role.TOOL and pending_call_ids:
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_call_ids.pop(0),
                    "content": m.content,
                }
            )
        elif m.role is Role.TOOL:
            wire.append(
                {"role": "user", "content": f"Observation (recorded):\n{m.content}"}
            )
        else:
            wire.append({"role": m.role.value, "content": m.content})

    return wire


def _schema_tokens(schemas: Sequence[ToolSchema], counter: TokenCounter) -> int:
    return sum(counter.count_text(json.dumps(s.to_wire())) for s in schemas)


def _build_engine(
    strategy: Strategy,
    store: TranscriptStore,
    counter: TokenCounter,
    budget_tokens: int,
    questions: Sequence[Question],
    llm: LLMClient,
    system_prompt: str,
) -> Engine:
    # Reserve the fixed costs every strategy pays identically (system prompt,
    # tool schemas, the largest question) so the budget strategies see applies
    # to their active view alone.
    probe = Engine(store=store, counter=counter, budget=Budget(1), llm=llm)
    tool_tokens = _schema_tokens([t.schema for t in strategy.tools(probe)], counter)
    question_tokens = max((counter.count_text(q.question) for q in questions), default=0)
    system_tokens = counter.count_text(system_prompt)
    effective = max(1, budget_tokens - tool_tokens - question_tokens - system_tokens)
    return Engine(
        store=store,
        counter=counter,
        budget=Budget(max_context_tokens=effective),
        llm=llm,
    )


def run_sample(
    *,
    strategy: Strategy,
    sample: ReplaySample,
    questions: Sequence[Question],
    llm: LLMClient,
    counter: TokenCounter,
    budget_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
    # Generous on purpose: reasoning draws from the same budget (observed
    # p99 ~256, max ~395 tokens per call before any content). We pay for
    # usage, not the cap, so headroom is nearly free while truncation
    # quietly destroys answers.
    max_answer_tokens: int = 2048,
    temperature: float = 0.0,
) -> RunResult:
    result = RunResult(
        strategy=strategy.name,
        dataset=sample.dataset,
        sample_id=sample.sample_id,
        budget_tokens=budget_tokens,
    )

    store = TranscriptStore()
    counter = CachedCounter(counter)  # per-turn view measurement needs memoization
    engine = _build_engine(
        strategy, store, counter, budget_tokens, questions, llm, system_prompt
    )
    n_msgs = len(sample.messages)
    est = engine.counter.count_text(" ".join(m.content for m in sample.messages))
    print(
        f"[run] {strategy.name} x {sample.sample_id[:24]}: {n_msgs} msgs, "
        f"~{est // 1000}K raw tokens, {len(questions)} questions -- replaying",
        file=sys.stderr,
        flush=True,
    )

    # --- replay: append, compact, measure -----------------------------------
    try:
        for i, msg in enumerate(sample.messages):
            store.append(msg.role, msg.content, meta=dict(msg.meta))
            strategy.on_turn(engine)

            view = strategy.assemble(engine)
            tokens = counter.count_messages(view)
            overflow = tokens > engine.budget.max_context_tokens
            head = "\n".join(m.content for m in view[:3])[:500]
            result.turns.append(
                TurnRecord(
                    index=i,
                    message_id=store.messages()[-1].id,
                    view_tokens=tokens,
                    overflow=overflow,
                    view_head_sig=hashlib.sha1(head.encode()).hexdigest()[:12],
                )
            )
            result.peak_view_tokens = max(result.peak_view_tokens, tokens)
            if overflow and result.overflowed_at is None:
                result.overflowed_at = i
    except Exception as exc:  # strategy/engine failure is part of the data
        result.error = f"replay: {exc}"
        _capture_stats(strategy, result)
        return result

    if result.overflowed_at is not None:
        _capture_stats(strategy, result)  # anchor died: unanswered questions score 0
        return result

    # --- question time -------------------------------------------------------
    for qn, q in enumerate(questions, 1):
        try:
            record = _ask(
                strategy=strategy,
                engine=engine,
                question=q,
                llm=llm,
                max_answer_tokens=max_answer_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )
            print(
                f"[answer] {qn}/{len(questions)} kind={q.kind:11s} "
                f"tools={record.tool_calls} "
                + ("ERROR" if record.error else f"score={record.score:.2f}"),
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            # One bad question (provider failure, tool error) must not
            # discard the remaining questions for this sample.
            record = AnswerRecord(
                question=q.question,
                gold=q.answer,
                kind=q.kind,
                category=q.category,
                prediction="",
                score=0.0,
                score_method="error",
                error=str(exc)[:300],
            )
            print(
                f"[answer] {qn}/{len(questions)} kind={q.kind:11s} "
                f"FAILED: {str(exc)[:120]}",
                file=sys.stderr,
                flush=True,
            )
        result.answers.append(record)
        result.usage.add(record.usage)
        if record.model and not result.model:
            result.model = record.model
            result.provider = record.provider

    _capture_stats(strategy, result)
    return result


def _strip_tool_artifacts(wire: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild an answer wire without tool-call/tool-result messages.

    Keeps system, view, and question messages (assistant texts stay — they
    carry narration). Used for the last-resort re-ask when a tool-saturated
    wire keeps returning empty-stop completions.
    """
    kept = []
    for m in wire:
        if m.get("role") == "tool":
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            continue
        kept.append(m)
    return [
        *kept,
        {
            "role": "user",
            "content": (
                "Based on the session history above, answer the question "
                "above in one or two sentences. If you truly cannot, say "
                "what you do know that is relevant."
            ),
        },
    ]


def _ask(
    *,
    strategy: Strategy,
    engine: Engine,
    question: Question,
    llm: LLMClient,
    max_answer_tokens: int,
    temperature: float,
    system_prompt: str,
) -> AnswerRecord:
    view = strategy.assemble(engine)
    engine_tools = strategy.tools(engine)
    schemas = [t.schema for t in engine_tools]
    by_name = {t.schema.name: t for t in engine_tools}

    wire: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *view_to_wire(view),
        {"role": "user", "content": question.question},
    ]

    usage = Usage()
    tool_call_count = 0
    prediction = ""
    model = provider = ""
    best_text = ""  # last non-empty narration alongside tool calls

    for _step in range(MAX_TOOL_STEPS + 1):
        # On the final forced answer the tools are removed: a model still
        # trying to search emits tool_calls with empty text and we record
        # a blank. Without tools it must put the answer in content.
        final_step = _step == MAX_TOOL_STEPS
        completion = llm.complete(
            wire,
            tools=None if final_step else (schemas or None),
            max_tokens=max_answer_tokens,
            temperature=temperature,
            # answers can blank the same way summaries did (reasoning eats
            # the cap at length); escalate rather than record an empty
            # prediction — symmetric across every arm
            require_content=True,
        )
        usage.add(completion.usage)
        model, provider = completion.model or model, completion.provider or provider

        if completion.tool_calls and _step < MAX_TOOL_STEPS:
            tool_call_count += len(completion.tool_calls)
            if completion.text.strip():
                best_text = completion.text.strip()
            wire.append(assistant_tool_message(completion.tool_calls, completion.text))
            for tc in completion.tool_calls:
                engine_tool = by_name.get(tc.name)
                if engine_tool is None:
                    output = f"error: unknown tool {tc.name}"
                else:
                    output = engine_tool.execute(parse_tool_arguments(tc.arguments))
                wire.append(tool_result_message(tc.id, str(output)))
            continue

        # Reasoning-only responses: the model put the whole answer in the
        # reasoning channel and left content empty. Fall back to the last
        # non-empty reasoning line, where the concluding answer typically
        # lives; without this ~30% of short answers scored as blank.
        text = completion.text.strip()
        if not text and completion.reasoning.strip():
            lines = [ln.strip() for ln in completion.reasoning.splitlines() if ln.strip()]
            text = lines[-1] if lines else ""
        if not text:
            # A tool-saturated wire with no tools left yields an empty
            # stop completion (measured: finish=stop, 0 text, 0 reasoning).
            # Prefer the model's own narration from the tool rounds; as a
            # last resort force one no-tools nudge call.
            text = best_text
        if not text:
            nudged = llm.complete(
                [*wire, {"role": "user", "content":
                 "Answer now in one or two sentences using what you found."}],
                max_tokens=max_answer_tokens,
                temperature=temperature,
                require_content=True,
            )
            usage.add(nudged.usage)
            text = nudged.text.strip() or (
                nudged.reasoning.strip().splitlines()[-1]
                if nudged.reasoning.strip() else ""
            )
        if not text:
            # A tool-saturated wire can keep returning empty-stop even when
            # nudged (observed deterministically on one pi session across
            # all arms). Re-ask on a clean wire: tool artifacts stripped.
            reask = llm.complete(
                _strip_tool_artifacts(wire),
                max_tokens=max_answer_tokens,
                temperature=temperature,
                require_content=True,
            )
            usage.add(reask.usage)
            text = reask.text.strip() or (
                reask.reasoning.strip().splitlines()[-1]
                if reask.reasoning.strip() else ""
            )
        if not text and os.environ.get("CONTEXT_EVAL_DEBUG_ANSWERS"):
            print(
                f"[answer-debug] q={question.question[:60]!r} "
                f"finish={completion.finish_reason} "
                f"text_len={len(completion.text)} "
                f"reason_len={len(completion.reasoning)} "
                f"tools_in_wire={sum(1 for w in wire if w.get('role') == 'tool')}",
                file=sys.stderr,
            )
        prediction = text
        break

    score: Score = score_question(question, prediction)
    return AnswerRecord(
        question=question.question,
        gold=question.answer,
        kind=question.kind,
        category=question.category,
        prediction=prediction,
        score=score.value,
        score_method=score.method,
        usage=usage,
        tool_calls=tool_call_count,
        model=model,
        provider=provider,
        meta=dict(question.meta),
    )


def _capture_stats(strategy: Strategy, result: RunResult) -> None:
    stats = strategy.stats()
    result.overhead_usage = stats.overhead_usage
    result.compactions = stats.compactions
    result.empty_summaries = stats.empty_summaries
