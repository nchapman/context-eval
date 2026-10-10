from collections.abc import Mapping
from typing import Any

from context_eval.datasets.base import ReplayMessage, ReplaySample
from context_eval.driver import MAX_TOOL_STEPS, run_sample, view_to_wire
from context_eval.questions import Question, inject_needles
from context_eval.strategies import build
from context_eval.tokens import CharEstimator
from context_eval.types import PromptMessage, Role, ToolCall, Usage


class FakeLLM:
    """Scripted completions; records every call for assertions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, *, tools=None, max_tokens=1024,
                 temperature=0.0, reasoning=None, require_content=False):
        from context_eval.provider import Completion

        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": list(tools or [])}
        )
        if callable(self.script[0]):
            self.script[0](self)  # dynamic behavior (e.g. raise on Nth call)
            raise AssertionError("dynamic script step must raise or rewrite")
        item = self.script.pop(0)
        if isinstance(item, str):
            return Completion(
                text=item,
                tool_calls=(),
                usage=Usage(prompt_tokens=10, completion_tokens=3, llm_calls=1),
                finish_reason="stop",
                model="fake-model",
                provider="fake-provider",
            )
        return Completion(
            text="",
            tool_calls=(
                ToolCall(id=item["id"], name=item["name"], arguments=item["arguments"]),
            ),
            usage=Usage(prompt_tokens=10, completion_tokens=3, llm_calls=1),
            finish_reason="tool_calls",
            model="fake-model",
            provider="fake-provider",
        )


def tiny_sample():
    messages = [ReplayMessage(role=Role.USER, content="Fix the failing test in foo.py")]
    for i in range(4):
        messages.append(
            ReplayMessage(
                role=Role.ASSISTANT,
                content="",
                meta={
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ]
                },
            )
        )
        messages.append(ReplayMessage(role=Role.TOOL, content=f"test output {i} " + "y" * 40))
    return ReplaySample(dataset="t/1/openhands", sample_id="tiny-1", messages=tuple(messages))


class TestViewToWire:
    def test_pairs_tool_calls_with_observations(self):
        view = [
            PromptMessage(
                role=Role.ASSISTANT,
                content="",
                meta={
                    "tool_calls": [
                        {"id": "c0", "function": {"name": "bash", "arguments": "{}"}}
                    ]
                },
            ),
            PromptMessage(role=Role.TOOL, content="obs"),
        ]
        wire = view_to_wire(view)

        assert wire[0]["tool_calls"][0]["id"] == "c0"
        assert wire[1] == {"role": "tool", "tool_call_id": "c0", "content": "obs"}

    def test_unpaired_observation_renders_as_user(self):
        wire = view_to_wire([PromptMessage(role=Role.TOOL, content="orphan")])

        assert wire[0]["role"] == "user"
        assert "orphan" in wire[0]["content"]

    def test_plain_messages_pass_through(self):
        wire = view_to_wire([PromptMessage(role=Role.USER, content="hi")])
        assert wire == [{"role": "user", "content": "hi"}]


class TestReplayWireIntegration:
    def test_full_context_strategy_produces_paired_wire(self):
        """The recorded tool history reaches the model as a real tool exchange."""
        llm = FakeLLM(["answer"])
        run_sample(
            strategy=build("full-context"),
            sample=tiny_sample(),
            questions=[Question(kind="needle", question="q?", answer="a")],
            llm=llm,
            counter=CharEstimator(),
            budget_tokens=100_000,
        )

        wire = llm.calls[0]["messages"]
        assistant_with_calls = [m for m in wire if m.get("tool_calls")]
        tool_results = [m for m in wire if m["role"] == "tool"]
        assert assistant_with_calls, "assistant tool_calls must survive to the wire"
        assert len(tool_results) == 4
        for a, t in zip(assistant_with_calls, tool_results, strict=True):
            assert t["tool_call_id"] == a["tool_calls"][0]["id"]
        # No flattened fallback in the paired region
        assert not any(
            m["role"] == "user" and m["content"].startswith("Observation (recorded)")
            for m in wire
        )


class TestRunSample:
    def test_replay_records_occupancy_without_llm_calls(self):
        llm = FakeLLM([])
        result = run_sample(
            strategy=build("full-context"),
            sample=tiny_sample(),
            questions=[],
            llm=llm,
            counter=CharEstimator(),
            budget_tokens=100_000,
        )

        assert len(result.turns) == 9
        assert result.turns[0].view_tokens > 0
        assert result.overflowed_at is None
        assert llm.calls == []  # open-loop replay: no generation
        assert result.error is None

    def test_full_context_overflows_at_tight_budget(self):
        result = run_sample(
            strategy=build("full-context"),
            sample=tiny_sample(),
            questions=[Question(kind="needle", question="q?", answer="a")],
            llm=FakeLLM([]),
            counter=CharEstimator(),
            budget_tokens=50,
        )

        assert result.overflowed_at is not None
        assert result.answers == []
        assert result.error is None

    def test_sliding_window_answers_questions_with_telemetry(self):
        sample, questions = inject_needles(tiny_sample(), n=2, seed=3)
        llm = FakeLLM(["the value", "no idea"])
        result = run_sample(
            strategy=build("sliding-window"),
            sample=sample,
            questions=questions,
            llm=llm,
            counter=CharEstimator(),
            budget_tokens=4_000,
        )

        assert result.overflowed_at is None
        assert len(result.answers) == len(questions)
        assert result.usage.llm_calls == 2
        assert result.usage.prompt_tokens == 20
        assert result.model == "fake-model"
        assert result.provider == "fake-provider"
        first = llm.calls[0]["messages"]
        assert first[0]["role"] == "system"
        assert first[-1]["content"] == questions[0].question

    def test_tool_loop_interleaves_recall_tools(self):
        from context_eval.provider import ToolSchema
        from context_eval.strategies.base import EngineTool
        from context_eval.strategies.full_context import FullContext

        class RecallStrategy(FullContext):
            name = "fake-recall"

            def tools(self, engine):
                schema = ToolSchema(
                    name="recall",
                    description="Recall a message",
                    parameters={
                        "type": "object",
                        "properties": {"needle": {"type": "string"}},
                    },
                )

                def execute(args: Mapping[str, Any]) -> str:
                    return f"recalled[{args.get('needle')}]"

                return [EngineTool(schema=schema, execute=execute)]

        llm = FakeLLM([
            {"id": "call_1", "name": "recall", "arguments": '{"needle": "token"}'},
            "the token is abc-123",
        ])
        result = run_sample(
            strategy=RecallStrategy(),
            sample=tiny_sample(),
            questions=[
                Question(kind="needle", question="What is the token?", answer="abc-123")
            ],
            llm=llm,
            counter=CharEstimator(),
            budget_tokens=100_000,
        )

        record = result.answers[0]
        assert record.score == 1.0
        assert record.tool_calls == 1
        assert result.usage.llm_calls == 2
        second = llm.calls[1]["messages"]
        roles = [m["role"] for m in second]
        assert "tool" in roles
        tool_results = [m for m in second if m["role"] == "tool"]
        # Historical observations pair earlier; the recall result is last
        assert tool_results[-1]["content"] == "recalled[token]"

    def test_tool_loop_terminates_at_step_cap(self):
        from context_eval.provider import ToolSchema
        from context_eval.strategies.base import EngineTool
        from context_eval.strategies.full_context import FullContext

        class ToolStrategy(FullContext):
            name = "toolful"

            def tools(self, engine):
                schema = ToolSchema(
                    name="t",
                    description="t",
                    parameters={"type": "object", "properties": {}},
                )
                return [EngineTool(schema=schema, execute=lambda args: "ok")]

        endless = [{"id": f"c{i}", "name": "t", "arguments": "{}"} for i in range(99)]
        llm = FakeLLM([*endless[:MAX_TOOL_STEPS], "final answer after cap"])
        result = run_sample(
            strategy=ToolStrategy(),
            sample=tiny_sample(),
            questions=[Question(kind="needle", question="q?", answer="final")],
            llm=llm,
            counter=CharEstimator(),
            budget_tokens=100_000,
        )

        assert result.answers[0].prediction == "final answer after cap"
        assert len(llm.calls) == MAX_TOOL_STEPS + 1

    def test_unknown_tool_returns_error_not_crash(self):
        from context_eval.provider import ToolSchema
        from context_eval.strategies.base import EngineTool
        from context_eval.strategies.full_context import FullContext

        class NoTools(FullContext):
            name = "toolless-with-call"

            def tools(self, engine):
                schema = ToolSchema(
                    name="real",
                    description="real",
                    parameters={"type": "object", "properties": {}},
                )
                return [EngineTool(schema=schema, execute=lambda args: "ok")]

        llm = FakeLLM([
            {"id": "c1", "name": "nonexistent", "arguments": "{}"},
            "gave up",
        ])
        result = run_sample(
            strategy=NoTools(),
            sample=tiny_sample(),
            questions=[Question(kind="needle", question="q?", answer="x")],
            llm=llm,
            counter=CharEstimator(),
            budget_tokens=100_000,
        )

        followup = llm.calls[1]["messages"]
        tool_results = [m for m in followup if m["role"] == "tool"]
        assert tool_results[-1]["content"].startswith("error: unknown tool")
        assert result.answers[0].prediction == "gave up"

    def test_question_phase_failure_is_contained_per_question(self):
        script = [
            "first answer",
            RuntimeError("boom on second question"),
        ]

        class Exploding:
            def __init__(self):
                self.n = 0

            def complete(self, messages, *, tools=None, max_tokens=1024,
                 temperature=0.0, reasoning=None, require_content=False):
                from context_eval.provider import Completion

                item = script[self.n]
                self.n += 1
                if isinstance(item, Exception):
                    raise item
                return Completion(
                    text=item,
                    tool_calls=(),
                    usage=Usage(prompt_tokens=5, completion_tokens=1, llm_calls=1),
                    finish_reason="stop",
                    model="fake",
                )

        result = run_sample(
            strategy=build("full-context"),
            sample=tiny_sample(),
            questions=[
                Question(kind="needle", question="q1?", answer="a"),
                Question(kind="needle", question="q2?", answer="b"),
                Question(kind="needle", question="q3?", answer="first answer"),
            ],
            llm=Exploding(),
            counter=CharEstimator(),
            budget_tokens=100_000,
        )

        assert len(result.answers) == 3  # later questions still asked
        assert result.answers[0].prediction == "first answer"
        assert result.answers[1].error is not None
        assert "boom" in result.answers[1].error
        assert result.answers[1].score_method == "error"
        assert result.error is None  # run itself did not fail

    def test_strategy_error_recorded_not_raised(self):
        class Exploding(build("sliding-window").__class__):
            name = "exploding"

            def assemble(self, engine):
                raise RuntimeError("boom")

        result = run_sample(
            strategy=Exploding(),
            sample=tiny_sample(),
            questions=[],
            llm=FakeLLM([]),
            counter=CharEstimator(),
            budget_tokens=1_000,
        )

        assert result.error is not None
        assert "boom" in result.error
