from collections.abc import Mapping
from typing import Any

from context_eval.datasets.base import ReplayMessage, ReplaySample
from context_eval.driver import run_sample, view_to_wire
from context_eval.questions import Question, inject_needles
from context_eval.strategies import build
from context_eval.tokens import CharEstimator
from context_eval.types import Role, ToolCall, Usage


class FakeLLM:
    """Scripted completions; records every call for assertions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, *, tools=None, max_tokens=1024, temperature=0.0):
        from context_eval.provider import Completion

        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": list(tools or [])}
        )
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
            tool_calls=(ToolCall(id=item["id"], name=item["name"], arguments=item["arguments"]),),
            usage=Usage(prompt_tokens=10, completion_tokens=3, llm_calls=1),
            finish_reason="tool_calls",
            model="fake-model",
            provider="fake-provider",
        )


def tiny_sample():
    messages = [ReplayMessage(role=Role.USER, content="Fix the failing test in foo.py")]
    for i in range(4):
        messages.append(
            ReplayMessage(role=Role.ASSISTANT, content="", meta={"tool_calls": [
                {"id": f"c{i}", "function": {"name": "bash", "arguments": "{}"}}
            ]})
        )
        messages.append(ReplayMessage(role=Role.TOOL, content=f"test output {i} " + "y" * 40))
    return ReplaySample(dataset="t/1/openhands", sample_id="tiny-1", messages=tuple(messages))


class TestViewToWire:
    def test_pairs_tool_calls_with_observations(self):
        view = [
            ReplayMessage(role=Role.ASSISTANT, content="", meta={"tool_calls": [
                {"id": "c0", "function": {"name": "bash", "arguments": "{}"}}
            ]}),
            ReplayMessage(role=Role.TOOL, content="obs"),
        ]
        wire = view_to_wire(view)

        assert wire[0]["tool_calls"][0]["id"] == "c0"
        assert wire[1] == {"role": "tool", "tool_call_id": "c0", "content": "obs"}

    def test_unpaired_observation_renders_as_user(self):
        wire = view_to_wire([ReplayMessage(role=Role.TOOL, content="orphan")])

        assert wire[0]["role"] == "user"
        assert "orphan" in wire[0]["content"]

    def test_plain_messages_pass_through(self):
        wire = view_to_wire([ReplayMessage(role=Role.USER, content="hi")])
        assert wire == [{"role": "user", "content": "hi"}]


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
        # Wire: system + view + question in the first call
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
            questions=[Question(kind="needle", question="What is the token?", answer="abc-123")],
            llm=llm,
            counter=CharEstimator(),
            budget_tokens=100_000,
        )

        record = result.answers[0]
        assert record.score == 1.0
        assert record.tool_calls == 1
        assert result.usage.llm_calls == 2
        # Second call contains the tool exchange
        second = llm.calls[1]["messages"]
        roles = [m["role"] for m in second]
        assert "tool" in roles
        tool_msg = next(m for m in second if m["role"] == "tool")
        assert tool_msg["content"] == "recalled[token]"

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
