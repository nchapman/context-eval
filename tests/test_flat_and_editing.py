"""Tests for flat-compact and context-editing strategies."""

from __future__ import annotations

from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine
from context_eval.strategies.context_editing import ContextEditing
from context_eval.strategies.flat_compact import FlatCompact
from context_eval.tokens import CharEstimator
from context_eval.types import Role, Usage


class StubLLM:
    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, messages, *, tools=None, max_tokens=1024, temperature=0.0):
        self.calls += 1
        self.prompts.append(messages[-1]["content"] if messages else "")

        class R:
            text = "summary: fixed flaky parser test; version 2.3.1; pytest 8.2"
            tool_calls = ()
            usage = Usage(prompt_tokens=50, completion_tokens=20)

        return R()


def make_engine(msgs, budget_tokens, llm=None):
    store = TranscriptStore()
    for role, content in msgs:
        store.append(Role(role), content)
    return Engine(
        store=store,
        counter=CharEstimator(),
        budget=Budget(max_context_tokens=budget_tokens),
        llm=llm,
    )


def corpus():
    return [
        ("user", "we pinned pytest to 8.2.1 in CI"),
        ("assistant", "updated pyproject"),
        ("tool", "test run output: 412 passed, 3 failed " + "x" * 400),
        ("assistant", "fixed two of the failures"),
        ("tool", "coverage report " + "y" * 400),
        ("user", "what was the pinned version?"),
    ]


class TestFlatCompact:
    def test_no_compaction_under_threshold(self):
        llm = StubLLM()
        eng = make_engine(corpus(), budget_tokens=10_000, llm=llm)
        s = FlatCompact()
        for _ in range(len(eng.store.messages())):
            s.on_turn(eng)
        assert s.stats().compactions == 0
        assert llm.calls == 0
        view = s.assemble(eng)
        assert len(view) == 1 + len(corpus())  # tail marker + all msgs verbatim

    def test_compacts_once_over_threshold_and_keeps_tail(self):
        llm = StubLLM()
        eng = make_engine(corpus(), budget_tokens=200, llm=llm)
        s = FlatCompact()
        for _ in range(len(eng.store.messages())):
            s.on_turn(eng)

        assert s.stats().compactions == 1
        assert llm.calls == 1
        view = s.assemble(eng)
        joined = "\n".join(m.content for m in view)
        assert "[Session summary" in joined
        assert "[Recent context — verbatim]" in joined
        # the newest message is kept verbatim
        assert any("pinned version" in m.content for m in view)

    def test_summary_of_summary_on_repeated_triggers(self):
        llm = StubLLM()
        store = TranscriptStore()
        eng = Engine(
            store=store,
            counter=CharEstimator(),
            budget=Budget(max_context_tokens=300),
            llm=llm,
        )
        s = FlatCompact()
        for i in range(12):
            store.append(Role.TOOL, f"block {i} " + "z" * 200)
            s.on_turn(eng)  # interleaved like the driver: append, then react
        assert s.stats().compactions >= 2  # compounding regime engaged
        # later compactions folded the previous summary back in
        assert any("PREVIOUS SUMMARY" in p for p in llm.prompts[1:])

    def test_search_tools_cover_archive(self):
        llm = StubLLM()
        eng = make_engine(corpus(), budget_tokens=10_000, llm=llm)
        s = FlatCompact()
        for _ in range(len(eng.store.messages())):
            s.on_turn(eng)
        tools = {t.schema.name: t for t in s.tools(eng)}
        assert "search_history" in tools and "recall_history" in tools
        hits = tools["search_history"].execute({"query": "pytest pinned"})
        assert "8.2.1" in hits or hits != "no matches"


class TestContextEditing:
    def test_under_budget_is_verbatim(self):
        eng = make_engine(corpus(), budget_tokens=10_000)
        view = ContextEditing().assemble(eng)
        assert [m.content for m in view] == [c for _, c in corpus()]

    def test_clears_oldest_tool_payloads_first(self):
        eng = make_engine(corpus(), budget_tokens=200)
        view = ContextEditing().assemble(eng)
        contents = [m.content for m in view]
        # oldest tool result cleared to a stub, newest stays verbatim
        assert any(c.startswith("[cleared] test run output") for c in contents)
        assert not any(c.startswith("[cleared] coverage") for c in contents)
        # non-tool messages untouched
        assert "we pinned pytest to 8.2.1 in CI" in contents

    def test_never_uses_llm(self):
        llm = StubLLM()
        eng = make_engine(corpus(), budget_tokens=200, llm=llm)
        ContextEditing().assemble(eng)
        assert llm.calls == 0


def test_registry_entries():
    from context_eval.strategies import REGISTRY, build

    assert set(REGISTRY) >= {"flat-compact", "context-editing"}
    assert build("flat-compact").name == "flat-compact"
    assert build("context-editing").name == "context-editing"
