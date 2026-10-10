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
        self.kwargs: list[dict] = []

    def complete(self, messages, *, tools=None, max_tokens=1024,
                 temperature=0.0, reasoning=None, require_content=False):
        self.calls += 1
        self.prompts.append(messages[-1]["content"] if messages else "")
        self.kwargs.append({"reasoning": reasoning, "require_content": require_content})

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

    def test_pre_trigger_view_is_full_verbatim_history(self):
        """The regime that matters: view between keep-tail and trigger.

        Deployed /compact shows the full verbatim history until the trigger
        fires; the 25% tail cap must NOT silently evict 60% of the session
        pre-compaction (this exact bug shipped once -- keep it pinned)."""
        llm = StubLLM()
        msgs = [("tool", f"block {i} " + "z" * 200) for i in range(30)]
        eng = make_engine(msgs, budget_tokens=6_000, llm=llm)  # ~1.7K raw < 0.9*6K
        s = FlatCompact()
        for _ in range(len(eng.store.messages())):
            s.on_turn(eng)
        assert s.stats().compactions == 0
        view = s.assemble(eng)
        assert len(view) == 1 + 30  # every message, not a 25% tail
        blocks = sorted(int(m.content.split()[1]) for m in view if m.content.startswith("block"))
        assert blocks == list(range(30))  # all thirty survived verbatim

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
        assert "history_search" in tools and "history_recall" in tools
        hits = tools["history_search"].execute({"query": "pytest pinned"})
        assert "8.2.1" in hits or hits != "no matches"


class TestContextEditing:
    def test_under_budget_is_verbatim(self):
        eng = make_engine(corpus(), budget_tokens=10_000)
        view = ContextEditing().assemble(eng)
        assert [m.content for m in view] == [c for _, c in corpus()]

    def test_clears_oldest_tool_payloads_first(self):
        # trigger at EDIT_TAU=0.90: 255-token corpus vs 216 → clears only
        # enough to get back under
        eng = make_engine(corpus(), budget_tokens=240)
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

    def test_assistant_heavy_transcript_stays_over_budget(self):
        """Documented regime boundary (same class as ARC): editing only
        clears TOOL payloads, so assistant-heavy transcripts cannot fit.
        The driver records overflow -- that is data, not a bug."""
        msgs = [("assistant", "thinking " + "a" * 300)] * 30
        eng = make_engine(msgs, budget_tokens=400)
        view = ContextEditing().assemble(eng)
        assert all(m.content.startswith("thinking") for m in view)  # untouched


def test_count_messages_includes_tool_call_arguments():
    """Wire-format fairness: tool-call args ride the wire but live in meta.
    Without counting them, views on SWE traces undercount ~10-16% (measured
    on data/swe_smoke.jsonl) in a strategy-dependent direction."""
    from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS, CharEstimator
    from context_eval.types import Message, Role

    counter = CharEstimator()
    bare = Message(id="m1", role=Role.ASSISTANT, content="run the tests", meta={})
    calls = Message(
        id="m2",
        role=Role.ASSISTANT,
        content="run the tests",
        meta={"tool_calls": [{"name": "bash", "arguments": {"cmd": "uv run pytest" * 20}}]},
    )
    bare_cost = counter.count_messages([bare])
    calls_cost = counter.count_messages([calls])
    assert bare_cost == (len("run the tests") + 3) // 4 + MESSAGE_OVERHEAD_TOKENS
    assert calls_cost > bare_cost * 2  # argument tokens materially counted


def test_strategies_hold_budget_under_tool_args_load():
    """Regression: internal budget arithmetic must match count_messages
    (content + overhead + tool_call args). Before the fix, window/editing
    kept messages whose args pushed the real view over budget, and
    flat-compact's trigger under-read ~1.5K tokens so it compacted too
    late and overflowed (live smoke: flat overflow@91 with the trigger
    threshold already passed)."""
    calls = [{"name": "bash", "arguments": {"cmd": "x" * 4000}}]  # ~1000 tok of args
    msgs = []
    for i in range(40):
        msgs.append(
            ("assistant", f"step {i}", {"tool_calls": calls})
        )
        msgs.append(("tool", "result " + "r" * 120))
    budget = 16_000

    store = TranscriptStore()
    for role, content, *meta in msgs:
        store.append(Role(role), content, meta=meta[0] if meta else {})
    eng = Engine(
        store=store,
        counter=CharEstimator(),
        budget=Budget(max_context_tokens=budget),
    )
    oracle = CharEstimator()

    from context_eval.strategies.sliding_window import SlidingWindow

    for strat in (SlidingWindow(), ContextEditing()):
        view = strat.assemble(eng)
        real = oracle.count_messages(view)
        assert real <= budget, f"{strat.name}: view {real} > {budget}"

    # ARC cannot shrink assistant tool_call args (design boundary), but on
    # a tool-payload-dominated corpus its cited view must hold the budget.
    store2 = TranscriptStore()
    for i in range(40):
        store2.append(
            Role.ASSISTANT,
            f"step {i}",
            meta={"tool_calls": [{"name": "bash", "arguments": {"cmd": f"cmd {i}"}}]},
        )
        store2.append(Role.TOOL, "result " + "r" * 2400)  # ~600 tok each
    eng2 = Engine(
        store=store2,
        counter=CharEstimator(),
        budget=Budget(max_context_tokens=budget),
    )
    from context_eval.strategies.arc import ARCCitations

    arc_view = ARCCitations().assemble(eng2)
    arc_real = oracle.count_messages(arc_view)
    assert arc_real <= budget, f"arc: view {arc_real} > {budget}"

    # flat-compact's trigger reads the same wire cost as the oracle
    from context_eval.strategies.flat_compact import FlatCompact

    s = FlatCompact()
    s.on_turn(eng)  # llm is None: cannot compact, but the trigger reads true cost
    raw = s._raw_view_tokens(eng)
    verbatim = oracle.count_messages([m.to_prompt() for m in store.messages()])
    assert abs(raw - verbatim) < 32  # differs only by the summary marker (~2 msgs)

    # representation's trigger reads the same wire cost as the oracle
    from context_eval.strategies.representation import RepresentationMemory

    r = RepresentationMemory()
    r.on_turn(eng)  # llm is None: same no-compaction-but-true-cost path
    r_raw = r._raw_view_tokens(eng)
    assert abs(r_raw - verbatim) < 32


def test_registry_entries():
    from context_eval.strategies import REGISTRY, build

    assert set(REGISTRY) >= {"flat-compact", "context-editing"}
    assert build("flat-compact").name == "flat-compact"
    assert build("context-editing").name == "context-editing"


class TestCalibratedCall:
    def test_fold_uses_calibrated_kwargs(self):
        llm = StubLLM()
        eng = make_engine(corpus(), budget_tokens=200, llm=llm)
        s = FlatCompact()
        for _ in range(len(eng.store.messages())):
            s.on_turn(eng)
        assert llm.calls >= 1
        assert all(k["reasoning"] == {"enabled": False} for k in llm.kwargs)
        assert all(k["require_content"] for k in llm.kwargs)
        assert "150 words" in llm.prompts[0]
