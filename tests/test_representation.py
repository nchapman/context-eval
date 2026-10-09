"""Tests for the representation-memory strategy (Honcho-mechanism port)."""

from __future__ import annotations

from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine
from context_eval.strategies.representation import RepresentationMemory
from context_eval.tokens import CharEstimator
from context_eval.types import Role, Usage


class StubLLM:
    """Records every prompt; returns configurable completions in order."""

    def __init__(self, texts=None):
        self.calls = 0
        self.prompts: list[str] = []
        self._texts = texts or ["memory: pinned pytest 8.2.1; open flaky parser test"]

    def complete(self, messages, *, tools=None, max_tokens=1024, temperature=0.0):
        self.calls += 1
        self.prompts.append(messages[-1]["content"] if messages else "")
        text = self._texts[min(self.calls - 1, len(self._texts) - 1)]

        class R:
            tool_calls = ()
            usage = Usage(prompt_tokens=50, completion_tokens=20)
            finish_reason = "stop"

        r = R()
        r.text = text
        return r


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


def fill(n, marker="x"):
    return [
        ("user", f"message {i} about topic-{i}: {marker * 300}")
        for i in range(n)
    ]


def drive(eng, strat):
    """Replay every message through on_turn, as the driver does."""
    for _ in range(len(eng.store.messages())):
        strat.on_turn(eng)


class TestRepresentationMemory:
    def test_no_update_under_threshold(self):
        llm = StubLLM()
        eng = make_engine([("user", "hi")], 8000, llm)
        s = RepresentationMemory()
        drive(eng, s)
        assert llm.calls == 0
        assert s.stats().compactions == 0

    def test_pre_trigger_view_is_full_verbatim_history(self):
        msgs = [("user", f"m{i} " + "x" * 100) for i in range(5)]
        eng = make_engine(msgs, 8000)
        s = RepresentationMemory()
        view = s.assemble(eng)
        contents = [m.content for m in view]
        assert contents[0] == "[Recent context — verbatim]"
        assert all(any(f"m{i}" in c for c in contents[1:]) for i in range(5))

    def test_update_fires_over_threshold_and_keeps_tail(self):
        llm = StubLLM()
        eng = make_engine(fill(30), 1500, llm)
        s = RepresentationMemory()
        drive(eng, s)
        assert llm.calls == 1
        assert "pytest 8.2.1" in s._memory
        view = s.assemble(eng)
        assert "Agent memory" in view[0].content
        assert "[Recent context — verbatim]" in view[1].content
        # the tail is the most recent messages, still verbatim
        assert "topic-29" in view[-1].content
        stats = s.stats()
        assert stats.compactions == 1
        assert stats.empty_summaries == 0

    def test_update_input_is_memory_plus_delta_only(self):
        """The O(memory + delta) invariant: no update prompt ever contains
        messages older than the previous fold -- the compounding input of
        fold-over-fold summarizers cannot occur here."""
        llm = StubLLM(texts=[
            "memory v1: topic-0 tracked",
            "memory v2: topic-0, topic-10 tracked",
            "memory v3: should never be needed",
        ])
        eng = make_engine(fill(30), 1500, llm)
        s = RepresentationMemory()
        drive(eng, s)
        first_fold_end = s._folded_upto
        assert "topic-0" in llm.prompts[0]

        # grow history until the next trigger; the second fold must see
        # only the new delta plus the memory text -- none of the first
        # fold's transcript
        i = 30
        while llm.calls < 2 and i < 120:
            eng.store.append(Role.USER, f"message {i} about topic-{i}: " + "y" * 300)
            s.on_turn(eng)
            i += 1
        assert llm.calls == 2
        second = llm.prompts[1]
        assert "memory v1" in second  # current memory is the input
        assert "topic-29" in second  # new delta arrives since first fold
        # a message folded by update 1 must not reappear as transcript
        # (anchored to the full phrase: "topic-3" is a prefix of "topic-30")
        assert "about topic-3:" not in second
        assert "about topic-10:" not in second
        assert s._folded_upto > first_fold_end

    def test_empty_update_preserves_memory_and_counts(self):
        llm = StubLLM(texts=["memory v1: first fold landed", "memory v2 arrived"])
        base_complete = llm.complete

        def complete(messages, **kw):
            r = base_complete(messages, **kw)
            if llm.calls == 2:
                r.text = ""
                r.finish_reason = "length"
            return r

        llm.complete = complete
        eng = make_engine(fill(30), 1500, llm)
        s = RepresentationMemory()
        drive(eng, s)
        assert s._memory == "memory v1: first fold landed"

        i = 30
        while s._updates < 2 and i < 120:
            eng.store.append(Role.USER, f"message {i} about topic-{i}: " + "z" * 300)
            s.on_turn(eng)
            i += 1
        # the truncated update must not have wiped the memory
        assert s._memory == "memory v1: first fold landed"
        assert s.stats().empty_summaries == 1
        assert s.stats().compactions == 2

    def test_search_tools_cover_archive(self):
        llm = StubLLM()
        eng = make_engine(fill(30), 1500, llm)
        s = RepresentationMemory()
        drive(eng, s)
        tools = {t.schema.name: t.execute for t in s.tools(eng)}
        out = tools["history_search"]({"query": "topic-27"})
        assert "m000" in out
        mid = out.split(":")[0]
        assert "topic-27" in tools["history_recall"]({"id": mid})

    def test_registry_contains_representation(self):
        from context_eval.strategies import REGISTRY, build

        assert "representation" in REGISTRY
        assert build("representation").name == "representation"

    def test_first_update_empty_leaves_arm_doc_less(self):
        """flat's historical silent suite failure: if the FIRST update
        truncates to nothing, the view carries no map at all -- only the
        tail. The telemetry must flag it."""
        llm = StubLLM(texts=[""])
        base_complete = llm.complete

        def complete(messages, **kw):
            r = base_complete(messages, **kw)
            r.text = ""
            r.finish_reason = "length"
            return r

        llm.complete = complete
        eng = make_engine(fill(30), 1500, llm)
        s = RepresentationMemory()
        drive(eng, s)
        assert s._memory == ""
        assert s.stats().empty_summaries == 1
        view = s.assemble(eng)
        assert not any("[Agent memory" in m.content for m in view)
        assert view[0].content == "[Recent context — verbatim]"

    def test_parity_with_flat_fold_sequence(self):
        """Comparability drift-catcher: on an identical transcript, both
        arms must fire updates on the same turns with equal per-update
        input sizes (verified at review time; this pins it)."""
        from context_eval.strategies.flat_compact import FlatCompact

        texts = ["m: " + "fact. " * 30] * 6
        flat_llm, rep_llm = StubLLM(texts), StubLLM(texts)
        msgs = fill(40)
        eng_f = make_engine(msgs, 4000, flat_llm)
        eng_r = make_engine(msgs, 4000, rep_llm)
        f, r = FlatCompact(), RepresentationMemory()
        for _ in range(len(msgs)):
            f.on_turn(eng_f)
            r.on_turn(eng_r)
        assert flat_llm.calls == rep_llm.calls
        for pf, pr in zip(flat_llm.prompts, rep_llm.prompts, strict=True):
            # identical deltas, differing only by the resident-doc framing
            assert abs(len(pf) - len(pr)) < 400

    def test_prompt_size_bounded_across_updates(self):
        """With a fixed-size memory, update prompts stay bounded by the
        trigger window + memory, not by total history length."""
        from context_eval.budget import Budget
        from context_eval.store import TranscriptStore
        from context_eval.strategies.base import Engine
        from context_eval.tokens import CharEstimator

        llm = StubLLM(texts=["m: " + "fact. " * 20] * 8)  # constant-size memory
        store = TranscriptStore()
        eng = Engine(store=store, counter=CharEstimator(),
                     budget=Budget(max_context_tokens=1500), llm=llm)
        s = RepresentationMemory()
        for i in range(120):  # interleave append + on_turn, as the driver does
            store.append(Role.USER, f"message {i} about topic-{i}: " + "y" * 300)
            s.on_turn(eng)
        assert llm.calls >= 3
        sizes = [len(p) for p in llm.prompts]
        # every prompt is bounded by (fixed memory + one trigger window);
        # a fold-over-fold input would grow linearly with history
        assert max(sizes) < 1.5 * min(sizes)
