from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine
from context_eval.strategies.lcm import LCMDagger
from context_eval.tokens import CachedCounter, CharEstimator
from context_eval.types import Role, Usage


class ScriptedLLM:
    """Returns queued summaries; fails loudly when the queue is exhausted."""

    def __init__(self, texts=None, fail=False):
        self.texts = list(texts or [])
        self.fail = fail
        self.calls = []

    def complete(self, messages, *, tools=None, max_tokens=1024, temperature=0.0):
        from context_eval.provider import Completion

        self.calls.append(messages[0]["content"])
        if self.fail:
            raise RuntimeError("summarizer unavailable")
        if not self.texts:
            raise AssertionError(
                f"unexpected extra summarizer call #{len(self.calls) + 1}"
            )
        text = self.texts.pop(0)
        return Completion(
            text=text,
            tool_calls=(),
            usage=Usage(prompt_tokens=100, completion_tokens=50, llm_calls=1),
            finish_reason="stop",
            model="fake",
        )


def make_engine(store, budget_tokens, llm=None):
    return Engine(
        store=store,
        counter=CachedCounter(CharEstimator()),
        budget=Budget(max_context_tokens=budget_tokens),
        llm=llm,
    )


def fill_turns(store, n, body="x" * 40):
    """n rounds of (user, assistant, tool): ~14 tokens per message."""
    for i in range(n):
        store.append(Role.USER, f"turn {i}: {body}")
        store.append(Role.ASSISTANT, f"step {i}")
        store.append(Role.TOOL, f"obs {i}: {body}")


class TestThresholds:
    def test_below_soft_threshold_stays_verbatim(self):
        store = TranscriptStore()
        fill_turns(store, 5)
        llm = ScriptedLLM()
        strategy = LCMDagger()
        engine = make_engine(store, 10_000, llm)
        strategy.on_turn(engine)

        view = strategy.assemble(engine)
        assert view == [m.to_prompt() for m in store.messages()]
        assert llm.calls == []
        assert strategy.stats().compactions == 0

    def test_crossing_soft_compacts_oldest_chunk(self):
        store = TranscriptStore()
        fill_turns(store, 60)  # ~60*3*14 = 2520 tokens
        llm = ScriptedLLM(["SUMMARY-1 [§m000001] fixed parser bug"])
        strategy = LCMDagger()
        engine = make_engine(store, 3_000, llm)  # soft=2100
        strategy.on_turn(engine)

        assert len(llm.calls) == 1
        assert "SEGMENT" in llm.calls[0]
        assert "[§m000001 " in llm.calls[0]  # chunk starts at the oldest message
        view = strategy.assemble(engine)
        flat = "\n".join(m.content for m in view)
        assert "SUMMARY-1" in flat
        assert "[Recent context" in flat
        # Oldest messages replaced: their original text left the view
        assert "turn 0:" not in flat
        assert engine.counter.count_messages(view) < engine.counter.count_messages(
            [m.to_prompt() for m in store.messages()]
        )
        stats = strategy.stats()
        assert stats.compactions == 1
        assert stats.overhead_usage.llm_calls == 1

    def test_view_brought_back_under_soft_threshold(self):
        store = TranscriptStore()
        fill_turns(store, 80)
        llm = ScriptedLLM([f"summary-{i}" for i in range(20)])
        strategy = LCMDagger()
        engine = make_engine(store, 3_000, llm)
        strategy.on_turn(engine)

        budget = engine.budget.max_context_tokens
        assert engine.counter.count_messages(strategy.assemble(engine)) <= int(
            budget * strategy.tau_soft
        ) + 50  # tolerance for one chunk granularity
        assert len(llm.calls) <= 20  # queue exhausted -> test fails loudly


class TestHierarchy:
    def test_four_siblings_consolidate_into_depth2(self):
        store = TranscriptStore()
        fill_turns(store, 120)
        llm = ScriptedLLM([f"d1-{i}" for i in range(4)] + ["d2-consolidated"])
        strategy = LCMDagger()
        engine = make_engine(store, 3_000, llm)
        strategy.on_turn(engine)

        flat = "\n".join(m.content for m in strategy.assemble(engine))
        assert "(D2, " in flat  # a consolidated parent is rendered
        assert "d2-consolidated" in flat
        # Consolidation prompt received the child summaries
        assert any("d1-0" in c and "SUMMARIES" in c for c in llm.calls)

    def test_nodes_render_in_chronological_order(self):
        store = TranscriptStore()
        fill_turns(store, 240)
        llm = ScriptedLLM([f"s{i}" for i in range(20)])
        strategy = LCMDagger()
        engine = make_engine(store, 3_000, llm)
        strategy.on_turn(engine)

        flat = "\n".join(m.content for m in strategy.assemble(engine))
        import re

        ranges = re.findall(r"§(m\d+)\.\.§(m\d+)", flat)
        assert len(ranges) >= 2
        # Summary blocks must be ordered oldest -> newest by covered range
        for (_prev_start, prev_stop), (next_start, _) in zip(ranges, ranges[1:], strict=False):
            assert next_start >= prev_stop, f"summary order inverted: {ranges}"

    def test_covers_track_message_ids_transitively(self):
        store = TranscriptStore()
        fill_turns(store, 120)
        llm = ScriptedLLM([f"s{i}" for i in range(10)])
        strategy = LCMDagger()
        engine = make_engine(store, 3_000, llm)
        strategy.on_turn(engine)

        all_covers = set()
        for n in strategy._nodes:
            if n is not None:
                all_covers.update(n.covers)
        assert all_covers  # ids are real store ids
        assert all(engine.store.get(mid) is not None for mid in list(all_covers)[:5])


class TestEscalation:
    def test_summarizer_sees_full_content_beyond_2k_chars(self):
        """D1 prompts must not truncate long observations: deep content
        (and injected needles anywhere in the observation) has to reach the
        summarizer, or LCM is handicapped vs flat-compact."""
        deep = "needle habitat: the answer value ZQX-99 lives here " + "f" * 4000
        store = TranscriptStore()
        store.append(Role.TOOL, deep)
        llm = ScriptedLLM(texts=["summary of deep"])
        eng = make_engine(store, budget_tokens=400, llm=llm)
        s = LCMDagger()
        store.append(Role.USER, "next turn")
        s.on_turn(eng)  # force a compaction
        assert llm.calls, "expected a summarizer call"
        assert "ZQX-99" in llm.calls[0]  # deep content reached the prompt

    def test_summarizer_failure_degrades_to_pointer_nodes(self):
        store = TranscriptStore()
        fill_turns(store, 60)
        llm = ScriptedLLM(fail=True)
        strategy = LCMDagger()
        engine = make_engine(store, 3_000, llm)
        strategy.on_turn(engine)  # must not raise

        flat = "\n".join(m.content for m in strategy.assemble(engine))
        assert "omitted deterministically" in flat
        assert "history_recall" in flat
        assert strategy.stats().compactions >= 1

    def test_no_llm_means_no_compaction(self):
        store = TranscriptStore()
        fill_turns(store, 60)
        strategy = LCMDagger()
        engine = make_engine(store, 3_000, None)
        strategy.on_turn(engine)

        assert strategy.stats().compactions >= 1  # pointer nodes
        view = strategy.assemble(engine)
        assert engine.counter.count_messages(view) < engine.counter.count_messages(
            [m.to_prompt() for m in store.messages()]
        )


class TestRecallTools:
    def test_search_and_recall_recover_needle(self):
        store = TranscriptStore()
        fill_turns(store, 3)
        needle = "NOTE (workspace audit): the canary deployment token is ab12-CD34Q."
        store.append(Role.TOOL, "z" * 200 + "\n" + needle)
        strategy = LCMDagger()
        engine = make_engine(store, 10_000, ScriptedLLM())
        strategy.on_turn(engine)

        tools = {t.schema.name: t for t in strategy.tools(engine)}
        hits = tools["history_search"].execute({"query": "canary deployment token"})
        # Search hits reference archive ids (snippets may legitimately reveal
        # content — search is itself a recovery path)
        assert any(mid in hits for mid in [m.id for m in store.messages()])

        needle_id = next(m.id for m in store.messages() if needle in m.content)
        recalled = tools["history_recall"].execute({"id": needle_id})
        assert needle in recalled
        assert strategy.stats().recall_tool_calls == 2

    def test_unknown_recall_id_errors(self):
        store = TranscriptStore()
        store.append(Role.USER, "hi")
        strategy = LCMDagger()
        engine = make_engine(store, 1_000)
        tools = {t.schema.name: t for t in strategy.tools(engine)}
        assert tools["history_recall"].execute({"id": "m999"}).startswith("error:")

    def test_search_handles_empty_query(self):
        store = TranscriptStore()
        store.append(Role.USER, "hello world")
        strategy = LCMDagger()
        engine = make_engine(store, 1_000)
        tools = {t.schema.name: t for t in strategy.tools(engine)}
        assert tools["history_search"].execute({"query": "  "}) == "no matches"


class TestFtsIndex:
    def test_ranks_relevant_matches(self):
        from context_eval.strategies.lcm import _FtsIndex

        idx = _FtsIndex()
        store = TranscriptStore()
        store.append(Role.USER, "the parser bug in tokenizer.rs")
        store.append(Role.USER, "unrelated weather talk")
        store.append(Role.USER, "fixed the parser in main.rs")
        idx.index(store.messages())

        hits = idx.search("parser")
        ids = [mid for mid, _ in hits]
        assert "m000001" in ids and "m000003" in ids
        assert "m000002" not in ids

    def test_snippet_contains_match(self):
        from context_eval.strategies.lcm import _FtsIndex

        idx = _FtsIndex()
        store = TranscriptStore()
        store.append(Role.TOOL, "output mentions frobnicator widget v2")
        idx.index(store.messages())
        _, snippet = idx.search("frobnicator")[0]
        assert "frobnicator" in snippet


class TestTopicsLine:
    """P1 prompt variant: summaries must enumerate named topics so the
    continuity probe measures the mechanism, not an unrequested omission."""

    def test_d1_prompt_requests_topics_line(self):
        from context_eval.strategies.lcm import _D1_PROMPT

        assert "TOPICS line" in _D1_PROMPT
        assert "never omit" in _D1_PROMPT.lower()

    def test_dn_prompt_unions_children_topics(self):
        from context_eval.strategies.lcm import _DN_PROMPT

        assert "union of the children" in _DN_PROMPT
        assert "never drop a named topic" in _DN_PROMPT

    def test_flat_shares_the_d1_topics_instruction(self):
        # flat imports _D1_PROMPT from lcm: the inventory instruction must
        # be symmetric across both summarizing arms or the comparison is
        # confounded.
        from context_eval.strategies.flat_compact import _D1_PROMPT as flat_d1
        from context_eval.strategies.lcm import _D1_PROMPT as lcm_d1

        assert flat_d1 is lcm_d1


class TestTopicsSwitch:
    def test_env_switch_controls_inventory_instruction(self, monkeypatch):
        import importlib

        import context_eval.strategies.lcm as lcm_mod

        monkeypatch.setenv("CONTEXT_EVAL_TOPICS_LINES", "0")
        importlib.reload(lcm_mod)
        assert "TOPICS" not in lcm_mod._D1_PROMPT
        assert "TOPICS" not in lcm_mod._DN_PROMPT
        monkeypatch.setenv("CONTEXT_EVAL_TOPICS_LINES", "1")
        importlib.reload(lcm_mod)
        assert "TOPICS line" in lcm_mod._D1_PROMPT
        assert "union of the children" in lcm_mod._DN_PROMPT

    def test_summary_max_tokens_env_override(self, monkeypatch):
        import importlib

        import context_eval.strategies.lcm as lcm_mod

        monkeypatch.setenv("CONTEXT_EVAL_SUMMARY_MAX_TOKENS", "8192")
        importlib.reload(lcm_mod)
        assert lcm_mod.SUMMARY_MAX_TOKENS == 8192
        monkeypatch.delenv("CONTEXT_EVAL_SUMMARY_MAX_TOKENS")
        importlib.reload(lcm_mod)
        assert lcm_mod.SUMMARY_MAX_TOKENS == 4096
