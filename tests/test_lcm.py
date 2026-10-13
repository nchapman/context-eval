from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine
from context_eval.strategies.lcm import LCMDagger, summary_call_kwargs
from context_eval.tokens import CachedCounter, CharEstimator
from context_eval.types import Role, Usage


class ScriptedLLM:
    """Returns queued summaries; fails loudly when the queue is exhausted."""

    def __init__(self, texts=None, fail=False):
        self.texts = list(texts or [])
        self.fail = fail
        self.calls = []
        self.call_kwargs: list[dict] = []

    def complete(self, messages, *, tools=None, max_tokens=1024,
                 temperature=0.0, reasoning=None, require_content=False):
        from context_eval.provider import Completion

        self.calls.append(messages[0]["content"])
        self.call_kwargs.append({"reasoning": reasoning,
                                 "require_content": require_content})
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

    def test_phrase_match_outranks_vocabulary_collisions(self):
        """A probe line like `[handoff] Next:` must surface even when the
        corpus is full of single-term collisions (a pi session whose
        project keeps HANDOFF.md files buried it below top-5)."""
        from context_eval.strategies.lcm import _FtsIndex

        idx = _FtsIndex()
        texts = [
            "handoff note for the engine agent — priorities + reading list",
            "Now the MMQ attribution — trace both sides per the handoff",
            "The handoff answers my question directly and corrects it",
            "warp3090-perf/research/HANDOFF.md (6cf5a50)",
            "env.sh GOALS.md HANDOFF.md kvarn lib log.md",
        ]
        msgs = [
            type("M", (), {"id": f"m{i:06d}", "content": t})()
            for i, t in enumerate(texts)
        ]
        probe = type(
            "M", (), {"id": "m000099",
                       "content": "[handoff] Next: write the regression test on amber-BNU"}
        )()
        msgs.append(probe)
        idx.index(msgs)
        top_id, top_snip = idx.search("[handoff] Next:", limit=5)[0]
        assert top_id == "m000099"
        assert "amber-BNU" in top_snip


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

    def test_flat_no_longer_shares_d1(self):
        # flat now ships its own sectioned /compact template (dev100
        # finding: the shared 150-word prose cap discarded per-event
        # numbers). Pin the decoupling so the arms stay independently
        # faithful to their reference designs.
        import importlib

        flat = importlib.import_module("context_eval.strategies.flat_compact")
        assert not hasattr(flat, "_D1_PROMPT")
        assert "EVENTS:" in flat._FLAT_PROMPT



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


class TestCalibratedCall:
    def test_d1_and_dn_use_calibrated_kwargs(self):
        llm = ScriptedLLM(texts=[f"summary {i}" for i in range(12)])
        store = TranscriptStore()
        fill_turns(store, 60, body="x" * 60)
        eng = make_engine(store, 1_500, llm)
        s = LCMDagger()
        for _ in range(len(store.messages())):
            s.on_turn(eng)
        assert len(llm.calls) >= 2
        assert all(k["reasoning"] == {"enabled": False} for k in llm.call_kwargs)
        assert all(k["require_content"] for k in llm.call_kwargs)


class TestSummaryCallKwargs:
    def test_default_is_calibrated_off(self):
        kw = summary_call_kwargs()
        assert kw["reasoning"] == {"enabled": False}
        assert kw["max_tokens"] == 1024
        assert kw["require_content"] is True

    def test_env_switch_selects_reasoning_ablation(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_EVAL_SUMMARIZER_REASONING", "on")
        import importlib

        import context_eval.strategies.lcm as m
        importlib.reload(m)
        try:
            kw = m.summary_call_kwargs()
            assert "reasoning" not in kw
            assert kw["max_tokens"] == m.SUMMARY_MAX_TOKENS
            assert kw["require_content"] is True
        finally:
            monkeypatch.delenv("CONTEXT_EVAL_SUMMARIZER_REASONING")
            importlib.reload(m)


class TestBoundaryOverlap:
    def test_second_chunk_prompt_carries_context_tail(self):
        from context_eval.strategies.lcm import LCMDagger

        store = TranscriptStore()
        fill_turns(store, 60)  # ~2520 tokens: first on_turn crosses soft
        llm = ScriptedLLM(texts=["s1 [§m000001]", "s2 [§m000004]"])
        engine = make_engine(store, 3_000, llm)
        strategy = LCMDagger()
        strategy.on_turn(engine)
        assert len(llm.calls) == 1  # first chunk: no context yet
        assert "CONTEXT ENDS" not in llm.calls[0]
        fill_turns(store, 40)  # more history: next tooth crosses soft again
        strategy.on_turn(engine)
        # first call of this tooth summarizes the next chunk, with the
        # previous chunk's tail as context (later calls may consolidate)
        new_calls = llm.calls[1:]
        assert "CONTEXT ENDS" in new_calls[0]
        # the ids before CONTEXT ENDS also appeared in the first chunk's
        # prompt: they are overlap context, not new messages
        ctx = new_calls[0].split("CONTEXT ENDS")[0]
        import re as _re

        ctx_ids = set(_re.findall(r"§(m\d+)", ctx))
        assert ctx_ids and ctx_ids.issubset(set(_re.findall(r"§(m\d+)", llm.calls[0])))


class TestAttributedSnippets:
    def test_snippet_is_start_anchored_with_subject(self):
        from context_eval.strategies.lcm import _FtsIndex

        idx = _FtsIndex()
        idx.index([type("M", (), {"id": "m000001", "content":
            "By the way, I recently participated in a charity walk and "
            "managed to raise $250 through sponsors, which got me thinking "
            "about reducing waste. " + "filler " * 80})()])
        (mid, snip), = idx.search("raised $250 sponsors", limit=1)
        assert mid == "m000001"
        assert "I recently participated in a charity walk" in snip  # the subject
        assert "$250" in snip

    def test_deep_match_appends_window(self):
        from context_eval.strategies.lcm import _FtsIndex

        idx = _FtsIndex()
        idx.index([type("M", (), {"id": "m000001", "content":
            "Intro about cooking. " + "filler sentence here. " * 60
            + "The telescope aperture is 250mm on the Orion model."})()])
        (mid, snip), = idx.search("telescope aperture", limit=1)
        assert snip.startswith("Intro about cooking")
        assert "…[later]" in snip and "250mm" in snip

    def test_render_carries_role_header(self):
        from context_eval.store import TranscriptStore
        from context_eval.strategies.lcm import render_search_results
        from context_eval.types import Role

        store = TranscriptStore()
        store.append(Role.USER, "I raised $250 in a charity walk")
        out = render_search_results([("m000001", "I raised $250 ...")], store)
        assert out == "m000001 [user]: I raised $250 ..."


class TestDateAttribution:
    def _store(self):
        from context_eval.store import TranscriptStore
        from context_eval.types import Role

        store = TranscriptStore()
        store.append(Role.USER, "[2023/05/22] I did a charity walk and raised $250")
        store.append(Role.ASSISTANT, "Congratulations on the $250!")
        store.append(Role.USER, "[2023/06/10] Later, unrelated news about guitars")
        store.append(Role.ASSISTANT, "Nice.")
        return store

    def test_session_date_resolves_backwards_to_opener(self):
        from context_eval.strategies.lcm import _session_date

        s = self._store()
        assert _session_date(s, "m000002") == "2023/05/22"
        assert _session_date(s, "m000004") == "2023/06/10"
        assert _session_date(s, "m999999") == ""

    def test_render_carries_role_and_date(self):
        from context_eval.strategies.lcm import render_search_results

        out = render_search_results([("m000002", "Congratulations ...")], self._store())
        assert out == "m000002 [assistant] [2023/05/22]: Congratulations ..."


class TestSummaryDateClause:
    def test_d1_and_dn_mention_dates(self):
        from context_eval.strategies.lcm import _D1_PROMPT, _DN_PROMPT

        assert "date of every dated event" in _D1_PROMPT
        assert "dated event" in _DN_PROMPT


class TestFlatSectionedPrompt:
    def test_flat_prompt_sections_and_carry_forward(self):
        from context_eval.strategies.flat_compact import _FLAT_PROMPT

        for section in ("USER:", "EVENTS:", "FACTS:", "STATE:", "prior summary"):
            assert section in _FLAT_PROMPT
        assert "{segment}" not in _FLAT_PROMPT  # conversation-form: no placeholder


class TestConsolidationRepair:
    def test_missing_ids_and_topics_get_appended(self):
        from context_eval.strategies.lcm import _Node, _repair_consolidation

        kids = [
            _Node(depth=2, summary="A: x §m000001\nTOPICS: kites|open", covers=("m000001",)),
            _Node(depth=2, summary="B: y §m000002\nTOPICS: soot|done", covers=("m000002",)),
        ]
        parent = "Merged prose that dropped everything."
        out = _repair_consolidation(parent, kids)
        assert "§m000001" in out and "§m000002" in out
        assert "kites|open" in out and "soot|done" in out

    def test_kept_ids_untouched(self):
        from context_eval.strategies.lcm import _Node, _repair_consolidation

        kids = [_Node(depth=2, summary="A §m000001", covers=("m000001",))]
        out = _repair_consolidation("kept §m000001 fine", kids)
        assert "Refs kept" not in out
