"""Sawtooth arms: fixed schedule, interchangeable backs (the R10 experiment).

Cadence identity across modes is the experiment's load-bearing property —
every test that compares modes drives the SAME store through each.
"""

from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine
from context_eval.strategies.sawtooth import (
    BACK_TOKENS,
    SawtoothLedger,
    SawtoothLedger2,
    SawtoothLedger3,
    SawtoothLedger4,
    SawtoothNone,
    SawtoothRewrite,
    SawtoothSegments,
)
from context_eval.tokens import CachedCounter, CharEstimator
from context_eval.types import Role, Usage


class StubLLM:
    def __init__(self, texts=None):
        self.texts = list(texts or [])
        self.prompts: list[str] = []
        self.calls = 0

    def complete(self, messages, *, tools=None, max_tokens=1024,
                 temperature=0.0, reasoning=None, require_content=False):
        self.calls += 1
        self.prompts.append(messages[-1]["content"])

        class R:
            text = (
                self.texts.pop(0)
                if self.texts
                else "- misc | open | nothing new [§m0]"
            )
            tool_calls = ()
            usage = Usage(prompt_tokens=50, completion_tokens=20, llm_calls=1)

        return R()


def engine_for(n_msgs, budget, llm):
    store = TranscriptStore()
    for i in range(n_msgs):
        store.append(Role.TOOL, f"obs {i} " + "x" * 300)
    return Engine(
        store=store,
        counter=CachedCounter(CharEstimator()),
        budget=Budget(max_context_tokens=budget),
        llm=llm,
    )


def grow_drive(eng, strategy, n_msgs):
    """Append messages one at a time, sawing as the view refills — the
    deployment shape (and the only way to exercise multiple teeth)."""
    for i in range(n_msgs):
        eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
        strategy.on_turn(eng)
    return strategy


def build_all():
    return [
        SawtoothNone(),
        SawtoothRewrite(),
        SawtoothSegments(),
        SawtoothLedger(),
    ]


class TestCadenceIdentity:
    def test_same_teeth_and_boundaries_across_modes(self):
        """The back is priced at its cap, so tooth boundaries must be
        bit-identical across modes on the same transcript."""
        states = []
        for s in build_all():
            eng = engine_for(0, 20000, StubLLM())
            grow_drive(eng, s, 400)
            states.append((s._teeth, s._concluded_upto))
        assert states[0] == states[1] == states[2] == states[3]
        assert states[0][0] >= 2  # the transcript actually sawed

    def test_none_mode_makes_no_llm_calls(self):
        llm = StubLLM()
        s = grow_drive(engine_for(0, 20000, llm), SawtoothNone(), 400)
        assert llm.calls == 0
        assert s._teeth >= 2

    def test_head_stays_verbatim_after_cut(self):
        eng = engine_for(0, 20000, StubLLM())
        s = grow_drive(eng, SawtoothNone(), 400)
        view = s.assemble(eng)
        texts = [p.content for p in view]
        assert any("obs 399" in t for t in texts)  # newest message verbatim
        assert not any("obs 0\n" == t for t in texts)  # oldest was archived


class TestBackSemantics:
    def test_rewrite_folds_prior_back_into_next_tooth(self):
        llm = StubLLM(texts=["first fold", "second fold", "third fold"])
        s = SawtoothRewrite()
        eng = engine_for(0, 20000, llm)
        grow_drive(eng, s, 400)
        assert llm.calls == s._teeth
        assert any("PREVIOUS SUMMARY" in p for p in llm.prompts[1:])
        assert "first fold" in llm.prompts[1]

    def test_segments_never_re_derives(self):
        llm = StubLLM(texts=["seg A", "seg B", "seg C"])
        s = SawtoothSegments()
        eng = engine_for(0, 20000, llm)
        grow_drive(eng, s, 400)
        # every prompt is a raw chunk: no prior summary ever fed back in
        assert llm.prompts
        assert all("PREVIOUS SUMMARY" not in p for p in llm.prompts)
        assert not any("seg A" in p for p in llm.prompts[1:])

    def test_ledger_appends_and_builds_inventory(self):
        llm = StubLLM(texts=[
            "- corvid | open | flaky parser [§m1]",
            "- corvid | resolved | parser shipped [§m9]",
            "- amphora | paused | waiting on review [§m12]",
        ])
        s = SawtoothLedger()
        eng = engine_for(0, 20000, llm)
        grow_drive(eng, s, 400)
        assert len(s._records) >= 2
        inv = s._inventory()
        assert "corvid|resolved" in inv  # latest wins, mechanically
        # append-only: extraction never sees prior bullets or summaries
        assert all("PREVIOUS" not in p for p in llm.prompts)

    def test_back_respects_cap(self):
        llm = StubLLM(
            texts=[f"- t{i} | open | {'word ' * 200} [§m1]" for i in range(40)]
        )
        eng = engine_for(0, 20000, llm)
        s = grow_drive(eng, SawtoothLedger(), 4000)
        block = s._back_block(eng)
        used = eng.counter.count_text(block)
        inv = eng.counter.count_text(s._inventory())
        assert used <= BACK_TOKENS + inv + 200  # inventory + header slack


class TestToolParity:
    def test_all_modes_ship_identical_tool_set(self):
        eng = engine_for(10, 6000, StubLLM())
        sets = []
        for s in build_all():
            names = tuple(t.schema.name for t in s.tools(eng))
            sets.append(names)
        assert sets[0] == sets[1] == sets[2] == sets[3]
        assert "history_search" in sets[0]

    def test_archive_search_spans_cut_content(self):
        llm = StubLLM()
        eng = engine_for(0, 20000, llm)
        s = grow_drive(eng, SawtoothNone(), 400)
        search = next(t for t in s.tools(eng) if t.schema.name == "history_search")
        out = search.execute({"query": "obs 0"})
        assert out != "no matches" and "m00000" in out


class TestLedger2:
    def test_prompt_carries_known_topics_and_weighting(self):
        from context_eval.strategies.sawtooth import _EXTRACT_PROMPT_V2

        assert "KNOWN TOPIC" in _EXTRACT_PROMPT_V2
        assert "~5 lines per topic" in _EXTRACT_PROMPT_V2
        assert "change what the agent does next" in _EXTRACT_PROMPT_V2
        assert "{known_topics}" in _EXTRACT_PROMPT_V2

    def test_extraction_prompt_includes_existing_topic_names(self):
        llm = StubLLM(texts=["- corvid | open | x [§m1]",
                             "- corvid | resolved | y [§m2]"])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger2()
        grow_drive(eng, s, 400)
        # from the second tooth on, the prompt names prior topics
        assert any("KNOWN TOPICS so far" in p and "corvid" in p for p in llm.prompts)

    def test_resident_block_is_topic_balanced(self):
        # one prolific topic + two quiet topics, interleaved across teeth
        # (each tooth here is one extraction call)
        texts = ["- flood | open | line 0 [§m1]", "- flood | open | line 1 [§m1]",
                 "- quiet-a | open | only [§m2]", "- quiet-b | open | only [§m3]"]
        texts += [f"- flood | open | line {i} [§m1]" for i in range(2, 10)]
        llm = StubLLM(texts=texts)
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger2()
        grow_drive(eng, s, 400)
        block = s._back_block(eng)
        lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
        # round-robin: quiet topics surface in the first wave, not buried
        # under the flood topic's backlog
        assert any("quiet-a" in ln for ln in lines[:4]), lines[:4]
        assert any("quiet-b" in ln for ln in lines[:4]), lines[:4]

    def test_resident_lines_strip_ids(self):
        llm = StubLLM(texts=["- corvid | open | flaky parser [§m000014]"])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger2()
        grow_drive(eng, s, 400)
        block = s._back_block(eng)
        assert "§" not in block and "corvid" in block

    def test_cadence_identity_holds_with_ledger2(self):
        states = []
        for s in (SawtoothNone(), SawtoothRewrite(), SawtoothSegments(),
                  SawtoothLedger(), SawtoothLedger2()):
            eng = engine_for(0, 20000, StubLLM())
            grow_drive(eng, s, 400)
            states.append((s._teeth, s._concluded_upto))
        assert len(set(states)) == 1


class TestLedger3:
    def test_prompt_distinguishes_workstreams_and_caps_words(self):
        from context_eval.strategies.sawtooth import _EXTRACT_PROMPT_V3

        assert "distinct workstreams keep their own names" in _EXTRACT_PROMPT_V3
        assert "at most 15 words" in _EXTRACT_PROMPT_V3
        # v2's wording preserved for its own label's runs
        from context_eval.strategies.sawtooth import _EXTRACT_PROMPT_V2

        assert "genuinely new work" in _EXTRACT_PROMPT_V2

    def test_cadence_identity_holds_with_ledger3(self):
        states = []
        for s in (SawtoothNone(), SawtoothSegments(), SawtoothLedger(),
                  SawtoothLedger3()):
            eng = engine_for(0, 20000, StubLLM())
            grow_drive(eng, s, 400)
            states.append((s._teeth, s._concluded_upto))
        assert len(set(states)) == 1


class TestLedger4:
    def test_notes_never_move_status(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        lines = [
            _Ledger4Line(0, "- tq4 | resolved | ships as production tier [§m1]"),
            _Ledger4Line(1, "- tq4 | note | stale comment remains [§m2]"),
            _Ledger4Line(2, "- tq4 | note | another janitorial note [§m3]"),
        ]
        s = SawtoothLedger4()
        s._lines = lines
        assert "tq4|resolved" in s._inventory4()  # note cannot out-vote it

    def test_state_change_lines_move_status(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        s = SawtoothLedger4()
        s._lines = [
            _Ledger4Line(0, "- corvid | open | started parser fix [§m1]"),
            _Ledger4Line(1, "- corvid | note | investigating [§m2]"),
            _Ledger4Line(2, "- corvid | resolved | parser shipped 2.3.1 [§m3]"),
            _Ledger4Line(3, "- corvid | note | cleanup later [§m4]"),
        ]
        assert "corvid|resolved" in s._inventory4()

    def test_notes_only_topic_defaults_open(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        s = SawtoothLedger4()
        s._lines = [_Ledger4Line(0, "- amphora | note | reading papers [§m1]")]
        assert "amphora|open" in s._inventory4()

    def test_ids_and_brackets_stripped(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        ln = _Ledger4Line(0, "- tq4 | note | fused decode at 62% peak [§m1, §m2]")
        assert "[" not in ln.text and "]" not in ln.text
        assert "§" not in ln.text
        assert "62%" in ln.text

    def test_near_duplicates_suppressed_at_append(self):
        llm = StubLLM(texts=[
            "- tq4 | resolved | tq4 ships as production KV tier [§m1]",
            "- tq4 | resolved | tq4 ships as production tier KV [§m2]",
            "- kvarn | note | port started [§m3]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger4()
        grow_drive(eng, s, 400)
        assert len([ln for ln in s._lines if ln.topic == "tq4"]) == 1

    def test_resident_block_is_topic_grouped(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        s = SawtoothLedger4()
        s._lines = [
            _Ledger4Line(0, "- tq4 | resolved | ships [§m1]"),
            _Ledger4Line(1, "- kvarn | note | port started [§m2]"),
            _Ledger4Line(2, "- tq4 | note | needle 3/3 [§m3]"),
        ]
        eng = engine_for(10, 20000, StubLLM())
        block = s._back_block4(eng)
        assert "## tq4 — resolved" in block
        assert "## kvarn — open" in block
        assert block.index("## tq4") < block.index("needle 3/3")

    def test_cadence_identity_holds_with_ledger4(self):
        states = []
        for s in (SawtoothNone(), SawtoothLedger(), SawtoothLedger4()):
            eng = engine_for(0, 20000, StubLLM())
            grow_drive(eng, s, 400)
            states.append((s._teeth, s._concluded_upto))
        assert len(set(states)) == 1


class TestLedger4Regressions:
    def test_dedupe_cannot_eat_a_state_change(self):
        # The seeded currency flips are this shape: a note naming the
        # project, then the status flip restating it. Suppression must
        # not strand the topic in the note-default "open".
        llm = StubLLM(texts=[
            "- corvid | note | admin panel revamp nicknamed corvid [§m1]",
            "- corvid | paused | admin panel revamp corvid paused [§m2]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger4()
        grow_drive(eng, s, 400)
        assert "corvid|paused" in s._inventory4()

    def test_status_transition_with_identical_text_lands(self):
        llm = StubLLM(texts=[
            "- tq4 | resolved | tq4 ships as production tier [§m1]",
            "- tq4 | paused | tq4 ships as production tier [§m2]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger4()
        grow_drive(eng, s, 400)
        assert "tq4|paused" in s._inventory4()

    def test_identical_note_restatement_still_suppressed(self):
        llm = StubLLM(texts=[
            "- tq4 | note | tq4 ships as production tier [§m1]",
            "- tq4 | note | tq4 ships as production tier [§m2]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger4()
        grow_drive(eng, s, 400)
        assert len([ln for ln in s._lines if ln.topic == "tq4"]) == 1

    def test_unknown_status_word_demotes_to_note_keeps_line(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        ln = _Ledger4Line(0, "- tq4 | shipped | big release landed today [§m1]")
        assert ln.topic == "tq4"
        assert not ln.is_state_change
        assert "shipped" in ln.text and "big release" in ln.text

    def test_malformed_rows_dropped_not_crashed(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        for raw in ("no pipes here", "-  | note | empty topic [§m1]",
                    "- " + "x" * 80 + " | note | runaway topic [§m2]"):
            assert _Ledger4Line(0, raw).topic == ""

    def test_three_plus_id_brackets_cleanup(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        ln = _Ledger4Line(0, "- tq4 | note | fused decode at 62% peak [§a, §b, §c]")
        assert "[" not in ln.text and "]" not in ln.text and "," not in ln.text

    def test_casefold_duplicate_topics_merge_one_group(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        s = SawtoothLedger4()
        s._lines = [
            _Ledger4Line(0, "- kvarn | note | port started [§m1]"),
            _Ledger4Line(1, "- KVarN | resolved | port landed [§m2]"),
        ]
        inv = s._inventory4()
        assert "kvarn|resolved" in inv and inv.count("kvarn") == 1
        eng = engine_for(10, 20000, StubLLM())
        block = s._back_block4(eng)
        assert block.count("## ") == 1  # one merged group

    def test_back_block4_respects_cap_under_pressure(self):
        from context_eval.strategies.sawtooth import (
            BACK_TOKENS,
            _Ledger4Line,
        )
        eng = engine_for(10, 20000, StubLLM())
        counter = eng.counter
        for label, lines in [
            ("many topics", [_Ledger4Line(i, f"- topic{i:03d} | note | fact {i} [§m{i}]")
                             for i in range(200)]),
            ("few topics many lines", [_Ledger4Line(i, f"- t{i // 100} | note | fact {i} [§m{i}]")
                                       for i in range(600)]),
            ("tiny remaining after huge inventory",
             [_Ledger4Line(0, f"- {n} | note | v [§m1]")
              for n in (f"awideepname{i:02d}" for i in range(30))]),
        ]:
            s = SawtoothLedger4()
            s._lines = lines
            block = s._back_block4(eng)
            cost = counter.count_text(block)
            assert cost <= BACK_TOKENS + 64, (label, cost)


class TestLedger4MegaDepth:
    def test_bold_topic_names_do_not_fork(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        a = _Ledger4Line(0, "- **kvarn** | note | port started [§m1]")
        b = _Ledger4Line(1, "- kvarn | resolved | port landed [§m2]")
        assert a.topic == b.topic == "kvarn"

    def test_comma_id_list_fully_stripped(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        ln = _Ledger4Line(0, "- kvarn | note | community ports exist [§m8, 20, 24]")
        assert "[" not in ln.text and "20" not in ln.text
        assert "community ports" in ln.text

    def test_case_variant_folds_to_first_seen(self):
        llm = StubLLM(texts=[
            "- KVarN | note | port started [§m1]",
            "- kvarn | resolved | port landed [§m2]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger4()
        grow_drive(eng, s, 400)
        assert {ln.topic for ln in s._lines if ln.topic != "misc"} == {"KVarN"}

    def test_known_topics_passes_top100_ranked_by_activity(self):
        from context_eval.strategies.sawtooth import _Ledger4Line

        s = SawtoothLedger4()
        s._lines = [
            _Ledger4Line(0, f"- t{i:03d} | note | v [§m1]") for i in range(120)
        ] + [_Ledger4Line(120, "- t000 | resolved | late flip [§m2]")]
        known = s._known_topics_v4()
        names = known.split("):\n")[1].strip().split(", ")
        assert len(names) == 100
        assert names[0] == "t000"  # last activity outranks creation order

    def test_substream_never_absorbs_umbrella(self):
        llm = StubLLM(texts=[
            "- kvarn gate | note | gate script drafted [§m1]",
            "- kvarn | resolved | port landed [§m2]",
            "- kvarn | note | follow-up audit [§m3]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger4()
        grow_drive(eng, s, 400)
        topics = {ln.topic for ln in s._lines}
        assert "kvarn" in topics and "kvarn gate" in topics

    def test_single_token_names_need_exact_match(self):
        llm = StubLLM(texts=[
            "- test | note | suite green [§m1]",
            "- text | note | grammar masking [§m2]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger4()
        grow_drive(eng, s, 400)
        assert {"test", "text"} <= {ln.topic for ln in s._lines}
        assert len({t for t in (ln.topic for ln in s._lines) if t != "misc"}) == 2


class TestBackBlockHardCap:
    def test_v4_block_fits_cap_with_dominating_headers(self):
        # regression (mega grid): 139 topics x 1 short line each -- group
        # headers outgrow the per-line slack, block overshoots BACK_TOKENS,
        # trigger fires a turn late -> view overflows budget -> run dies
        from context_eval.strategies.sawtooth import (
            BACK_TOKENS,
            _Ledger4Line,
        )
        s = SawtoothLedger4()
        s._lines = []
        seq = 0
        for t in range(139):  # mega-live shape: real topic count, 15-word notes
            for k in range(3):
                s._lines.append(_Ledger4Line(
                    seq, f"- project stream number {t:03d} | in progress | "
                         f"kernel revision {k} shipped at {t} percent quality gate [§m{seq}]"))
                seq += 1
        eng = engine_for(10, 20000, StubLLM())
        cost = eng.counter.count_text(s._back_block4(eng))
        assert cost <= BACK_TOKENS, f"block {cost} > cap {BACK_TOKENS}"

    def test_v1_block_fits_cap_with_huge_inventory(self):
        from context_eval.strategies.conclusions import _Record
        from context_eval.strategies.sawtooth import BACK_TOKENS

        s = SawtoothLedger()
        s._records = [
            _Record(
                i,
                f"- a very deep project name {i:03d} |open| "
                f"shipped kernel revision {i} [§m{i}]",
            )
            for i in range(300)
        ]
        eng = engine_for(10, 20000, StubLLM())
        cost = eng.counter.count_text(s._back_block(eng))
        assert cost <= BACK_TOKENS, f"block {cost} > cap {BACK_TOKENS}"

    def test_all_mode_blocks_fit_cap(self):
        from context_eval.strategies.sawtooth import BACK_TOKENS

        for strat, filler in [
            (SawtoothSegments(), lambda i: f"chunk {i} " + "summary text " * 30),
            (SawtoothLedger2(), lambda i: f"- topic{i:03d} |open| fact {i} [§m{i}]"),
        ]:
            if strat.mode == "segments":
                strat._segment_summaries = [filler(i) for i in range(200)]
            else:
                from context_eval.strategies.conclusions import _Record
                strat._records = [
                    _Record(i, filler(i)) for i in range(200)
                ]
            eng = engine_for(10, 20000, StubLLM())
            block = strat._back_block(eng)
            cost = eng.counter.count_text(block)
            assert cost <= BACK_TOKENS, (strat.mode, cost)

    def test_rewrite_fold_capped_to_back_budget(self):
        from context_eval.strategies.sawtooth import _HEADER, BACK_TOKENS

        llm = StubLLM(texts=["all workstreams done. " * 400])
        eng = engine_for(10, 20000, llm)
        s = SawtoothRewrite()
        grow_drive(eng, s, 400)
        cost = eng.counter.count_text(s._back_text)
        assert cost <= BACK_TOKENS - eng.counter.count_text(_HEADER) - 2


class TestProfileKind:
    def test_profile_row_parses_as_ledger_line(self):
        from context_eval.strategies.sawtooth import _Ledger5Line

        raw = ("- career/research | profile | works in DL for medical "
               "image analysis, wants advanced material [§77]")
        ln = _Ledger5Line(0, raw)
        assert ln.kind == "profile"
        assert "medical image" in ln.text

    def test_profile_not_demoted_to_note(self):
        from context_eval.strategies.sawtooth import _Ledger5Line

        raw = "reading/prefs | profile | prefers fantasy over sci-fi [§12]"
        ln = _Ledger5Line(0, raw)
        assert ln.kind == "profile" and "profile:" not in ln.text
