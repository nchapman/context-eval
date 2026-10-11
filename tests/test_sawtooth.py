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
