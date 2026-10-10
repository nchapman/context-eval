"""Conclusions arm: append-only store, mechanical inventory, bounded view."""

from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine
from context_eval.strategies.conclusions import ConclusionsMemory
from context_eval.tokens import CachedCounter, CharEstimator
from context_eval.types import Role, Usage


class StubLLM:
    def __init__(self, texts=None):
        self.texts = list(texts or [])
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []
        self.calls = 0

    def complete(self, messages, *, tools=None, max_tokens=1024,
                 temperature=0.0, reasoning=None, require_content=False):
        self.calls += 1
        self.prompts.append(messages[-1]["content"])
        self.kwargs.append({"reasoning": reasoning, "require_content": require_content})

        class R:
            text = self.texts.pop(0) if self.texts else "- misc | open | nothing [§m0]"
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


def drive(eng, strategy):
    for _ in range(len(eng.store.messages())):
        strategy.on_turn(eng)


class TestExtraction:
    def test_trigger_extracts_span_append_only(self):
        llm = StubLLM(texts=[
            "- corvid | open | parser fix started [§m0]\n"
            "- VELA-9 | resolved | shipped 2.3.1 [§m1]",
        ])
        s = ConclusionsMemory()
        eng = engine_for(60, 1500, llm)
        drive(eng, s)
        assert llm.calls >= 1
        assert s._concluded_upto > 0
        n0 = len(s._records)
        assert n0 >= 2
        # calibrated operating point on every extraction call
        assert all(k["reasoning"] == {"enabled": False} for k in llm.kwargs)
        assert all(k["require_content"] for k in llm.kwargs)
        # append-only: nothing new ages out, nothing re-extracts
        done = s._concluded_upto
        s.on_turn(eng)
        assert s._concluded_upto == done
        assert len(s._records) == n0

    def test_extraction_prompt_states_budget_and_provenance(self):
        llm = StubLLM()
        s = ConclusionsMemory()
        eng = engine_for(60, 1500, llm)
        drive(eng, s)
        assert "TOPIC | status" in llm.prompts[0]
        assert "~150 words" in llm.prompts[0]

    def test_inventory_latest_status_wins(self):
        llm = StubLLM(texts=[
            "- corvid | open | started [§m0]\n"
            "- corvid | resolved | done [§m1]\n"
            "- kestrel | open | new [§m2]",
        ])
        s = ConclusionsMemory()
        eng = engine_for(60, 1500, llm)
        drive(eng, s)
        inv = s._inventory()
        assert "corvid|resolved" in inv
        assert "corvid|open" not in inv
        assert "kestrel|open" in inv

    def test_assembly_keeps_inventory_and_fits_budget(self):
        llm = StubLLM(texts=[f"- topic{i} | open | fact {i} [§m{i}]" for i in range(40)])
        s = ConclusionsMemory()
        eng = engine_for(80, 1500, llm)
        drive(eng, s)
        view = s.assemble(eng)
        assert "TOPICS:" in view[0].content  # inventory always resident
        total = sum(eng.counter.count_message(m) for m in view)
        assert total <= 1500 + 60  # small slack for the header line

    def test_search_tool_finds_old_conclusions(self):
        llm = StubLLM(texts=["- corvid | open | flaky parser fix [§m0]"])
        s = ConclusionsMemory()
        eng = engine_for(60, 1500, llm)
        drive(eng, s)
        tool = s.tools(eng)[0]
        assert "corvid" in tool.execute({"query": "flaky parser"})
        assert tool.execute({"query": "zzzznothing"}) == "no matching conclusions"

    def test_stats_report_telemetry(self):
        llm = StubLLM(texts=["- a | open | b [§m0]"])
        s = ConclusionsMemory()
        eng = engine_for(60, 1500, llm)
        drive(eng, s)
        stats = s.stats()
        assert stats.compactions == llm.calls
        assert stats.empty_summaries == 0
        assert stats.overhead_usage.llm_calls == llm.calls


class TestEmptyExtraction:
    def test_blank_completion_counted_not_fatal(self):
        llm = StubLLM(texts=[""])
        s = ConclusionsMemory()
        eng = engine_for(60, 1500, llm)
        drive(eng, s)
        assert s.stats().empty_summaries >= 1
        assert s._records == []

    def test_no_per_turn_thrash(self):
        """Regression: raw-total triggers fired one extraction per turn;
        projected-view triggers fire per aging-out batch (the hang)."""
        llm = StubLLM(texts=["- t | open | x [§m0]"] * 20)
        s = ConclusionsMemory()
        eng = engine_for(80, 1500, llm)
        drive(eng, s)
        assert llm.calls <= 5  # ~4 batches for 80 msgs at this budget
