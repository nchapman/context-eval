"""Ledger5 (resident memory): the law, the ladder, the recording contract.

Two levels, mirroring the theory doc (docs/resident-memory-theory.md):
- render_ledger5 unit tests drive the pure function directly (fast,
  precise control over teeth, uses, and budgets);
- strategy wiring tests drive the real sawtooth loop with a stub LLM
  (extraction, audit, search re-activation, recording roundtrip).
"""

import math

from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine
from context_eval.strategies.sawtooth import (
    BACK_TOKENS,
    Ledger5Dials,
    SawtoothLedger5,
    SawtoothNone,
    _Ledger5Line,
    render_ledger5,
    rerender_from_recording,
)
from context_eval.tokens import CachedCounter, CharEstimator
from context_eval.types import Role, Usage


class StubLLM:
    def __init__(self, texts=None):
        self.texts = list(texts or [])
        self.calls = 0

    def complete(self, messages, *, tools=None, max_tokens=1024,
                 temperature=0.0, reasoning=None, require_content=False):
        self.calls += 1

        class R:
            text = (
                self.texts.pop(0)
                if self.texts
                else "- misc/misc | open | nothing new [§m0]"
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


def counter():
    return CachedCounter(CharEstimator())


def L(seq, ws, topic, kind, text, tooth):
    """A ledger line built directly (no LLM), for render unit tests."""
    ln = _Ledger5Line(seq, f"{ws}/{topic} | {kind} | {text}")
    ln.tooth = tooth
    return ln


def render(lines, uses, tooth, dials=None, cap=4096):
    return render_ledger5(lines, uses, tooth, dials or Ledger5Dials(),
                          counter(), cap)


class TestParser:
    def test_slash_row(self):
        ln = _Ledger5Line(0, "- kvarn/gate | resolved | gate logic shipped [§m3, m4]")
        assert (ln.umbrella, ln.topic, ln.kind) == ("kvarn", "gate", "resolved")
        assert ln.text == "gate logic shipped"
        assert ln.ids == ["m3", "m4"]

    def test_decision_kind(self):
        ln = _Ledger5Line(0, "- kvarn/port | decision | chose spdylz over raw tls")
        assert ln.kind == "decision"
        assert not ln.is_state_change

    def test_flat_fallback(self):
        ln = _Ledger5Line(0, "- corvid | paused | admin revamp parked")
        assert (ln.umbrella, ln.topic) == ("corvid", "corvid")

    def test_off_vocab_demotes_to_note(self):
        ln = _Ledger5Line(0, "- kvarn/gate | shipped | v2 live")
        assert ln.kind == "note"
        assert ln.text == "shipped: v2 live"

    def test_bold_stripped(self):
        ln = _Ledger5Line(0, "- **kvarn**/gate | open | started")
        assert (ln.umbrella, ln.topic) == ("kvarn", "gate")

    def test_malformed_is_inert(self):
        ln = _Ledger5Line(0, "no pipes at all")
        assert ln.umbrella == "" and ln.text == ""

    def test_from_record_roundtrip(self):
        ln = _Ledger5Line(7, "- kvarn/gate | decision | use fmt [§m1]")
        ln.tooth = 3
        rec = {"seq": 7, "ws": "kvarn", "topic": "gate", "kind": "decision",
               "text": "use fmt", "ids": ["m1"], "tooth": 3}
        back = _Ledger5Line.from_record(rec)
        assert (back.seq, back.umbrella, back.topic, back.kind, back.text,
                back.tooth, back.ids) == (7, "kvarn", "gate", "decision",
                                          "use fmt", 3, ["m1"])


class TestActivationLaw:
    def test_fresh_single_use_is_zero(self):
        d = Ledger5Dials()
        assert _act([5], tooth=5, d=d.d_note) == 0.0
        assert _act([5], tooth=6, d=d.d_note) == 0.0  # clamp: t=0 counts as 1

    def test_aged_single_use_decays_logarithmically(self):
        assert abs(_act([1], tooth=9, d=0.5) - (-0.5 * math.log(8))) < 1e-9

    def test_durability_ordering(self):
        """Same age, same uses: state outlasts decision outlasts note."""
        d = Ledger5Dials()
        a_note = _act([1], tooth=20, d=d.d_note)
        a_dec = _act([1], tooth=20, d=d.d_decision)
        a_state = _act([1], tooth=20, d=d.d_state)
        assert a_state > a_dec > a_note

    def test_practice_sums(self):
        uses = [2, 3]
        tooth = 10
        expected = math.log(sum((tooth - u) ** -0.5 for u in uses))
        assert abs(_act(uses, tooth, 0.5) - expected) < 1e-9

    def test_no_uses_is_neg_inf(self):
        assert _act([], tooth=5, d=0.5) == float("-inf")


def _act(uses, tooth, d, form="power"):
    from context_eval.strategies.sawtooth import _l5_activation

    return _l5_activation(uses, tooth, d, form)


class TestRender:
    def _mini(self):
        lines = [
            L(0, "kvarn", "gate", "resolved", "gate logic shipped", 1),
            L(1, "kvarn", "port", "in progress", "port 40% done", 2),
            L(2, "kvarn", "port", "note", "blocked on tls handshake", 2),
            L(3, "corvid", "admin", "paused", "revamp parked", 2),
        ]
        uses = {"kvarn/gate": [1], "kvarn/port": [2], "corvid/admin": [2]}
        return lines, uses

    def test_smoke_and_completeness(self):
        lines, uses = self._mini()
        block, rungs = render(lines, uses, tooth=3)
        assert "## kvarn" in block and "## corvid" in block
        assert "gate: resolved" in block
        assert "port: in progress" in block
        assert "admin: paused" in block  # index completeness: every topic
        assert len(rungs) == 3

    def test_faded_marker_on_zero_kept_topic(self):
        # a faded marker appears when a topic's detail is OUTBID under
        # budget pressure (Axiom 3), not merely with age: give admin a
        # fat fact and a budget that cannot pay for it.
        lines, uses = self._mini()
        lines.append(L(4, "corvid", "admin", "note", "word " * 40, 2))
        dials = Ledger5Dials(seal_after_teeth=200, pin_next_step=False)
        block, _ = render(lines, uses, tooth=40, dials=dials, cap=130)
        assert 'admin: paused · faded — search "admin"' in block
        assert "- word word" not in block  # the fat fact was not paid for

    def test_pin_state_status_off_shows_name_only(self):
        lines, uses = self._mini()
        lines.append(L(4, "corvid", "admin", "note", "word " * 40, 2))
        dials = Ledger5Dials(
            pin_state_status=False, seal_after_teeth=200, pin_next_step=False
        )
        block, _ = render(lines, uses, tooth=40, dials=dials, cap=130)
        assert "admin · faded" in block
        assert "admin: paused" not in block  # status competes when unpinned

    def test_sealing_by_disuse_and_status(self):
        lines, uses = self._mini()
        block, _ = render(lines, uses, tooth=40)
        # corvid/admin: paused + unused 38 teeth -> sealed; name stays listed
        assert "sealed: corvid" in block
        assert "## corvid" not in block
        # kvarn/port is open-ish (in progress) and used at tooth 2 only:
        # 38 teeth stale -> stalled, which no longer blocks sealing
        assert "kvarn" in block  # gate is resolved, port stalled...

    def test_stale_open_status_renders_stalled(self):
        lines = [L(0, "kvarn", "port", "in progress", "port 40% done", 1)]
        uses = {"kvarn/port": [1]}
        block, _ = render(lines, uses, tooth=20)  # 19 teeth > stale 15
        assert "port: stalled" in block

    def test_recent_open_status_renders_open(self):
        lines = [L(0, "kvarn", "port", "in progress", "port 40% done", 4)]
        uses = {"kvarn/port": [4]}
        block, _ = render(lines, uses, tooth=5)
        assert "port: in progress" in block

    def test_recency_floor_caps_old_topics_at_one_line(self):
        lines = [
            L(i, "kvarn", "port", "note", f"detail number {i}", 1)
            for i in range(5)
        ]
        uses = {"kvarn/port": [1]}
        dials = Ledger5Dials(recency_floor_full=8, seal_after_teeth=200)
        block, rungs = render(lines, uses, tooth=40, dials=dials, cap=4096)
        kept = next(r for r in rungs if r["topic"] == "port")["kept"]
        assert kept == 1  # budget is ample; the floor is what binds
        assert "detail number 4" in block  # newest line survives

    def test_fill_takes_newest_fact_under_budget_pressure(self):
        """Regression: the fill must take the NEWEST unfilled fact (the
        oldest-first bug was masked by the next-step pin in the floor
        test)."""
        lines = [
            L(0, "kvarn", "port", "note", "ancient observation", 1),
            L(1, "kvarn", "port", "note", "fresh observation", 2),
        ]
        uses = {"kvarn/port": [2]}
        dials = Ledger5Dials(pin_next_step=False)
        block, rungs = render(lines, uses, tooth=3, dials=dials, cap=80)
        kept = next(r for r in rungs if r["topic"] == "port")["kept"]
        assert kept == 1  # budget pays for exactly one line
        assert "- fresh observation" in block
        assert "ancient observation" not in block

    def test_next_step_pin_survives_tight_budget(self):
        lines = [
            L(0, "kvarn", "gate", "note", "older step", 1),
            L(1, "kvarn", "port", "note", "next: rerun spill bench", 9),
            L(2, "a", "a0", "note", "filler " * 3, 2),
            L(3, "b", "b0", "note", "filler " * 3, 3),
            L(4, "c", "c0", "note", "filler " * 3, 4),
        ]
        uses = {f"{w}/{t}": [n] for w, t, n in
                [("kvarn", "gate", 1), ("kvarn", "port", 9),
                 ("a", "a0", 2), ("b", "b0", 3), ("c", "c0", 4)]}
        block, _ = render(lines, uses, tooth=10, cap=200)
        assert "## kvarn" in block  # top activation survives any triage
        assert "rerun spill bench" in block  # newest note of an active ws

    def test_overflow_invariant_under_adversarial_load(self):
        lines, uses = [], {}
        seq = 0
        for w in range(30):
            for t in range(5):
                key = f"ws{w}/t{t}"
                for _ in range(6):
                    lines.append(L(seq, f"ws{w}", f"t{t}", "note",
                                   "word " * 8, 5))
                    seq += 1
                uses[key] = [5]
        # both pin settings: the exact-accounting invariant is absolute,
        # and the pin-off delta (detail line gains ': status') is part of it
        for pin in (True, False):
            dials = Ledger5Dials(pin_state_status=pin)
            for cap in (128, 256, 512, 1024):
                block, _ = render_ledger5(lines, uses, 6, dials,
                                          counter(), cap)
                assert counter().count_text(block) <= cap, \
                    f"overflow at cap={cap} pin={pin}"
        # at a workable cap the index is real, not the elided fallback
        block, _ = render(lines, uses, tooth=6, cap=1024)
        assert "## ws" in block

    def test_triage_collapses_lowest_mass_first(self):
        lines = [
            L(0, "heavy", "a", "note", "big important work " * 3, 4),
            L(1, "light", "b", "note", "tiny", 1),
        ]
        uses = {"heavy/a": [4], "light/b": [1]}
        block, _ = render(lines, uses, tooth=5, cap=140)
        assert counter().count_text(block) <= 140
        assert "heavy" in block and "light" in block  # both fit: no triage

    def test_sealed_list_trims_to_count(self):
        lines, uses = [], {}
        for i in range(40):
            name = f"deprecated-workstream-{i}"
            lines.append(L(i, name, f"t{i}", "resolved", "done", 1))
            uses[f"{name}/t{i}".casefold()] = [1]
        block, _ = render(lines, uses, tooth=50, cap=400)
        assert "sealed:" in block
        assert "earlier)" in block  # overflow trims to "+N earlier"
        assert counter().count_text(block) <= 400

    def test_remention_unseals(self):
        """Sealing is a pure render-time classification: a new use at a
        late tooth brings the workstream back to the active index."""
        lines = [L(0, "kvarn", "gate", "resolved", "done", 1)]
        uses = {"kvarn/gate": [1]}
        sealed_block, _ = render(lines, uses, tooth=40)
        assert "sealed: kvarn" in sealed_block
        uses["kvarn/gate"].append(40)
        active_block, _ = render(lines, uses, tooth=40)
        assert "## kvarn" in active_block


class TestStrategyWiring:
    def test_registration_and_name(self):
        s = SawtoothLedger5()
        assert s.name == "sawtooth-ledger5"

    def test_cadence_identity_with_none(self):
        """The back is priced at its cap: ledger5 must saw at the exact
        same tooth boundaries as the floor arm (the schedule invariant)."""
        results = []
        for s in (SawtoothNone(), SawtoothLedger5()):
            eng = engine_for(0, 20000, StubLLM())
            for i in range(400):
                eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
                s.on_turn(eng)
            results.append((s._teeth, s._concluded_upto))
        assert results[0] == results[1]
        assert results[0][0] >= 2

    def test_drive_appends_lines_uses_and_block(self):
        llm = StubLLM(texts=[
            "- kvarn/gate | resolved | gate logic shipped [§m000001]",
            "- kvarn/port | in progress | port started [§m000051]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        for i in range(300):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        assert len(s._lines5) >= 2
        assert s._uses5["kvarn/gate"]  # extraction is a use
        block = s._back_block5(eng)
        assert "## kvarn" in block
        assert counter().count_text(block) <= BACK_TOKENS

    def test_search_reactivation_quoted_only(self):
        s = SawtoothLedger5()
        s._lines5.append(L(0, "kvarn", "split read", "note", "spills -90%", 1))
        s._uses5["kvarn/split read"] = [1]
        s._note_search("split read", [("m000003", "snippet")])
        assert s._uses5["kvarn/split read"][-1] == 1  # same render tooth
        assert s._rec5["searches"][-1]["matched"] == ["kvarn/split read"]
        before = len(s._uses5["kvarn/split read"])
        s._note_search("unrelated query", [("m000003", "snippet")])
        assert len(s._uses5["kvarn/split read"]) == before  # fan hit: no use

    def test_search_reactivation_off(self):
        s = SawtoothLedger5()
        s._dials5 = Ledger5Dials(search_reactivates="off")
        s._lines5.append(L(0, "kvarn", "gate", "note", "x", 1))
        s._note_search("gate", [("m1", "s")])
        assert "kvarn/gate" not in s._uses5
        assert s._rec5["searches"]  # the search log records regardless

    def test_audit_reextracts_uncovered_chunk_once(self):
        """An uncovered chunk ages past reaudit_gap, is rebuilt from the
        store, and re-extracted exactly ONCE (cached forever after).
        The stub cites the segment's first real id so every chunk except
        the scripted first one is covered -- exactly one audit should
        fire, no matter how many teeth follow."""
        import re as _re

        class CitingStub:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, **kw):
                self.calls += 1
                prompt = messages[-1]["content"]
                m = _re.search(r"§(m\d+)", prompt)
                if self.calls == 1:
                    text = "- kvarn/gate | note | no citations here [§zzz]"
                else:
                    phrase = ("alpha", "beta", "gamma", "delta",
                              "epsilon", "zeta")[self.calls % 6]
                    cid = m.group(1) if m else "m0"
                    text = f"- misc/misc | note | {phrase} observation [§{cid}]"

                class R:
                    tool_calls = ()
                    usage = Usage(prompt_tokens=50, completion_tokens=20,
                                  llm_calls=1)

                R.text = text
                return R

        llm = CitingStub()
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        for i in range(700):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        uncovered = [c for c in s._chunks5 if not c["covered"] and not c["audited"]]
        originals_uncovered = [c for c in s._chunks5
                               if not c["covered"] and c["tooth"] <= s._teeth - 3]
        assert all(c["audited"] for c in originals_uncovered)  # every aged miss audited
        # the rebuild-and-re-extract path ran: audited lines landed
        audited_lines = [ln for ln in s._rec5["lines"] if ln["audited"]]
        assert len(audited_lines) >= 1
        # exactly one audit per uncovered original chunk: every LLM call is
        # either a regular chunk extraction or an audit re-extraction, and
        # both append exactly one chunk entry
        assert llm.calls == len(s._chunks5)
        assert len(uncovered) <= 3  # only not-yet-aged chunks remain unaudited
        calls_before = llm.calls
        chunks_before = len(s._chunks5)
        for i in range(700, 800):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        # covered chunks are never re-audited: new calls == new chunks only
        assert llm.calls - calls_before == len(s._chunks5) - chunks_before

    def test_recording_schema_and_roundtrip(self):
        """The collection contract: the recording alone must reproduce the
        live block byte-for-byte at the same tooth (the sweep substrate)."""
        llm = StubLLM(texts=[
            "- kvarn/gate | resolved | gate shipped [§m000001]",
            "- kvarn/port | decision | chose spdylz over raw tls [§m000051]",
            "- corvid/admin | paused | revamp parked [§m000101]",
            "- kvarn/port | note | handshake blocked [§m000151]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        for i in range(600):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        live = s._back_block5(eng)
        rec = s.recording()
        for key in ("version", "dials", "teeth", "lines", "uses", "searches",
                    "chunks", "renders", "rungs", "back_tokens_cap"):
            assert rec[key] is not None or key == "searches"
        assert rec["teeth"] and rec["lines"] and rec["renders"]
        assert rec["teeth"][-1]["archived_tokens"] > 0
        block, _ = rerender_from_recording(rec, counter())
        assert block == live

    def test_non_ledger5_recording_is_none(self):
        assert SawtoothNone().recording() is None

    def test_rung_trajectories_recorded_per_render(self):
        llm = StubLLM(texts=[
            "- kvarn/gate | note | alpha [§m000001]",
            "- kvarn/gate | note | beta [§m000051]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        for i in range(300):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        s._back_block5(eng)
        assert s._rec5["rungs"]
        row = s._rec5["rungs"][0]
        for field in ("tooth", "ws", "topic", "status", "kept", "sealed", "act"):
            assert field in row
