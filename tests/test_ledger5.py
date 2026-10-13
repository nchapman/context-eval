"""Ledger5 (resident memory): the law, the ladder, the recording contract.

Two levels, mirroring the theory doc (docs/resident-memory-theory.md):
- render_ledger5 unit tests drive the pure function directly (fast,
  precise control over teeth, uses, and budgets);
- strategy wiring tests drive the real sawtooth loop with a stub LLM
  (extraction, audit, search re-activation, recording roundtrip).
"""

import math

from context_eval.budget import Budget
from context_eval.datasets.base import ReplayMessage, ReplaySample
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
            model = "fake-model"
            provider = "fake-provider"
            finish_reason = "stop"

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
        # paused + 38 teeth disused -> sealed, but the NAME stays listed
        assert "sealed: corvid" in block
        assert "## corvid" not in block
        assert "admin:" not in block  # the topic line collapsed with it
        # an open/in-progress status BLOCKS sealing while not stale ...
        lines2 = [L(0, "kvarn", "port", "in progress", "port 40% done", 1)]
        uses2 = {"kvarn/port": [1]}
        block2, _ = render(lines2, uses2, tooth=15,
                           dials=Ledger5Dials(seal_after_teeth=10,
                                              stale_status_teeth=100))
        assert "## kvarn" in block2  # discriminating form, not substring
        assert "sealed" not in block2
        # ... and stale retirement unblocks it again
        block3, _ = render(lines2, uses2, tooth=15,
                           dials=Ledger5Dials(seal_after_teeth=10,
                                              stale_status_teeth=5))
        assert "## kvarn" not in block3
        assert "sealed: kvarn" in block3

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

    def test_recency_floor_caps_aged_topics_at_three_lines(self):
        """Aged topics get a soft cap (3), not the old hard stop at one:
        arity-N sums need several co-resident facts from one aged topic
        (m25 charity: $250+$600+$5,000 unreachable at 21 teeth)."""
        lines = [
            L(i, "kvarn", "port", "note", f"detail number {i}", 1)
            for i in range(5)
        ]
        uses = {"kvarn/port": [1]}
        dials = Ledger5Dials(recency_floor_full=8, seal_after_teeth=200)
        block, rungs = render(lines, uses, tooth=40, dials=dials, cap=4096)
        kept = next(r for r in rungs if r["topic"] == "port")["kept"]
        assert kept == 3  # budget is ample; the aged cap is what binds
        assert "detail number 4" in block  # newest lines survive

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
        """When the index reservation itself overflows, the lowest-mass
        workstream collapses into the sealed tier first, and its rung
        rows must agree with the emitted block."""
        lines, uses = [], {}
        seq = 0
        for ws, last_use in (("heavy-ws", 5), ("mid-ws", 4), ("weak-ws", 1)):
            for t in range(8):  # fat index: names long, many topics
                name = f"{ws}-long-topic-name-{t}"
                lines.append(L(seq, ws, name, "note", "word " * 6, last_use))
                seq += 1
                uses[f"{ws}/{name}".casefold()] = [last_use]
        block, rungs = render(lines, uses, tooth=6, cap=400)
        assert counter().count_text(block) <= 400
        assert "## heavy-ws" in block and "## mid-ws" in block  # survivors
        assert "sealed: weak-ws" in block  # lowest mass collapsed first
        assert all(r["sealed"] for r in rungs if r["ws"] == "weak-ws")
        assert not any(r["sealed"] for r in rungs if r["ws"] == "heavy-ws")

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

    def test_sealed_line_carries_last_status(self):
        """mega-a qual finding: a names-only sealed line made cued-status
        probes answer 'sealed' (memory-tier vocabulary) instead of the
        topic's status. The sealed tier is the workstream-level L2 rung:
        name + last status, detail surrendered."""
        lines = [
            L(0, "corvid", "admin", "paused", "revamp parked", 1),
            L(1, "amphora", "queue", "resolved", "rework shipped", 1),
        ]
        uses = {"corvid/admin": [1], "amphora/queue": [1]}
        block, _ = render(lines, uses, tooth=40)
        assert "sealed: corvid (paused)" in block
        assert "amphora (resolved)" in block
        assert "- revamp parked" not in block  # detail is still surrendered

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
            if i % 120 == 100:
                s._back_block5(eng)  # mid-session renders: multiple teeth
        live = s._back_block5(eng)
        rec = s.recording()
        for key in ("version", "dials", "teeth", "lines", "uses", "searches",
                    "chunks", "renders", "rungs", "back_tokens_cap"):
            assert key in rec
        assert rec["teeth"] and rec["lines"] and rec["renders"]
        assert rec["teeth"][-1]["archived_tokens"] > 0
        block, _ = rerender_from_recording(rec, counter())
        assert block == live
        # the SWEEP contract, stronger than the final-render roundtrip:
        # every recorded render replays byte-exact at its own tooth. A
        # render at tooth R must not contain lines born at tooth R --
        # this is the assertion that catches the birth-filter off-by-one
        # (the final render alone cannot: no line is ever born at the
        # final render's tooth).
        assert len(rec["renders"]) >= 3  # session long enough to bind
        for r in rec["renders"]:
            replay, _ = rerender_from_recording(rec, counter(), tooth=r["render_tooth"])
            assert replay == r["block"], f"render at tooth {r['render_tooth']} diverged"

    def test_roundtrip_stable_under_post_render_searches(self):
        """Regression (review): question-time searches append uses AFTER
        the last recorded render. The render's uses_upto snapshot must
        keep the roundtrip exact -- and the swept path (explicit tooth,
        all uses) must differ, proving the snapshot hides data rather
        than discarding it."""
        llm = StubLLM(texts=[
            "- kvarn/port | note | port fresh detail line [§m000001]",
            "- kvarn/old | note | old aged detail line [§m000051]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        s._dials5 = Ledger5Dials(resident_budget=220)  # bind the budget
        for i in range(300):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        recorded = s._back_block5(eng)  # the render the model answered from
        # question-time search re-activates the AGED topic, post-render
        s._note_search("old", [("m000051", "snippet")])
        rec = s.recording()
        block, _ = rerender_from_recording(rec, counter())
        assert block == recorded  # snapshot: the search use cannot leak in
        # the explicit-tooth sweep at the SAME render tooth is now also
        # exact (it prefers the render's own snapshot): a post-render
        # search did not exist at that render and must not count
        tooth = rec["renders"][-1]["render_tooth"]
        swept, _ = rerender_from_recording(rec, counter(), tooth=tooth)
        assert swept == recorded
        # ...but the snapshot hides data rather than discarding it: one
        # tooth later the search use legitimately counts (search uses may
        # stamp at the render tooth), and the re-activated aged topic
        # resurfaces
        later, _ = rerender_from_recording(rec, counter(), tooth=tooth + 1)
        assert later != recorded

    def test_sweep_excludes_lines_born_after_tooth(self):
        """Regression (pre-collection gate): _l5_index filtered USES by
        tooth but not LINES -- a tooth-40 render included lines born at
        tooth 150, and the activation fallback (f.tooth, unfiltered)
        rated them maximally hot. Sweeps must be counterfactual, not
        clairvoyant: found by exercising the sweep path against the
        recorded mega-a ledger before the collection run."""
        lines = [
            L(0, "corvid", "admin", "note", "early work detail", 2),
            L(1, "zeta", "late", "note", "line born far in the future", 150),
        ]
        uses = {"corvid/admin": [2]}
        early, _ = render(lines, uses, tooth=10)
        assert "corvid" in early
        assert "zeta" not in early  # not born yet -- must not appear
        # boundary: a line born AT the render tooth is still future --
        # extraction stamps the post-increment tooth, renders happen at
        # _teeth+1, so tooth-R lines first appear in the R+1 render
        boundary, _ = render(lines, uses, tooth=150)
        assert "zeta" not in boundary
        after, _ = render(lines, uses, tooth=151)
        assert "zeta" in after  # ...but present one tooth later

    def test_sweep_mutates_dials_deterministically(self):
        """The actual purpose of the substrate: same ledger, new dials,
        deterministic difference."""
        texts = [
            f"- ws{i // 4}/t{i % 4} | note | {'word ' * 12} [§m{i:06d}]"
            for i in range(12)
        ]
        llm = StubLLM(texts=texts)
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        for i in range(700):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        s._back_block5(eng)
        rec = s.recording()
        recorded = rec["renders"][-1]["block"]
        assert counter().count_text(recorded) > 260  # fat enough to shrink
        from dataclasses import replace
        tight = replace(Ledger5Dials(**rec["dials"]), resident_budget=260)
        block, _ = rerender_from_recording(rec, counter(), dials=tight)
        assert block != recorded  # a smaller resident budget re-renders
        assert counter().count_text(block) <= 260
        # the dials= override never mutates the recording (sweep loops
        # share one recording): a fresh default roundtrip is still exact
        assert rec["dials"]["resident_budget"] != 260
        again, _ = rerender_from_recording(rec, counter())
        assert again == recorded

    def test_sweep_search_arm_off(self):
        """search_reactivates='off' offline: the recorded search uses are
        dropped. At the search's own tooth, the off arm therefore renders
        EXACTLY the view that existed before the search -- the use never
        counted -- while the recorded (quoted) arm sees it."""
        llm = StubLLM(texts=[
            "- kvarn/old | note | old aged detail line [§m000051]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        for i in range(2400):  # ~41 teeth: the aged topic seals (>30)
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        recorded = s._back_block5(eng)  # pre-search render: topic sealed
        assert "old aged detail line" not in recorded
        s._note_search("old", [("m000051", "snippet")])  # names it
        rec = s.recording()
        assert any(u["source"] == "search" for u in rec["uses"])
        tooth = rec["renders"][-1]["render_tooth"]
        from dataclasses import replace
        off = replace(Ledger5Dials(**rec["dials"]), search_reactivates="off")
        blk, _ = rerender_from_recording(rec, counter(), tooth=tooth, dials=off)
        assert "old aged detail line" not in blk
        assert "kvarn (stalled)" in blk  # sealed tier (name + last status)
        assert blk == recorded  # the search never counted under 'off'
        # ...while the recorded quoted arm sees the use one tooth later
        # (at the search's own tooth the render's uses_upto snapshot
        # correctly excludes it -- the search postdates that render)
        quoted, _ = rerender_from_recording(rec, counter(), tooth=tooth + 1)
        assert "old aged detail line" in quoted
        assert "kvarn (stalled)" not in quoted  # unsealed by the search

    def test_sweep_search_arm_any_hit(self):
        """search_reactivates='any-hit' offline: synthesized from the
        recorded search log -- a query that does NOT name the topic but
        whose results hit the topic's line ids re-activates it."""
        llm = StubLLM(texts=[
            "- kvarn/old | note | old aged detail line [§m000051]",
        ])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        for i in range(300):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        recorded = s._back_block5(eng)
        # live quoted arm: an unrelated query names nothing -> no use
        s._note_search("unrelated words", [("m000051", "snippet")])
        rec = s.recording()
        assert not any(u["source"] == "search" for u in rec["uses"])
        assert rec["searches"][0]["matched"] == []
        from dataclasses import replace
        anyhit = replace(
            Ledger5Dials(**rec["dials"]), search_reactivates="any-hit"
        )
        blk, _ = rerender_from_recording(rec, counter(), dials=anyhit)
        # the result-id hit synthesized a use: the aged topic re-activates
        assert blk != recorded
        assert "old aged detail line" in blk

    def test_sweep_collection_time_dials_raise(self):
        """reaudit_gap / clock / use_sources are baked in by the live run;
        sweeping them offline would silently render an unchanged block
        that the sweep would misreport as 'no effect'. They raise."""
        from dataclasses import replace

        import pytest
        llm = StubLLM(texts=["- kvarn/gate | note | gate work [§m000001]"])
        eng = engine_for(0, 20000, llm)
        s = SawtoothLedger5()
        for i in range(120):
            eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
            s.on_turn(eng)
        s._back_block5(eng)
        rec = s.recording()
        for fld, val in (("reaudit_gap", 9), ("clock", "wall"),
                         ("use_sources", "search-only")):
            d = replace(Ledger5Dials(**rec["dials"]), **{fld: val})
            with pytest.raises(ValueError, match=fld):
                rerender_from_recording(rec, counter(), dials=d)


    def test_composition_detail_mass_vs_max(self):
        """composition_detail: 'max' ranks workstreams by their hottest
        topic, 'mass' by summed activation. Fixture: one four-event hot
        topic vs three two-event warm topics -- max favors the hot
        singleton, mass the summed warm set (act values: ln 4 = 1.39 for
        hot; 3 x ln(1 + 2^-0.5) = 1.63 for warm)."""
        lines = [
            L(0, "hot", "single", "note", "one very hot topic", 2),
            L(1, "warm", "a", "note", "warm topic a", 3),
            L(2, "warm", "b", "note", "warm topic b", 4),
            L(3, "warm", "c", "note", "warm topic c", 5),
        ]
        uses = {
            "hot/single": [8, 8, 8, 8],
            "warm/a": [7, 8], "warm/b": [7, 8], "warm/c": [7, 8],
        }
        by_max, _ = render(lines, uses, tooth=9, dials=Ledger5Dials(composition_detail="max"))
        by_mass, _ = render(lines, uses, tooth=9, dials=Ledger5Dials(composition_detail="mass"))
        # 'max' puts the single-hot workstream first; 'mass' the summed one
        assert by_max.index("## hot") < by_max.index("## warm")
        assert by_mass.index("## warm") < by_mass.index("## hot")

    def test_activation_form_exp_diagnostic(self):
        """activation_form='exp' (diagnostic): ln sum e^{-t/8} compresses
        the tail harder than the power law, so relative activation can
        flip. Fixture (tooth 8): aaa uses ages {1,5} -> power .369 >
        bbb's .347 (ages {2,2}), but exp .349 < bbb's .443 -- the header
        order flips between the laws."""
        lines = [
            L(0, "aaa", "topic", "note", "aaa work", 2),
            L(1, "bbb", "topic", "note", "bbb work", 3),
        ]
        uses = {"aaa/topic": [3, 7], "bbb/topic": [6, 6]}
        power, _ = render(lines, uses, tooth=8, dials=Ledger5Dials())
        exp_, _ = render(
            lines, uses, tooth=8, dials=Ledger5Dials(activation_form="exp")
        )
        assert power.index("## aaa") < power.index("## bbb")
        assert exp_.index("## bbb") < exp_.index("## aaa")

    def test_run_sample_captures_recording_end_to_end(self):
        """The capture wiring itself: run_sample with a REAL ledger5 arm
        must put the recording on the RunResult (the sidecar link)."""
        from context_eval.driver import run_sample
        from context_eval.questions import Question

        sample = ReplaySample(
            dataset="swe", sample_id="ledger5-capture",
            messages=tuple(
                ReplayMessage(role=Role.TOOL, content=f"obs {i} " + "x" * 300,
                              meta={})
                for i in range(200)
            ),
            meta={},
        )
        llm = StubLLM(texts=[
            "- kvarn/gate | open | gate work started [§m000001]",
            "the answer",
        ])
        result = run_sample(
            strategy=SawtoothLedger5(),
            sample=sample,
            questions=[Question(kind="needle", question="What?",
                                answer="the answer")],
            llm=llm,
            counter=CachedCounter(CharEstimator()),
            budget_tokens=20_000,
        )
        assert result.recording is not None
        assert result.recording["lines"]  # extraction landed and was captured
        assert result.recording["teeth"]
        assert result.recording["renders"]
        row = result.to_json()
        assert "recording" not in row[:200]  # and stayed out of the row

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


class TestExtractPrompt:
    """The v1-codename clause (bench-validated on the collection runs)
    must stay in the shipped prompt: codename mentions are their own line,
    never a note under another workstream."""

    def test_codename_exception_clause(self):
        from context_eval.strategies.sawtooth import _EXTRACT_PROMPT_V5

        assert "any surface" in _EXTRACT_PROMPT_V5
        assert "tracker ids" in _EXTRACT_PROMPT_V5
        assert "NEVER a note under another workstream" in _EXTRACT_PROMPT_V5
        assert "one line each" in _EXTRACT_PROMPT_V5
        # template slots intact
        assert "{known_topics}" in _EXTRACT_PROMPT_V5
        assert "{segment}" in _EXTRACT_PROMPT_V5
        # the guard rails survived the edit
        assert "At most 15 words" in _EXTRACT_PROMPT_V5
        assert "Never stamp a status on a note" in _EXTRACT_PROMPT_V5


class TestBulletOrRow:
    """Bare-row format drift: the model sometimes drops the leading '- '
    (captured verbatim from a bench response — 16 valid rows, zero of
    them bullets, including a seeded status line). The bullet gate alone
    silently discards the whole chunk."""

    def test_bare_rows_parse(self):
        from context_eval.strategies.sawtooth import _bullet_or_row, _Ledger5Line

        drift = (
            "warp3090/plan | note | vec decode 13.6 t/s vs 34.7 MMA [§m007802]",
            "bramble | open | investigation started [§m007736]",
        )
        for line in drift:
            row = _bullet_or_row(line)
            assert row is not None, line
            ln = _Ledger5Line(0, row)
            assert ln.umbrella and ln.topic and ln.ids

    def test_bullets_still_parse_and_prose_rejected(self):
        from context_eval.strategies.sawtooth import _bullet_or_row

        assert _bullet_or_row("- ws/t | note | fine [§m1]") is not None
        assert _bullet_or_row("* ws/t | note | fine [§m1]") is not None
        # prose without a row structure must not sneak in
        assert _bullet_or_row("here are the lines I found") is None
        assert _bullet_or_row("") is None
        # pipe-free bare text is not a row
        assert _bullet_or_row("no pipes at all") is None
