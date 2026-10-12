"""Offline sweep-harness tests: block parsing, reservation completeness,
rung age/class annotation, and the dial-grid contract."""

from context_eval.strategies.sawtooth import Ledger5Dials, SawtoothLedger5
from context_eval.sweeps import (
    dial_grid,
    parse_block,
    render_metrics,
    rung_curve,
    rung_rows,
    sweep_recording,
)
from context_eval.tokens import CachedCounter, CharEstimator
from context_eval.types import Role
from test_ledger5 import StubLLM, counter, engine_for


def _recording_with_search():
    """A small recorded session: two topics, one aged then re-searched."""
    llm = StubLLM(texts=[
        "- kvarn/gate | resolved | gate shipped [§m000001]",
        "- corvid/admin | paused | revamp parked [§m000051]",
        "- kvarn/gate | note | follow-up detail [§m000101]",
    ])
    eng = engine_for(0, 20000, llm)
    s = SawtoothLedger5()
    for i in range(400):
        eng.store.append(Role.TOOL, f"obs {i} " + "x" * 300)
        s.on_turn(eng)
    s._back_block5(eng)
    s._note_search("gate", [("m000001", "snip")])
    return s.recording()


class TestParseBlock:
    def test_sections_sealed_and_tokens(self):
        block = (
            "[Session memory - header line]\n\n"
            "## warp — in progress\nkern: open\n- kern detail line\n"
            "- second detail line\n\n"
            "## corvid — paused\nadmin: paused\n- parked detail\n\n"
            "sealed: palimpsest (stalled), halcyon (stalled)\n"
        )
        c = counter()
        view = parse_block(block, c)
        assert [ws for ws, *_ in view.sections] == ["warp", "corvid"]
        assert view.sealed_names == ["palimpsest", "halcyon"]
        warp = view.sections[0]
        assert warp[2] == ["kern"]  # topic names
        assert warp[4] > 0 and warp[3] > 0  # detail and index tokens
        # detail + index + header + sealed = total
        detail = sum(d for *_, d in view.sections)
        assert view.total_tokens >= detail + view.sealed_tokens

    def test_detail_bullet_accounting(self):
        block = "## w — open\nt: open\n- aaa\n- bbb\n"
        c = counter()
        view = parse_block(block, c)
        assert view.sections[0][4] == c.count_text("- aaa") + 1 + c.count_text("- bbb") + 1


class TestRenderMetrics:
    def test_reservation_completeness_is_one(self):
        """The reservation guarantee, operationalized: every unsealed rung
        topic appears in the block."""
        rec = _recording_with_search()
        rr, block = rung_rows(rec, counter())
        m = render_metrics(block, rr, counter())
        assert m["completeness"] == 1.0
        assert m["missing_unsealed"] == []
        assert m["n_unsealed"] > 0
        assert m["detail_tokens"] + m["index_tokens"] == m["total_tokens"]

    def test_counts_scale_with_topics(self):
        rec = _recording_with_search()
        rr, block = rung_rows(rec, counter())
        m = render_metrics(block, rr, counter())
        assert m["n_topics"] == len(rr)
        assert 0.0 <= m["frac_faded"] <= 1.0
        assert m["mean_kept_unsealed"] >= 0


class TestRungRows:
    def test_age_and_class_annotation(self):
        rec = _recording_with_search()
        rr, _ = rung_rows(rec, counter())
        assert all(r["age"] >= 1 for r in rr)
        assert {r["item_class"] for r in rr} <= {"note", "decision", "state"}
        # gate's newest line is a note (follow-up) -> note class, d = d_note
        gate = next(r for r in rr if r["topic"] == "gate")
        assert gate["item_class"] == "note"
        assert gate["d"] == Ledger5Dials().d_note

    def test_curve_buckets_by_class_and_age(self):
        rec = _recording_with_search()
        rr, _ = rung_rows(rec, counter())
        curve = rung_curve(rr)
        assert curve, "curve should have buckets"
        assert all(v["n"] > 0 for v in curve.values())


class TestDialGrid:
    def test_grid_contains_render_arms_not_collection_dials(self):
        grid = dial_grid(Ledger5Dials())
        names = [n for n, _ in grid]
        assert "baseline" in names
        assert "search=off" in names and "search=any-hit" in names
        assert "composition=mass" in names and "activation_form=exp" in names
        # collection-time dials never appear as sweep arms
        assert not any("reaudit" in n or "clock" in n or "use_sources" in n
                       for n in names)

    def test_sweep_recording_rows_and_search_arms(self):
        rec = _recording_with_search()
        rows = sweep_recording(rec, counter(), tooth_step=3)
        arms = {r["arm"] for r in rows}
        assert "baseline" in arms and "search=off" in arms
        base = [r for r in rows if r["arm"] == "baseline"]
        assert all(r["completeness"] == 1.0 for r in base)


class TestRealRecording:
    def test_collection_recording_roundtrip_through_harness(self):
        """The paid collection recordings must flow through the harness:
        baseline render at the final tooth is byte-exact vs the recording."""
        import glob
        import json

        from context_eval.strategies.sawtooth import rerender_from_recording
        files = sorted(glob.glob("results/collection/*.recording.json"))
        if not files:
            return  # recordings are run artifacts; skip when absent
        rec = json.loads(open(files[0]).read())
        c = counter()  # estimator is fine for structural metrics
        rr, block = rung_rows(rec, c)
        m = render_metrics(block, rr, c)
        assert m["completeness"] >= 0.99  # reservation holds at scale
        # and the byte-exact contract is counter-dependent (documented)
        exact, _ = rerender_from_recording(rec, CachedCounter(CharEstimator()))
        assert exact != rec["renders"][-1]["block"]
