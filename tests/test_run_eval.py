from context_eval.datasets.base import ReplayMessage, ReplaySample, read_jsonl
from context_eval.run_eval import build_questions, parse_args
from context_eval.types import Role


def locomo_like():
    return ReplaySample(
        dataset="locomo",
        sample_id="conv-1",
        messages=tuple(
            ReplayMessage(role=Role.USER, content=f"Alice: turn {i}", meta={"speaker": "Alice"})
            for i in range(6)
        ),
        meta={
            "qa": [
                {
                    "question": "What is Alice's cat named?",
                    "answer": "Whiskers",
                    "category_name": "single_hop",
                    "category": 2,
                },
                {
                    "question": "Where did Bob move?",
                    "answer": "no information available",
                    "category_name": "adversarial",
                    "category": 5,
                },
            ]
        },
    )


def swe_like():
    return ReplaySample(
        dataset="open-swe-traces/v1.0/openhands",
        sample_id="r-1",
        messages=tuple(
            ReplayMessage(role=Role.TOOL, content=f"obs {i} " + "x" * 100) for i in range(6)
        ),
        meta={},
    )


class TestBuildQuestions:
    def test_needles_plus_locomo_qa(self):
        injected, questions = build_questions(
            locomo_like(), needles=2, seed=5, max_q=0, skip_locomo=False
        )

        kinds = [q.kind for q in questions]
        assert kinds.count("needle") == 2
        assert kinds.count("locomo") == 2
        locomo_q = next(q for q in questions if q.kind == "locomo")
        assert locomo_q.category == "single_hop"
        assert locomo_q.answer == "Whiskers"

    def test_skip_locomo_flag(self):
        _, questions = build_questions(locomo_like(), needles=3, seed=5, max_q=0, skip_locomo=True)
        assert all(q.kind == "needle" for q in questions)

    def test_max_q_caps_with_seed(self):
        _, q1 = build_questions(locomo_like(), needles=2, seed=9, max_q=2, skip_locomo=False)
        _, q2 = build_questions(locomo_like(), needles=2, seed=9, max_q=2, skip_locomo=False)

        assert len(q1) == 2
        assert [q.question for q in q1] == [q.question for q in q2]

    def test_swe_sample_gets_needles_only(self):
        injected, questions = build_questions(
            swe_like(), needles=4, seed=1, max_q=0, skip_locomo=False
        )
        assert all(q.kind == "needle" for q in questions)
        assert len(questions) == 4
        # SWE needles ride inside existing tool observations
        assert sum(1 for m in injected.messages if m.meta.get("needle")) == 4


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.chdir("/tmp")  # argparse shouldn't touch cwd anyway
        args = parse_args([])
        assert args.strategies == "full-context,sliding-window"
        assert args.budgets == "16000,32000"
        assert args.samples == 4
        assert args.needles == 4
        assert args.max_questions == 0
        assert not args.skip_locomo_qa


def test_question_cap_preserves_all_needles():
    """Cap samples QA; every needle survives (depth curves depend on them)."""
    msgs = tuple(ReplayMessage(role=Role.USER, content=f"m{i}", meta={}) for i in range(20))
    qa = [
        {"question": f"q{i}?", "answer": f"a{i}", "category": 1, "category_name": "single"}
        for i in range(150)
    ]
    sample = ReplaySample(dataset="locomo", sample_id="s", messages=msgs, meta={"qa": qa})

    _, questions = build_questions(sample, needles=4, seed=3, max_q=16, skip_locomo=False)
    assert len(questions) == 16
    kinds = [q.kind for q in questions]
    assert kinds.count("needle") == 4  # all needles kept
    assert kinds.count("locomo") == 12  # QA capped, seeded subset

def test_rerun_truncates_output(tmp_path, monkeypatch):
    """Output files replace, never accumulate: append mode once mixed a
    killed uncapped leg with a relaunched capped one. Behavioral -- main()
    is driven end to end with the network and replay stubbed."""
    import json

    from context_eval import run_eval as re_mod
    from context_eval.driver import RunResult

    sample = tmp_path / "s.jsonl"
    sample.write_text(json.dumps({
        "dataset": "swe", "sample_id": "x1",
        "messages": [{"role": "user", "content": "hi", "meta": {}}],
        "meta": {},
    }) + "\n")

    class FakeClient:
        pass

    def fake_run_sample(**kwargs):
        return RunResult(strategy=kwargs["strategy"].name, dataset="swe",
                         sample_id="x1", budget_tokens=kwargs["budget_tokens"])

    monkeypatch.setattr(re_mod, "OpenRouterClient", lambda: FakeClient())
    monkeypatch.setattr(re_mod, "default_counter", lambda: "c")
    monkeypatch.setattr(re_mod, "run_sample", fake_run_sample)

    out = tmp_path / "runs.jsonl"
    out.write_text('{"stale": true}\n')  # pre-existing junk from a killed run
    for _ in range(2):
        rc = re_mod.main(["--path", str(sample), "--strategies", "sliding-window",
                          "--budgets", "8000", "--samples", "1", "--out", str(out)])
        assert rc == 0
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(lines) == 1  # replaced, not appended: no stale, no doubling
    assert "stale" not in lines[0]


def test_recording_sidecar_written_per_budget(tmp_path, monkeypatch):
    """ledger5's recording is the collect-once-sweep-many substrate: it
    must land as a sidecar (never in the row), named per budget so
    multi-budget runs do not clobber each other."""
    import json

    from context_eval import run_eval as re_mod
    from context_eval.driver import RunResult

    sample = tmp_path / "s.jsonl"
    sample.write_text(json.dumps({
        "dataset": "swe", "sample_id": "x1",
        "messages": [{"role": "user", "content": "hi", "meta": {}}],
        "meta": {},
    }) + "\n")

    class FakeClient:
        pass

    def fake_run_sample(**kwargs):
        result = RunResult(strategy=kwargs["strategy"].name, dataset="swe",
                           sample_id="x1", budget_tokens=kwargs["budget_tokens"])
        result.recording = {"version": 1, "budget": kwargs["budget_tokens"]}
        return result

    monkeypatch.setattr(re_mod, "OpenRouterClient", lambda: FakeClient())
    monkeypatch.setattr(re_mod, "default_counter", lambda: "c")
    monkeypatch.setattr(re_mod, "run_sample", fake_run_sample)

    out = tmp_path / "runs.jsonl"
    rc = re_mod.main(["--path", str(sample), "--strategies", "sliding-window",
                      "--budgets", "8000,16000", "--samples", "1", "--out", str(out)])
    assert rc == 0
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:  # the recording never bloats the result row
        assert "recording" not in row
    sidecars = sorted(tmp_path.glob("runs.sliding-window.*.recording.json"))
    assert len(sidecars) == 2  # one per budget, no clobbering
    budgets = {json.loads(p.read_text())["budget"] for p in sidecars}
    assert budgets == {8000, 16000}


def test_runresult_to_json_drops_recording():
    """The row contract: recording is sidecar-only."""
    import json

    from context_eval.driver import RunResult

    result = RunResult(strategy="s", dataset="d", sample_id="x",
                       budget_tokens=1000, recording={"big": [1, 2, 3]})
    parsed = json.loads(result.to_json())
    assert "recording" not in parsed
    assert parsed["strategy"] == "s"


class TestTraceQa:
    def test_build_questions_appends_trace_qa(self):
        from context_eval.run_eval import build_questions
        sample = read_jsonl("data/locomo.jsonl").__iter__().__next__()
        _, qs = build_questions(sample, needles=2, seed=0, max_q=0,
                                skip_locomo=True,
                                trace_qa=[{"question": "q?", "answer": "a"}])
        tq = [q for q in qs if q.kind == "trace_qa"]
        assert len(tq) == 1 and tq[0].answer == "a" and tq[0].category == "trace"

    def test_trace_qa_respects_max_q_pool(self):
        from context_eval.run_eval import build_questions
        sample = read_jsonl("data/locomo.jsonl").__iter__().__next__()
        _, qs = build_questions(sample, needles=2, seed=0, max_q=3,
                                skip_locomo=True,
                                trace_qa=[{"question": f"q{i}?", "answer": "a"} for i in range(5)])
        assert len([q for q in qs if q.kind == "trace_qa"]) == 1  # 2 needles + 1 pool slot
