import json

from context_eval.report import (
    aggregate,
    depth_curves,
    load_runs,
    needle_depth,
    render_markdown,
    wilson_interval,
)


def make_run(strategy, sample, budget, scores, overflowed=False, depth_base=0, extra_answers=None):
    usage = {"prompt_tokens": 1000, "completion_tokens": 10, "reasoning_tokens": 0,
             "cached_tokens": 500, "cost": 0.001, "llm_calls": 1}
    answers = [
        {
            "question": f"q{i}",
            "gold": "g",
            "kind": "needle",
            "category": f"needle@{depth_base + i * 20}%",
            "prediction": "p",
            "score": s,
            "score_method": "needle_substring",
            "usage": usage,
            "tool_calls": i % 2,
        }
        for i, s in enumerate(scores)
    ]
    if extra_answers:
        answers.extend(extra_answers)
    return {
        "strategy": strategy,
        "dataset": "d",
        "sample_id": sample,
        "budget_tokens": budget,
        "turns": [
            {
                "index": i,
                "message_id": f"m{i}",
                "view_tokens": 100,
                "overflow": False,
                # head sig changes on turn 5 -> one rewrite
                "view_head_sig": "aaa" if i < 5 else "bbb",
            }
            for i in range(10)
        ],
        "answers": answers,
        "usage": {
            "prompt_tokens": 1000 * len(answers),
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "cost": 0.001 * len(answers),
            "llm_calls": len(answers),
        },
        "overhead_usage": {
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "cost": 0.002,
            "llm_calls": 2,
        },
        "compactions": 3,
        "peak_view_tokens": 900,
        "overflowed_at": 4 if overflowed else None,
        "model": "fake-model",
        "provider": "Baidu",
        "error": None,
    }


class TestWilson:
    def test_all_success_bounds_at_one(self):
        lo, hi = wilson_interval(10, 10)
        assert hi == 1.0
        assert lo > 0.7  # Wilson lower bound at p=1, n=10

    def test_zero_n(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_interval_contains_point_estimate(self):
        lo, hi = wilson_interval(3, 10)
        assert lo < 0.3 < hi
        assert 0 <= lo < hi <= 1

    def test_half_success_spans_half(self):
        lo, hi = wilson_interval(50, 100)
        assert 0.4 < lo < 0.5 < hi < 0.6


class TestLoadRuns:
    def test_dedupe_keeps_last(self, tmp_path):
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text(json.dumps(make_run("arc", "s1", 16000, [1, 1])) + "\n")
        b.write_text(json.dumps(make_run("arc", "s1", 16000, [0, 0])) + "\n")

        runs = load_runs(a, b)
        assert len(runs) == 1
        assert [x["score"] for x in runs[0]["answers"]] == [0, 0]

    def test_missing_file_skipped(self, tmp_path):
        assert load_runs(tmp_path / "nope.jsonl") == []

    def test_distinct_keys_kept(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text(
            json.dumps(make_run("arc", "s1", 16000, [1])) + "\n"
            + json.dumps(make_run("arc", "s1", 32000, [0])) + "\n"
            + json.dumps(make_run("lcm", "s1", 16000, [1])) + "\n"
        )
        assert len(load_runs(f)) == 3


class TestAggregate:
    def test_cell_math(self):
        runs = [
            make_run("lcm", "s1", 16000, [1, 1, 0]),
            make_run("lcm", "s2", 16000, [1, 0], overflowed=True),
        ]
        cells = aggregate(runs)
        assert len(cells) == 1
        c = cells[0]
        assert c.strategy == "lcm"
        assert c.runs == 2
        assert c.overflowed == 1
        assert c.questions == 5
        assert c.acc == 3 / 5
        assert c.pass_tok_per_q == 1000.0
        assert c.conv_view_tok_mean == 1000.0  # 10 turns x 100 tokens
        assert c.overhead_tok_mean == 600.0
        assert c.compactions_mean == 3.0
        # run1 tool_calls=[0,1,0]=1, run2=[0,1]=1 -> 2/5
        assert c.tool_calls_per_q == 0.4
        assert c.cost_per_q == 0.001
        assert c.overhead_cost_mean == 0.002
        assert c.cached_share == 0.5
        assert c.view_rewrite_rate == 0.1  # 1 rewrite per 10 turns, one run

    def test_error_answers_excluded_from_acc(self):
        err_answer = {
            "question": "q", "gold": "g", "kind": "needle", "category": "needle@10%",
            "prediction": "", "score": 0.0, "score_method": "error",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                       "reasoning_tokens": 0, "llm_calls": 0},
            "tool_calls": 0, "error": "boom",
        }
        runs = [make_run("arc", "s1", 16000, [1, 1], extra_answers=[err_answer])]
        c = aggregate(runs)[0]
        assert c.questions == 3
        assert c.answered == 2
        assert c.acc == 1.0

    def test_full_context_labels(self):
        runs = [
            make_run("full-context", "s1", 16000, [1]),
            make_run("full-context", "s1", 262144, [1]),
        ]
        cells = aggregate(runs)
        labels = {c.budget: c.label for c in cells}
        assert labels[16000] == "full-context (capped@16k)"
        assert labels[262144] == "full-context (uncapped)"


class TestDepthCurves:
    def test_parsing(self):
        assert needle_depth({"category": "needle@40%"}) == 0.4
        assert needle_depth({"category": "single_hop"}) is None

    def test_buckets(self):
        runs = [
            make_run("lcm", "s1", 16000, [1, 1, 0], depth_base=0),   # 0%, 20%, 40%
            make_run("arc", "s1", 16000, [0, 0], depth_base=60),     # 60%, 80%
        ]
        curves = depth_curves(runs)
        assert curves["lcm"] == [
            ("0-25%", 1.0, 2),
            ("25-50%", 0.0, 1),
            ("50-75%", 0.0, 0),
            ("75-100%", 0.0, 0),
        ]
        assert curves["arc"][2] == ("50-75%", 0.0, 1)
        assert curves["arc"][3] == ("75-100%", 0.0, 1)


class TestMarkdown:
    def test_renders_table_and_depth(self):
        runs = [make_run("lcm", "s1", 16000, [1, 0])]
        md = render_markdown(aggregate(runs), depth_curves(runs))
        assert "| lcm@16k |" in md
        assert "Needle accuracy by depth" in md
        assert "0-25%" in md
