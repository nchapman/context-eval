"""LongMemEval-S loader: flattening, timestamps, QA wiring, stratification."""

from __future__ import annotations

import json
from pathlib import Path

from context_eval.datasets.base import read_jsonl
from context_eval.datasets.longmemeval import convert
from context_eval.judge import locomo_score
from context_eval.types import Role


def _mini_raw(tmp_path: Path) -> Path:
    raw = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "Which two cities?",
            "answer": "Paris; Rome",
            "question_date": "2023/06/01 (Thu) 10:00",
            "answer_session_ids": ["s1"],
            "haystack_dates": ["2023/05/01 09:00", "2023/05/02 10:00"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "hello there"},
                    {"role": "assistant", "content": "hi"},
                ],
                [
                    {"role": "assistant", "content": "earlier reply"},
                    {"role": "user", "content": "and then?"},
                ],
            ],
        },
        {
            "question_id": "q2",
            "question_type": "temporal-reasoning",
            "question": "Which came first?",
            "answer": "the move",
            "question_date": "2023/06/01 (Thu) 10:00",
            "answer_session_ids": ["s2"],
            "haystack_dates": ["2023/05/01 09:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "we moved"}],
            ],
        },
    ]
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(raw))
    return p


class TestConvert:
    def test_flatten_timestamps_and_qa(self, tmp_path):
        out = tmp_path / "out.jsonl"
        stats = convert(_mini_raw(tmp_path), out)
        assert stats["questions"] == 2
        samples = read_jsonl(out)
        s1 = next(s for s in samples if s.sample_id == "q1")
        roles = [m.role for m in s1.messages]
        assert roles == [
            Role.USER, Role.ASSISTANT, Role.ASSISTANT, Role.USER,
        ]
        # first user turn of each session carries the session timestamp
        assert s1.messages[0].content.startswith("[2023/05/01 09:00] hello")
        assert s1.messages[3].content.startswith("[2023/05/02 10:00] and then?")
        # assistant-first session stamps the FIRST user turn, not the assistant
        assert not s1.messages[2].content.startswith("[")
        # qa wired for the locomo kind path, original type preserved
        qa = s1.meta["qa"][0]
        assert qa["category_name"] == "multi-session"
        assert qa["answer"] == "Paris; Rome"
        assert s1.meta["question_date"] == "2023/06/01 (Thu) 10:00"

    def test_stratified_order_balances_types(self, tmp_path):
        out = tmp_path / "out.jsonl"
        convert(_mini_raw(tmp_path), out)
        samples = read_jsonl(out)
        # round-robin: no two adjacent samples share a type when >1 type exists
        cats = [s.meta["qa"][0]["category_name"] for s in samples]
        assert len(set(cats[:2])) == 2  # balanced head for --samples N subsets

    def test_longmemeval_types_score_via_locomo_f1(self):
        # every type must fall into an F1 branch (none is 'adversarial')
        for t in (
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "multi-session",
            "temporal-reasoning",
            "knowledge-update",
        ):
            v = locomo_score(t, "Business Administration degree", "Business Administration")
            assert 0.0 < v <= 1.0, t
