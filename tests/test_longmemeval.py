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


class _FakeLLM:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    def complete(self, wire, **kw):
        class _C:
            text = self.verdict

        self.prompts.append(wire[0]["content"])
        return _C()


class TestJudge:
    def _q(self, kind="longmemeval", category="multi-session"):
        from context_eval.questions import Question

        return Question(
            kind=kind, question="Which song?", answer="Evolution",
            category=category, meta={},
        )

    def test_official_judge_yes_no(self):
        from context_eval.judge import score_question

        llm = _FakeLLM("Yes.")
        sc = score_question(self._q(), "The song was **Evolution** from the album", llm=llm)
        assert (sc.value, sc.method) == (1.0, "longmemeval_judge")
        assert "intermediate steps" in llm.prompts[0]  # multi-session template
        llm_no = _FakeLLM("NO")
        sc = score_question(self._q(), "some other song", llm=llm_no)
        assert (sc.value, sc.method) == (0.0, "longmemeval_judge")

    def test_knowledge_update_uses_update_template(self):
        from context_eval.judge import score_question

        llm = _FakeLLM("yes")
        score_question(self._q(category="knowledge-update"), "was X, now Y", llm=llm)
        assert "updated answer" in llm.prompts[0]

    def test_temporal_uses_core_template(self):
        from context_eval.judge import score_question

        llm = _FakeLLM("yes")
        score_question(self._q(category="temporal-reasoning"), "6:45 AM", llm=llm)
        assert "intermediate steps" not in llm.prompts[0]

    def test_no_llm_falls_back_to_f1(self):
        from context_eval.judge import score_question

        sc = score_question(self._q(), "Evolution", llm=None)
        assert sc.method == "longmemeval_f1_fallback" and sc.value == 1.0

    def test_build_questions_emits_longmemeval_kind(self, tmp_path):
        from context_eval.datasets.base import read_jsonl
        from context_eval.run_eval import build_questions

        out = tmp_path / "out.jsonl"
        convert(_mini_raw(tmp_path), out)
        sample = read_jsonl(out)[0]
        _, qs = build_questions(
            sample, needles=0, seed=1, max_q=0, skip_locomo=False
        )
        assert len(qs) == 1
        assert qs[0].kind == "longmemeval"
        assert qs[0].category in ("multi-session", "temporal-reasoning")
