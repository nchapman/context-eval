"""Tests for the continuity probe: seeded topic markers + uncued question."""

from __future__ import annotations

from context_eval.datasets.base import ReplayMessage, ReplaySample
from context_eval.judge import score_question
from context_eval.questions import CONTINUITY_QUESTION, Question, inject_topics
from context_eval.run_eval import build_questions
from context_eval.types import Role


def chat_sample(n_user=30):
    msgs = []
    for i in range(n_user):
        msgs.append(ReplayMessage(role=Role.USER, content=f"message number {i}", meta={}))
        msgs.append(ReplayMessage(role=Role.ASSISTANT, content=f"reply {i}", meta={}))
    return ReplaySample(dataset="pi-session", sample_id="s", messages=tuple(msgs), meta={})


class TestInjectTopics:
    def test_seeds_deterministic_disjoint_markers(self):
        import random

        s = chat_sample()
        rng1 = random.Random(7)
        rng2 = random.Random(7)
        s1, seeded1, dist1 = inject_topics(s, 8, rng1)
        s2, seeded2, dist2 = inject_topics(s, 8, rng2)
        assert seeded1 == seeded2 and dist1 == dist2  # deterministic
        assert len(seeded1) == 8
        assert not set(seeded1) & set(dist1)  # distractors never seeded
        assert len(dist1) == 4
        # markers present in the modified transcript
        joined = "\n".join(m.content for m in s1.messages)
        assert all(t in joined for t in seeded1)
        assert not any(t in joined for t in dist1)
        # markers ride user/assistant turns only, and never needle turns
        for m in s1.messages:
            if m.meta.get("needle"):
                assert not any(t in m.content for t in seeded1)

    def test_question_contains_no_searchable_terms(self):
        # The probe must be uncued: no seeded token may appear in it,
        # or the model could search for what it mentions.
        import random

        s, seeded, _ = inject_topics(chat_sample(), 8, random.Random(3))
        for t in seeded:
            assert t.lower() not in CONTINUITY_QUESTION.lower()

    def test_zero_topics_is_noop(self):
        import random

        s = chat_sample()
        s2, seeded, dist = inject_topics(s, 0, random.Random(1))
        assert seeded == [] and dist == []
        assert [m.content for m in s2.messages] == [m.content for m in s.messages]


class TestScoring:
    def q(self, terms, distractors):
        return Question(
            kind="continuity",
            question=CONTINUITY_QUESTION,
            answer="",
            category="continuity",
            meta={"terms": terms, "distractors": distractors},
        )

    def test_recall_and_false_memory_penalty(self):
        q = self.q(["amphora", "kestrel", "osprey", "waypoint"], ["bramble", "halcyon"])
        good = score_question(q, "We discussed amphora (open), kestrel (paused), osprey (beta).")
        assert good.value == 0.75  # 3 of 4, no false memories
        liar = score_question(q, "amphora, kestrel, osprey, waypoint, and bramble.")
        assert liar.value == 1.0 - 0.25  # full recall minus one false memory
        assert score_question(q, "").value == 0.0

    def test_case_insensitive(self):
        q = self.q(["ORION-3"], [])
        assert score_question(q, "We did ORION-3 auth.").value == 1.0


class TestBuildQuestions:
    def test_continuity_question_added_and_survives_cap(self):
        s = chat_sample()
        s.meta["qa"] = [
            {"question": f"q{i}?", "answer": f"a{i}", "category": 1, "category_name": "single"}
            for i in range(50)
        ]
        _, qs = build_questions(s, needles=4, seed=5, max_q=10, skip_locomo=False, topics=6)
        kinds = [q.kind for q in qs]
        assert kinds.count("continuity") == 1
        assert kinds.count("needle") == 4
        assert len(qs) == 10
        cont = next(q for q in qs if q.kind == "continuity")
        assert len(cont.meta["terms"]) == 6
        assert len(cont.meta["distractors"]) == 4

    def test_no_topics_no_continuity(self):
        _, qs = build_questions(chat_sample(), needles=4, seed=1, max_q=0, skip_locomo=True)
        assert all(q.kind != "continuity" for q in qs)


class TestNamingPlacement:
    """Naming mentions must precede the currency flip zone.

    The [status] flip (55-75% depth) defines gold; a status-bearing naming
    line landing AFTER its flip makes gold contradict the latest evidence
    (chronology conflict found in the collection runs, e.g. corvid/sundial
    in mega-a-s1). Regression: markers used to spread over the whole
    session.
    """

    def test_naming_mentions_land_in_first_half(self):
        import random

        s, seeded, _ = inject_topics(chat_sample(), 8, random.Random(3))
        assert seeded
        half = len(s.messages) / 2
        idx = [
            i
            for i, m in enumerate(s.messages)
            if any(t in m.content for t in seeded)
        ]
        assert idx and all(i < half for i in idx)

    def test_naming_before_flip_after_currency_wiring(self):
        _, qs = build_questions(
            chat_sample(n_user=60), needles=0, seed=11, max_q=0,
            skip_locomo=True, topics=8, probes="currency",
        )
        cur = next(q for q in qs if q.kind == "currency")
        gold = dict(p.split(":") for p in cur.answer.split(" | "))
        texts = [m.content for m in _.messages]
        for t, status in gold.items():
            naming = next(
                (i for i, c in enumerate(texts) if t in c and "[status]" not in c), None
            )
            flip = next(
                (i for i, c in enumerate(texts) if f"[status] {t}: {status}" in c), None
            )
            assert flip is not None, t
            assert naming is not None, t
            assert naming < flip, f"{t}: naming@{naming} after flip@{flip}"
