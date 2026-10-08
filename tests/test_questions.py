from context_eval.datasets.base import ReplayMessage, ReplaySample
from context_eval.questions import inject_needles
from context_eval.types import Role


def swe_sample(n_obs=8):
    messages = [ReplayMessage(role=Role.USER, content="Fix the failing test")]
    for i in range(n_obs):
        messages.append(
            ReplayMessage(
                role=Role.ASSISTANT,
                content="",
                meta={"tool_calls": [{"id": f"c{i}", "function": {"name": "bash"}}]},
            )
        )
        messages.append(ReplayMessage(role=Role.TOOL, content=f"output {i}\n" + "x" * 50))
    return ReplaySample(
        dataset="open-swe-traces/v1.0/openhands",
        sample_id="t-1",
        messages=tuple(messages),
    )


def locomo_sample(n_turns=8):
    messages = [
        ReplayMessage(
            role=Role.USER,
            content=f"{'Alice' if i % 2 == 0 else 'Mel'}: turn {i}",
            meta={"speaker": "Alice" if i % 2 == 0 else "Mel"},
        )
        for i in range(n_turns)
    ]
    return ReplaySample(dataset="locomo", sample_id="conv-1", messages=tuple(messages))


class TestSweInjection:
    def test_appends_audit_lines_to_tool_observations(self):
        sample, questions = inject_needles(swe_sample(), n=3, seed=42)

        assert len(questions) == 3
        tool_msgs = [m for m in sample.messages if m.role is Role.TOOL]
        injected = [m for m in tool_msgs if m.meta.get("needle")]
        assert len(injected) == 3
        # Original content preserved before the appended line
        for m in injected:
            assert m.content.startswith("output ")

    def test_questions_carry_exact_answers_present_in_content(self):
        sample, questions = inject_needles(swe_sample(), n=4, seed=7)
        all_content = "\n".join(m.content for m in sample.messages)

        for q in questions:
            assert q.kind == "needle"
            assert q.answer in all_content
            assert q.question.endswith("?")
            assert q.category.startswith("needle@")

    def test_deterministic_with_same_seed(self):
        s1, q1 = inject_needles(swe_sample(), n=4, seed=99)
        s2, q2 = inject_needles(swe_sample(), n=4, seed=99)
        _, q3 = inject_needles(swe_sample(), n=4, seed=100)

        assert [q.answer for q in q1] == [q.answer for q in q2]
        assert [q.answer for q in q1] != [q.answer for q in q3]

    def test_depths_spread_across_trace(self):
        sample, questions = inject_needles(swe_sample(n_obs=20), n=4, seed=0)
        depths = sorted(q.meta["depth"] for q in questions)

        assert depths[0] < 0.4
        assert depths[-1] > 0.6
        assert len(set(depths)) == 4

    def test_zero_needles_is_noop(self):
        original = swe_sample()
        sample, questions = inject_needles(original, n=0, seed=1)

        assert questions == []
        assert sample is original


class TestLocomoInjection:
    def test_inserts_speaker_turn_after_position(self):
        sample, questions = inject_needles(locomo_sample(), n=2, seed=5)

        needle_turns = [m for m in sample.messages if m.meta.get("needle")]
        assert len(needle_turns) == 2
        for m in needle_turns:
            speaker = m.content.split(":", 1)[0]
            assert speaker in ("Alice", "Mel")
        # Inserted messages come right after an existing turn with same speaker
        for i, m in enumerate(sample.messages):
            if m.meta.get("needle"):
                prev = sample.messages[i - 1]
                assert prev.meta["speaker"] == m.content.split(":", 1)[0]

    def test_answers_appear_in_inserted_turns(self):
        sample, questions = inject_needles(locomo_sample(), n=3, seed=11)
        all_content = "\n".join(m.content for m in sample.messages)

        for q in questions:
            assert q.answer in all_content

    def test_capped_by_eligible_pool(self):
        sample, questions = inject_needles(locomo_sample(n_turns=2), n=5, seed=0)
        assert len(questions) <= 2
