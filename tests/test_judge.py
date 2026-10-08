from context_eval.judge import (
    LLMJudge,
    abstention_score,
    f1,
    f1_score,
    locomo_score,
    needle_score,
    score_question,
)
from context_eval.questions import Question
from context_eval.types import Usage


class TestNeedleScore:
    def test_exact_substring_case_insensitive(self):
        assert needle_score("The token is 9f3a-KM42Q.", "9f3a-km42q") == 1.0

    def test_embedding_in_longer_answer(self):
        assert needle_score("I recall it was rb-amber-304 overall", "rb-amber-304") == 1.0

    def test_miss(self):
        assert needle_score("The token is unknown to me", "9f3a-KM42Q") == 0.0


class TestF1Score:
    def test_exact_match_after_normalization(self):
        assert f1_score("Paris, France", "Paris France") == 1.0

    def test_articles_including_and_are_removed(self):
        # Authors' normalization strips a|an|the|and
        assert f1_score("Alice and Bob", "Alice Bob") == 1.0

    def test_stemming_matches_plurals(self):
        assert f1_score("the sessions", "session") == 1.0

    def test_partial_overlap(self):
        score = f1_score("Paris", "Paris France")
        assert 0.0 < score < 1.0

    def test_disjoint(self):
        assert f1_score("London", "Paris") == 0.0


class TestMultiAnswerF1:
    def test_comma_split_both_sides(self):
        assert f1("Alice, Bob", "Alice, Bob") == 1.0

    def test_and_conjunction_covers_both_gold_parts(self):
        # "and" is stripped by normalization; semicolons stay in one gold part
        assert f1("Alice and Bob", "Alice; Bob") > 0.9

    def test_partial_credit_across_gold_parts(self):
        score = f1("Alice", "Alice, Bob")
        assert 0.4 < score < 0.6


class TestAbstention:
    def test_authors_exact_phrases(self):
        assert abstention_score("There is no information available about that.") == 1.0
        assert abstention_score("That was not mentioned.") == 1.0

    def test_other_phrases_do_not_count(self):
        # Authors' check is exactly these two phrases
        assert abstention_score("That is unknown to me.") == 0.0

    def test_answering_fails_adversarial(self):
        assert abstention_score("Carol has a tabby cat named Whiskers.") == 0.0


class TestLocomoScore:
    def test_adversarial_dispatch(self):
        assert locomo_score("adversarial", "not mentioned", "x") == 1.0

    def test_multi_hop_dispatch(self):
        assert locomo_score("multi_hop", "Alice, Bob", "Alice, Bob") == 1.0

    def test_temporal_uses_first_semicolon_segment(self):
        score = locomo_score("temporal_reasoning", "7 May 2023", "7 May 2023; 8 May 2023")
        assert score == 1.0

    def test_single_hop_dispatch(self):
        assert locomo_score("single_hop", "Paris", "Paris") == 1.0


class TestScoreQuestion:
    def test_needle_dispatch(self):
        q = Question(kind="needle", question="q?", answer="AB-12", category="needle@50%")
        assert score_question(q, "it is ab-12!").value == 1.0

    def test_locomo_dispatch(self):
        q = Question(kind="locomo", question="q?", answer="a", category="adversarial")
        assert score_question(q, "not mentioned").value == 1.0

    def test_trace_qa_uses_f1(self):
        q = Question(kind="trace_qa", question="q?", answer="src/parser.rs")
        assert score_question(q, "src/parser.rs").value == 1.0
        # F1 with normalization: punctuation differences forgiven
        assert score_question(q, "src/parser.rs!").value == 1.0


class TestLLMJudge:
    def _fake(self, text):
        class Fake:
            def complete(self, messages, *, tools=None, max_tokens=1024, temperature=0.0):
                from context_eval.provider import Completion

                return Completion(
                    text=text,
                    tool_calls=(),
                    usage=Usage(prompt_tokens=1, completion_tokens=1, llm_calls=1),
                    finish_reason="stop",
                    model="fake",
                )

        return Fake()

    def test_yes_verdict(self):
        q = Question(kind="trace_qa", question="q?", answer="ref")
        assert LLMJudge(self._fake("YES")).score(q, "pred").value == 1.0

    def test_no_verdict(self):
        q = Question(kind="trace_qa", question="q?", answer="ref")
        assert LLMJudge(self._fake("NO")).score(q, "pred").value == 0.0

    def test_unparseable_falls_back_to_f1(self):
        q = Question(kind="trace_qa", question="q?", answer="ref answer")
        score = LLMJudge(self._fake("maybe??")).score(q, "ref answer")
        assert score.method == "llm_judge_f1_fallback"
        assert score.value == 1.0
