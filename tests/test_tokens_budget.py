import pytest

from context_eval.budget import Budget, BudgetExceededError, enforce
from context_eval.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    CharEstimator,
    TokenizerCounter,
    default_counter,
)
from context_eval.types import PromptMessage, Role


class TestCharEstimator:
    def test_rounds_up_to_whole_tokens(self):
        est = CharEstimator()
        assert est.count_text("") == 1  # floor of 1: non-empty framing
        assert est.count_text("abcd") == 1
        assert est.count_text("abcde") == 2
        assert est.count_text("a" * 400) == 100

    def test_count_messages_adds_framing_overhead(self):
        est = CharEstimator()
        msgs = [PromptMessage(role=Role.USER, content="x" * 40)] * 3
        expected = 3 * (10 + MESSAGE_OVERHEAD_TOKENS)
        assert est.count_messages(msgs) == expected


class TestTokenizerCounter:
    def test_counts_exact_tokens_when_tokenizer_available(self):
        pytest.importorskip("tokenizers")
        try:
            counter = TokenizerCounter()
        except Exception:
            pytest.skip("DeepSeek tokenizer.json not fetched yet")

        assert counter.count_text("") == 0
        n = counter.count_text("The quick brown fox jumps over the lazy dog.")
        assert 5 <= n <= 20
        # Determinism
        assert counter.count_text("The quick brown fox jumps over the lazy dog.") == n

    def test_message_overhead_applied(self):
        pytest.importorskip("tokenizers")
        try:
            counter = TokenizerCounter()
        except Exception:
            pytest.skip("DeepSeek tokenizer.json not fetched yet")

        msgs = [PromptMessage(role=Role.USER, content="hello")]
        assert counter.count_messages(msgs) == counter.count_text("hello") + (
            MESSAGE_OVERHEAD_TOKENS
        )


def test_default_counter_falls_back_consistently():
    counter = default_counter()
    # Either implementation must be deterministic
    assert counter.count_text("same text") == counter.count_text("same text")


class TestBudget:
    def test_rejects_nonpositive_limits(self):
        with pytest.raises(ValueError):
            Budget(max_context_tokens=0)
        with pytest.raises(ValueError):
            Budget(max_context_tokens=100, max_output_tokens=-1)

    def test_enforce_passes_at_and_under_budget(self):
        budget = Budget(max_context_tokens=100)
        enforce(100, budget, "ok")
        enforce(99, budget, "ok")

    def test_enforce_raises_with_details(self):
        budget = Budget(max_context_tokens=100)
        with pytest.raises(BudgetExceededError) as exc:
            enforce(101, budget, "strategy-x turn 42")
        assert exc.value.tokens == 101
        assert exc.value.budget == 100
        assert "strategy-x" in str(exc.value)
