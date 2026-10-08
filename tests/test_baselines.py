from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies import FullContext, SlidingWindow
from context_eval.strategies.base import Engine
from context_eval.tokens import CharEstimator
from context_eval.types import Role


def make_engine(store, budget_tokens):
    return Engine(
        store=store,
        counter=CharEstimator(),
        budget=Budget(max_context_tokens=budget_tokens),
    )


def fill(store, contents):
    for c in contents:
        store.append(Role.USER, c)
    return store


def msg40(label: str) -> str:
    """A distinct 40-char message -> exactly 10 est. tokens (+4 overhead)."""
    return (label + "-").ljust(40, "x")


class TestFullContext:
    def test_returns_entire_archive_verbatim_in_order(self):
        store = fill(TranscriptStore(), ["alpha", "beta", "gamma"])
        view = FullContext().assemble(make_engine(store, 10_000))

        assert [m.content for m in view] == ["alpha", "beta", "gamma"]
        assert [m.role for m in view] == [Role.USER] * 3

    def test_never_drops_even_over_budget(self):
        # Overflow is the anchor's failure mode; the harness records it.
        store = fill(TranscriptStore(), ["a" * 400])
        view = FullContext().assemble(make_engine(store, 10))
        assert len(view) == 1

    def test_on_turn_and_tools_are_noops(self):
        s = FullContext()
        engine = make_engine(fill(TranscriptStore(), ["x"]), 100)
        s.on_turn(engine)
        assert s.tools(engine) == []
        assert s.stats().compactions == 0


class TestSlidingWindow:
    def test_keeps_newest_messages_within_budget(self):
        store = fill(TranscriptStore(), [msg40(f"m{i}") for i in range(4)])
        # 14 tokens per message -> budget 28 keeps exactly the 2 newest
        view = SlidingWindow().assemble(make_engine(store, 28))

        assert [m.content for m in view] == [msg40("m2"), msg40("m3")]

    def test_drops_oldest_first(self):
        store = fill(TranscriptStore(), [msg40("a"), msg40("b"), msg40("c")])
        view = SlidingWindow().assemble(make_engine(store, 42))  # exactly 3 fit
        assert [m.content for m in view] == [msg40("a"), msg40("b"), msg40("c")]

        view = SlidingWindow().assemble(make_engine(store, 14))  # exactly 1 fits
        assert [m.content for m in view] == [msg40("c")]

    def test_empty_view_when_nothing_fits(self):
        store = fill(TranscriptStore(), ["a" * 400])  # 100 tokens + overhead
        view = SlidingWindow().assemble(make_engine(store, 10))
        assert view == []

    def test_respects_budget_exactly_at_boundaries(self):
        store = fill(TranscriptStore(), ["x" * 4] * 10)  # 1 token + 4 = 5 each
        engine = make_engine(store, 12)  # 2 messages = 10, 3 = 15
        view = SlidingWindow().assemble(engine)
        assert len(view) == 2
        assert engine.counter.count_messages(view) <= 12

    def test_preserves_original_order(self):
        store = fill(TranscriptStore(), [msg40("aa"), msg40("bb"), msg40("cc"), msg40("dd")])
        view = SlidingWindow().assemble(make_engine(store, 100))
        assert [m.content for m in view] == [msg40("aa"), msg40("bb"), msg40("cc"), msg40("dd")]
