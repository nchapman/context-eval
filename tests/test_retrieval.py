"""Tests for the retrieval-only strategy (window + archive search)."""

from __future__ import annotations

import pytest

from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.base import Engine
from context_eval.strategies.retrieval import Bm25Index, RetrievalOnly
from context_eval.tokens import CharEstimator
from context_eval.types import Role


def make_engine(msgs, budget_tokens):
    store = TranscriptStore()
    for role, content in msgs:
        store.append(Role(role), content)
    return Engine(
        store=store,
        counter=CharEstimator(),
        budget=Budget(max_context_tokens=budget_tokens),
    )


@pytest.fixture
def engine():
    msgs = [
        ("user", "we pinned pytest to 8.2.1 in the CI config"),
        ("assistant", "updated pyproject accordingly"),
        ("user", "unrelated chatter about lunch plans"),
        ("assistant", "noted"),
        ("user", "the flaky test was in parser/test_hex.py"),
    ]
    eng = make_engine(msgs, budget_tokens=10_000)
    strategy = RetrievalOnly()
    for _ in range(len(eng.store.messages())):
        strategy.on_turn(eng)
    return strategy, eng


class TestRetrievalOnly:
    def test_assemble_is_a_sliding_window(self, engine):
        strategy, eng = engine
        view = strategy.assemble(eng)
        assert [m.role.value for m in view][-1] == "user"
        assert "parser/test_hex.py" in view[-1].content
        # full tail fits: everything is resident
        assert len(view) == len(eng.store.messages())

    def test_window_respects_budget(self):
        eng = make_engine(
            [("user", "a" * 200), ("assistant", "b" * 200), ("user", "c" * 200)],
            budget_tokens=80,  # only the newest message fits
        )
        strategy = RetrievalOnly()
        strategy.on_turn(eng)
        view = strategy.assemble(eng)
        assert len(view) == 1

    def test_search_finds_old_content_with_snippets(self, engine):
        strategy, eng = engine
        tools = {t.schema.name: t for t in strategy.tools(eng)}
        result = tools["history_search"].execute({"query": "flaky pytest"})
        assert "parser/test_hex.py" in result
        assert result.splitlines()[0].startswith("m")

    def test_recall_returns_full_verbatim_message(self, engine):
        strategy, eng = engine
        tools = {t.schema.name: t for t in strategy.tools(eng)}
        hit = tools["history_search"].execute({"query": "CI config pinned"})
        mid = hit.splitlines()[0].split()[0]  # "m000003 [user]: ..."
        full = tools["history_recall"].execute({"id": mid})
        assert "pytest to 8.2.1" in full

    def test_recall_unknown_id_is_not_an_error(self, engine):
        strategy, eng = engine
        tools = {t.schema.name: t for t in strategy.tools(eng)}
        assert tools["history_recall"].execute({"id": "m999999"}) == "no such message id"

    def test_stats_count_tool_calls(self, engine):
        strategy, eng = engine
        tools = {t.schema.name: t for t in strategy.tools(eng)}
        tools["history_search"].execute({"query": "pytest"})
        tools["history_recall"].execute({"id": "m000000"})
        stats = strategy.stats()
        assert stats.recall_tool_calls == 2

    def test_registered(self):
        from context_eval.strategies import REGISTRY, build

        assert "retrieval" in REGISTRY
        assert build("retrieval").name == "retrieval"


class TestRanking:
    def test_identifier_aware_tokenize(self):
        from context_eval.strategies.retrieval import tokenize

        assert tokenize("parseHeader") == tokenize("parse header")
        assert tokenize("parse_header") == tokenize("parse header")
        assert tokenize("HTTPServer") == ["http", "server"]
        assert tokenize("sha256sum") == ["sha", "256", "sum"]

    def test_chunk_line_aligned(self):
        from context_eval.strategies.retrieval import CHUNK_CHARS, chunk_text

        lines = "".join(f"line {i} {'x' * 50}\n" for i in range(200))
        chunks = chunk_text(lines)
        assert all(len(c) <= CHUNK_CHARS for c in chunks)
        assert all(c.endswith("\n") or c is chunks[-1] for c in chunks)
        assert "".join(chunks) == lines

    def test_decay_ranks_recent_same_term_over_old(self):
        index = Bm25Index()
        index.add("m000001", "the timeout constant was set to 30 seconds early on")
        for i in range(2, 400):
            index.add(f"m{i:06d}", f"filler number {i} unrelated content")
        index.add("m000400", "updated the timeout constant to 60 seconds")
        hits = index.search("timeout constant", limit=2)
        assert hits[0][0] == "m000400"

    def test_floor_keeps_old_distinctive_content_alive(self):
        # old message holds a rare term; recent messages lack it entirely
        index = Bm25Index()
        index.add("m000001", "the zqxkv mutex guards the ring buffer")
        for i in range(2, 3000):
            index.add(f"m{i:06d}", "ordinary work chatter")
        hits = index.search("zqxkv", limit=1)
        assert hits and hits[0][0] == "m000001"

    def test_long_message_chunks_map_back_to_one_id(self):
        index = Bm25Index()
        index.add("m000001", "\n".join(f"line {i}" for i in range(500)))
        index.add("m000002", "short message")
        hits = index.search("line 480", limit=1)
        assert hits[0][0] == "m000001"
        assert "line 480" in hits[0][1]


class TestRankingOrder:
    def corpus(self):
        index = Bm25Index()
        index.add("m000001", "we pinned pytest to 8.2.1 in the CI config")
        index.add("m000002", "unrelated chatter about lunch plans")
        index.add("m000003", "the flaky test was in parser/test_hex.py")
        index.add("m000004", "reverted the pytest pin back to 8.0")
        return index

    def test_top_hit_per_query(self):
        for query, expected in (
            ("flaky test parser", "m000003"),
            ("lunch plans", "m000002"),
            ("pytest pin reverted", "m000004"),
        ):
            assert self.corpus().search(query, limit=1)[0][0] == expected

    def test_no_match_returns_empty(self):
        assert self.corpus().search("zzzz nonexistent terms", limit=1) == []
