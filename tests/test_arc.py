from context_eval.budget import Budget
from context_eval.store import TranscriptStore
from context_eval.strategies.arc import ARCCitations, make_citation
from context_eval.strategies.base import Engine
from context_eval.tokens import CharEstimator
from context_eval.types import Role


def make_engine(store, budget_tokens):
    return Engine(
        store=store, counter=CharEstimator(), budget=Budget(max_context_tokens=budget_tokens)
    )


def build_store(n_obs=6, obs_len=200, small_obs=False):
    store = TranscriptStore()
    store.append(Role.USER, "Fix the failing test")
    for i in range(n_obs):
        store.append(Role.ASSISTANT, f"step {i}")
        content = f"observation {i}: " + "z" * obs_len
        if small_obs and i % 2 == 0:
            content = f"tiny obs {i}"
        store.append(Role.TOOL, content)
    return store


class TestCitation:
    def test_contains_head_tail_and_recall_hint(self):
        content = "A" * 400 + "NEEDLE-VALUE" + "B" * 400
        citation = make_citation("m000003", content, tokens=204)

        assert citation.startswith("[§m000003 tool output — compacted, 204 tokens total]")
        assert "head: " in citation
        assert "tail: " in citation
        assert 'use the _recall tool with id "m000003"' in citation

    def test_short_content_omits_tail(self):
        citation = make_citation("m1", "short", tokens=1)
        assert "tail:" not in citation


class TestARCAssemble:
    def test_under_budget_returns_verbatim(self):
        store = build_store(obs_len=40)
        strategy = ARCCitations()
        view = strategy.assemble(make_engine(store, 10_000))

        assert view == [m.to_prompt() for m in store.messages()]
        assert strategy.stats().compactions == 0

    def test_over_budget_cites_largest_observations_first(self):
        store = build_store(n_obs=6, obs_len=2_000)  # ~500 tokens/obs; citations ~90
        strategy = ARCCitations()
        view = strategy.assemble(make_engine(store, 1_500))

        assert strategy.stats().compactions >= 1
        cited_contents = [m.content for m in view if m.content.startswith("[§")]
        assert cited_contents  # something got compacted
        # Verbatim messages still present
        assert any(m.content == "Fix the failing test" for m in view)
        assert any(m.content == "step 3" for m in view)

    def test_small_observations_stay_verbatim(self):
        store = build_store(n_obs=6, obs_len=2_000, small_obs=True)
        strategy = ARCCitations(rho_tokens=16)
        view = strategy.assemble(make_engine(store, 1_200))

        assert any("tiny obs" in m.content for m in view)  # under rho, verbatim
        assert strategy.stats().compactions >= 1

    def test_cited_view_fits_budget(self):
        store = build_store(n_obs=8, obs_len=2_400)
        engine = make_engine(store, 2_000)
        strategy = ARCCitations()
        view = strategy.assemble(engine)

        assert engine.counter.count_messages(view) <= 2_000

    def test_citations_are_stable_and_deterministic(self):
        store = build_store(n_obs=6, obs_len=2_000)
        engine = make_engine(store, 1_500)
        strategy = ARCCitations()

        view1 = strategy.assemble(engine)
        ids1 = strategy._cited_now
        view2 = strategy.assemble(engine)

        assert view1 == view2
        assert ids1 == strategy._cited_now

    def test_needle_mid_observation_recoverable_only_via_recall(self):
        store = build_store(n_obs=4, obs_len=2_000)
        needle = "NOTE (workspace audit): the canary deployment token is ab12-CD34Q."
        # Mid-log placement: previews legitimately leak head/tail content only
        store.append(Role.TOOL, "z" * 1_200 + "\n" + needle + "\n" + "z" * 800)
        strategy = ARCCitations()
        view = strategy.assemble(make_engine(store, 1_500))

        flat = "\n".join(m.content for m in view)
        assert needle not in flat  # compacted away from the view
        assert any(needle in m.content for m in store.messages())  # archive keeps it


class TestARCRecallTool:
    def test_recall_returns_full_original_content(self):
        store = build_store(n_obs=4, obs_len=2_000)
        strategy = ARCCitations()
        strategy.assemble(make_engine(store, 1_200))
        engine = make_engine(store, 1_200)
        tool = strategy.tools(engine)[0]

        cited_id = next(iter(strategy._cited_now))
        result = tool.execute({"id": cited_id})

        assert result.startswith(f"[§{cited_id} full content]")
        original = store.require(cited_id).content
        assert original in result

    def test_recall_accepts_section_prefix(self):
        store = build_store(n_obs=2, obs_len=100)
        strategy = ARCCitations()
        engine = make_engine(store, 10_000)
        tool = strategy.tools(engine)[0]

        assert "full content" in tool.execute({"id": "§m000003"})

    def test_unknown_id_returns_error(self):
        store = build_store(n_obs=2, obs_len=100)
        strategy = ARCCitations()
        tool = strategy.tools(make_engine(store, 10_000))[0]

        assert tool.execute({"id": "m999999"}).startswith("error:")

    def test_recall_counter_tracks_calls(self):
        store = build_store(n_obs=2, obs_len=100)
        strategy = ARCCitations()
        tool = strategy.tools(make_engine(store, 10_000))[0]

        tool.execute({"id": "m000003"})
        tool.execute({"id": "m000005"})

        assert strategy.stats().recall_tool_calls == 2

    def test_schema_shape(self):
        store = build_store(n_obs=1, obs_len=10)
        strategy = ARCCitations()
        schema = strategy.tools(make_engine(store, 10_000))[0].schema

        wire = schema.to_wire()
        assert wire["function"]["name"] == "_recall"
        assert wire["function"]["parameters"]["required"] == ["id"]
