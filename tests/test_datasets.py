import httpx

from context_eval.datasets.base import ReplaySample, read_jsonl, write_jsonl
from context_eval.datasets.open_swe import (
    PAGE_SIZE,
    ROWS_URL,
    OpenSWELoader,
    _ScannedRow,
    bucket_for,
    estimate_tokens,
    fetch_sample,
    normalize,
    select,
)
from context_eval.types import Role


def fake_row(instance_id="repo__x-1", n_pairs=2, obs_len=100, resolved=1):
    messages = [{"role": "system", "content": "You are OpenHands agent..."}]
    messages.append({"role": "user", "content": "Fix the failing test in foo.py"})
    for i in range(n_pairs):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "let me look at the files",
                "tool_calls": [
                    {"id": f"call_{i}", "function": {"name": "execute_bash", "arguments": "{}"}}
                ],
            }
        )
        messages.append({"role": "tool", "content": "x" * obs_len})
    return {
        "instance_id": instance_id,
        "repo": "owner/repo",
        "license": "MIT",
        "language": "python",
        "trajectory_id": f"traj-{instance_id}",
        "messages": messages,
        "tools": ["..."],
        "resolved": resolved,
        "metadata": {
            "category": "bug-fix",
            "teacher_model": {"name": "MiniMax-M2.5"},
        },
    }


class TestNormalize:
    def test_maps_row_to_replay_sample(self):
        sample = normalize("open-swe-traces/v1.0/openhands", fake_row())

        assert sample.dataset == "open-swe-traces/v1.0/openhands"
        assert sample.sample_id == "repo__x-1"
        assert sample.meta["teacher_model"] == "MiniMax-M2.5"
        assert sample.meta["resolved"] == 1
        assert sample.meta["harness"] == "openhands"

    def test_drops_system_and_keeps_replay_roles(self):
        sample = normalize("d/s", fake_row(n_pairs=3))
        roles = [m.role for m in sample.messages]

        assert roles == [Role.USER] + [Role.ASSISTANT, Role.TOOL] * 3

    def test_null_content_becomes_empty_and_tool_calls_carried(self):
        sample = normalize("d/s", fake_row())
        assistant = sample.messages[1]

        assert assistant.content == ""
        assert assistant.meta["tool_calls"][0]["id"] == "call_0"
        assert assistant.meta["had_reasoning"] is True

    def test_tool_observation_kept_verbatim(self):
        sample = normalize("d/s", fake_row(obs_len=42))
        tool_msg = sample.messages[2]

        assert tool_msg.role is Role.TOOL
        assert tool_msg.content == "x" * 42
        assert tool_msg.meta == {}


class TestBucketing:
    def test_boundaries(self):
        assert bucket_for(7_999) == "<8k"
        assert bucket_for(8_000) == "8-16k"
        assert bucket_for(16_000) == "16-32k"
        assert bucket_for(31_999) == "16-32k"
        assert bucket_for(64_000) == "64-128k"
        assert bucket_for(1_000_000) == ">128k"

    def test_estimate_tokens_uses_chars(self):
        assert estimate_tokens(399) == 99
        assert estimate_tokens(400) == 100


def scanned(instance_id, chars, bucket=None):
    sample = ReplaySample(
        dataset="d/s",
        sample_id=instance_id,
        messages=(),
        meta={},
    )
    est = estimate_tokens(chars)
    return _ScannedRow(
        sample=sample,
        est_tokens=est,
        bucket=bucket or bucket_for(est),
    )


class TestSelect:
    def test_spreads_across_buckets_and_dedupes_instances(self):
        rows = [
            scanned("a", 4_000 * 4),  # <8k
            scanned("a", 20_000 * 4),  # same instance, 16-32k
            scanned("b", 40_000 * 4),  # 32-64k
            scanned("c", 40_000 * 4),  # 32-64k
        ]
        picked = select(rows, target=10)

        ids = [r.sample.sample_id for r in picked]
        assert ids.count("a") == 1  # deduped despite two buckets

    def test_deterministic_with_seed(self):
        rows = [scanned(f"i{n}", 40_000 * 4) for n in range(50)]
        first = select(rows, target=5, seed=42)
        second = select(rows, target=5, seed=42)
        other = select(rows, target=5, seed=7)

        assert [r.sample.sample_id for r in first] == [r.sample.sample_id for r in second]
        assert [r.sample.sample_id for r in first] != [r.sample.sample_id for r in other]

    def test_small_target_still_picks_one_per_bucket(self):
        rows = [scanned("a", 4_000 * 4), scanned("b", 20_000 * 4)]
        picked = select(rows, target=1)

        assert len(picked) == 2  # spread beats exact target


class TestScan:
    def test_fetch_page_builds_correct_request(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"rows": []})

        loader = OpenSWELoader(config="v1.0", http_client=http_client(handler))
        loader.fetch_page("openhands", offset=300)

        assert captured["url"].startswith(ROWS_URL)
        assert "dataset=nvidia%2FOpen-SWE-Traces" in captured["url"]
        assert "config=v1.0" in captured["url"]
        assert "split=openhands" in captured["url"]
        assert "offset=300" in captured["url"]
        assert f"length={PAGE_SIZE}" in captured["url"]

    def test_scan_paginates_until_max_rows(self):
        pages = {0: [fake_row("r1"), fake_row("r2")], 100: [fake_row("r3")]}

        def handler(request: httpx.Request) -> httpx.Response:
            offset = int(httpx.URL(request.url).params["offset"])
            wrapped = [{"row": r} for r in pages.get(offset, [])]
            return httpx.Response(200, json={"rows": wrapped})

        loader = OpenSWELoader(config="v1.0", http_client=http_client(handler))
        rows = list(loader.scan("openhands", max_rows=3))

        assert [r.sample.sample_id for r in rows] == ["r1", "r2", "r3"]

    def test_scan_stops_on_empty_page(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"rows": []})

        loader = OpenSWELoader(http_client=http_client(handler))
        assert list(loader.scan("sweagent", max_rows=50)) == []


class TestFetchSample:
    def test_uses_cache_without_http(self, tmp_path):
        out = tmp_path / "sample.jsonl"
        write_jsonl(
            [normalize("d/s", fake_row("cached-1"))],
            out,
        )

        def handler(request):
            raise AssertionError("should not hit the network when cached")

        samples = fetch_sample(out=out, http_client=http_client(handler))
        assert [s.sample_id for s in samples] == ["cached-1"]
        assert read_jsonl(out)[0].sample_id == "cached-1"

    def test_round_trip_preserves_everything(self, tmp_path):
        sample = normalize("d/s", fake_row("rt-1"))
        out = tmp_path / "rt.jsonl"
        write_jsonl([sample], out)

        loaded = read_jsonl(out)[0]
        assert loaded.dataset == sample.dataset
        assert loaded.sample_id == sample.sample_id
        assert loaded.meta == sample.meta
        assert [m.content for m in loaded.messages] == [m.content for m in sample.messages]
        assert loaded.messages[1].meta["tool_calls"] == sample.messages[1].meta["tool_calls"]
        assert loaded.messages[1].meta["had_reasoning"] is True


def http_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))
