import httpx

from context_eval.datasets.base import read_jsonl
from context_eval.datasets.locomo import (
    LOCOMO_URL,
    QA_CATEGORIES,
    load_locomo,
    normalize_conversation,
)
from context_eval.types import Role


def fake_item():
    return {
        "sample_id": "conv-99",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_2": [
                {"speaker": "Bob", "dia_id": "D2:1", "text": "Second session first"},
                {"speaker": "Alice", "dia_id": "D2:2", "text": "Second session second"},
            ],
            "session_2_date_time": "10 am on 2 May, 2023",
            "session_10": [
                {"speaker": "Alice", "dia_id": "D10:1", "text": "Tenth session"},
            ],
            "session_10_date_time": "3 pm on 9 June, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "First!"},
            ],
            "session_1_date_time": "9 am on 1 May, 2023",
        },
        "qa": [
            {
                "question": "What did Alice say first?",
                "answer": "First!",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "Is anything known about Carol's cat?",
                "evidence": [],
                "category": 5,
                "adversarial_answer": "Carol has a tabby",
            },
        ],
        "event_summary": [],
    }


class TestNormalize:
    def test_sessions_sorted_numerically_not_lexically(self):
        sample = normalize_conversation(fake_item())
        sessions = [m.meta["session"] for m in sample.messages]

        assert sessions == ["session_1", "session_2", "session_2", "session_10"]

    def test_turns_are_speaker_prefixed_user_messages(self):
        sample = normalize_conversation(fake_item())
        first = sample.messages[0]

        assert first.role is Role.USER
        assert first.content == "Alice: First!"
        assert first.meta["dia_id"] == "D1:1"
        assert first.meta["session_ts"] == "9 am on 1 May, 2023"

    def test_qa_categories_resolved(self):
        sample = normalize_conversation(fake_item())
        qa = sample.meta["qa"]

        assert qa[0]["category_name"] == "single_hop"
        assert qa[1]["category_name"] == "adversarial"
        # Adversarial gold is abstention; the trap answer is carried separately
        assert qa[1]["answer"] == "no information available"
        assert qa[1]["adversarial_answer"] == "Carol has a tabby"
        assert sample.meta["n_sessions"] == 3
        assert sample.dataset == "locomo"

    def test_all_five_categories_mapped(self):
        assert set(QA_CATEGORIES) == {1, 2, 3, 4, 5}


class TestLoadLocomo:
    def test_downloads_and_caches_normalized_jsonl(self, tmp_path):
        raw = tmp_path / "locomo10.json"
        out = tmp_path / "locomo.jsonl"
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json=[fake_item()])

        client = httpx.Client(transport=httpx.MockTransport(handler))
        samples = load_locomo(out=out, raw_path=raw, http_client=client)

        assert captured["url"] == LOCOMO_URL
        assert [s.sample_id for s in samples] == ["conv-99"]
        assert raw.is_file() and out.is_file()

        def explode(request):
            raise AssertionError("cache miss hit the network")

        explode_client = httpx.Client(transport=httpx.MockTransport(explode))
        cached = load_locomo(out=out, raw_path=raw, http_client=explode_client)
        assert [s.sample_id for s in cached] == ["conv-99"]

        reread = read_jsonl(out)[0]
        assert reread.meta["qa"][0]["category_name"] == "single_hop"
