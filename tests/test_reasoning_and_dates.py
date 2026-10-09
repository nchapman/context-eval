"""Regression tests for reasoning-only responses and LoCoMo date anchors.

- deepseek-v4.1-flash sometimes emits the whole answer on the reasoning
  channel with empty content (~30% of short factual answers); without the
  fallback those scored as blank predictions.
- LoCoMo gold answers are calendar-anchored while the chat text says
  "this week"; without rendered session dates, temporal and single-hop
  categories are unanswerable in principle.
"""

from __future__ import annotations

import httpx

from context_eval.datasets.locomo import normalize_conversation as normalize
from context_eval.driver import _ask
from context_eval.provider import OpenRouterClient
from context_eval.questions import Question


def _stream_client(chunks: list[bytes]) -> OpenRouterClient:
    def handler(request):
        return httpx.Response(200, content=iter(chunks))

    return OpenRouterClient(
        api_key="k",
        stream=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_reasoning_content_captured_when_content_empty():
    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"\\nThe answer is 42."}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    result = _stream_client(chunks).complete([{"role": "user", "content": "q"}])
    assert result.text == ""
    assert result.reasoning == "thinking...\nThe answer is 42."


def test_ask_falls_back_to_last_reasoning_line():
    class _Engine:
        pass

    class _Strategy:
        def assemble(self, engine):
            return []

        def tools(self, engine):
            return []

    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"hmm, let me see"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"\\nthe version is 2.3.1"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    record = _ask(
        strategy=_Strategy(),
        engine=_Engine(),
        llm=_stream_client(chunks),
        question=Question(kind="needle", question="version?", answer="2.3.1"),
        max_answer_tokens=256,
        temperature=0.0,
        system_prompt="s",
    )
    assert record.prediction == "the version is 2.3.1"


def test_locomo_normalize_renders_session_dates():
    item = {
        "sample_id": "c1",
        "conversation": {
            "session_1_date_time": "9 June 2023, 10:00",
            "session_1": [
                {"speaker": "Caroline", "text": "I gave a speech this week", "dia_id": 1}
            ],
            "session_2_date_time": "5 July 2023, 11:00",
            "session_2": [
                {"speaker": "Melanie", "text": "I went to the museum", "dia_id": 2}
            ],
        }
    }
    sample = normalize(item)
    contents = [m.content for m in sample.messages]
    assert contents[0] == "[Session on 9 June 2023, 10:00]"
    assert contents[2] == "[Session on 5 July 2023, 11:00]"
    # dates also present in the rendered text the model actually sees
    assert sample.messages[0].meta["session_ts"].startswith("9 June 2023")
