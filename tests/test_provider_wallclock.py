"""Regression tests for SSE wall-clock guards in the provider.

A stalled stream that emits only keepalive comments resets httpx read
timeouts indefinitely; before the guards, one wedged call could hang a
whole eval run (observed: 34 min on a single LoCoMo run).
"""

from __future__ import annotations

import httpx
import pytest

from context_eval import provider as prov
from context_eval.provider import OpenRouterClient, ProviderError


def _client(handler) -> OpenRouterClient:
    return OpenRouterClient(
        api_key="k",
        stream=True,
        provider_only=("baidu/fp8",),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_comments_only_stream_times_out_and_exhausts_retries(monkeypatch):
    calls = {"n": 0, "posts": 0}

    def fake_monotonic():
        calls["n"] += 1
        return calls["n"] * 300.0  # 300s per call: past every guard

    monkeypatch.setattr(prov.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(prov.time, "sleep", lambda _s: None)

    def handler(request):
        calls["posts"] += 1
        return httpx.Response(200, content=iter([b": OPENROUTER PROCESSING\n\n"] * 3))

    client = _client(handler)
    with pytest.raises(ProviderError):
        client.complete([{"role": "user", "content": "q"}])

    # every attempt hit the wall-clock guard (not a silent hang)
    assert calls["posts"] == 7


def test_hard_deadline_fires_even_after_data_flows(monkeypatch):
    calls = {"n": 0, "posts": 0}

    def fake_monotonic():
        calls["n"] += 1
        return calls["n"] * 700.0  # first check already past the 600s cap

    monkeypatch.setattr(prov.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(prov.time, "sleep", lambda _s: None)

    def handler(request):
        calls["posts"] += 1
        chunks = [
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
            b": OPENROUTER PROCESSING\n\n",
        ]
        return httpx.Response(200, content=iter(chunks))

    client = _client(handler)
    with pytest.raises(ProviderError):
        client.complete([{"role": "user", "content": "q"}])
    assert calls["posts"] == 7
