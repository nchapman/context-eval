import json

import httpx
import pytest

from context_eval.provider import (
    OpenRouterClient,
    ProviderError,
    ToolSchema,
    assistant_tool_message,
    parse_tool_arguments,
    tool_result_message,
    wire_messages_from_prompt,
)


def ok_response(
    text="hi",
    tool_calls=None,
    prompt_tokens=10,
    completion_tokens=5,
    reasoning_tokens=0,
    model="deepseek/deepseek-chat-v3-0324",
):
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if reasoning_tokens:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "model": model,
        "provider_name": "baidu/fp8",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage,
    }


class TestParseUsage:
    def test_reads_cost_and_cached_tokens(self):

        import httpx

        from context_eval.provider import OpenRouterClient
        from context_eval.types import Usage

        body = {
            "usage": {
                "prompt_tokens": 2855,
                "completion_tokens": 16,
                "cost": 0.00000564,
                "prompt_tokens_details": {"cached_tokens": 2816},
                "completion_tokens_details": {"reasoning_tokens": 13},
            },
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }

        def handler(request):
            return httpx.Response(200, json=body)

        client = OpenRouterClient(
            api_key="k", stream=False, http_client=_client(handler)
        )
        result = client.complete([{"role": "user", "content": "q"}])

        assert result.usage.cached_tokens == 2816
        assert result.usage.cost == 0.00000564
        assert result.usage.reasoning_tokens == 13

        summed = Usage()
        summed.add(result.usage)
        summed.add(result.usage)
        assert summed.cost == 2 * 0.00000564
        assert summed.cached_tokens == 2 * 2816


class TestOpenRouterClient:
    def test_sends_openai_wire_format_with_auth(self, tmp_path):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("Authorization")
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_response())

        client = OpenRouterClient(
            model="deepseek/deepseek-chat",
            api_key="sk-test",
            stream=False,
            http_client=_client(handler),
        )
        result = client.complete(
            [{"role": "user", "content": "hello"}],
            tools=[ToolSchema(name="recall", description="r", parameters={"type": "object"})],
        )

        assert captured["auth"] == "Bearer sk-test"
        assert captured["url"].endswith("/chat/completions")
        body = captured["body"]
        assert body["model"] == "deepseek/deepseek-chat"
        assert body["temperature"] == 0.0
        assert body["tools"][0]["function"]["name"] == "recall"
        assert body["tool_choice"] == "auto"

        assert result.text == "hi"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.llm_calls == 1
        assert result.model == "deepseek/deepseek-chat-v3-0324"  # resolved model
        assert result.provider == "baidu/fp8"

    def test_parses_reasoning_token_detail_non_stream(self):
        def handler(request):
            return httpx.Response(
                200, json=ok_response(completion_tokens=20, reasoning_tokens=17)
            )

        client = OpenRouterClient(
            api_key="sk-test", stream=False, http_client=_client(handler)
        )
        result = client.complete([{"role": "user", "content": "q"}])

        assert result.usage.reasoning_tokens == 17
        assert result.usage.completion_tokens == 20

    def test_parses_tool_calls(self):
        wire_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "recall", "arguments": '{"id": "m000003"}'},
            }
        ]

        def handler(request):
            return httpx.Response(200, json=ok_response(tool_calls=wire_calls))

        client = OpenRouterClient(api_key="sk-test", stream=False, http_client=_client(handler))
        result = client.complete([{"role": "user", "content": "q"}])

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert (tc.id, tc.name, tc.arguments) == ("call_1", "recall", '{"id": "m000003"}')

    def test_missing_api_key_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = OpenRouterClient(api_key=None, base_url="http://x")
        with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
            client.complete([{"role": "user", "content": "q"}])

    def test_retries_transient_then_succeeds(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, json=ok_response())

        client = OpenRouterClient(api_key="sk-test", stream=False, http_client=_client(handler))
        result = client.complete([{"role": "user", "content": "q"}])
        assert result.text == "hi"
        assert len(attempts) == 3

    def test_non_transient_error_raises_immediately(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(401, text="bad key")

        client = OpenRouterClient(
            api_key="sk-bad",
            stream=False,
            provider_only=("baidu/fp8",),
            http_client=_client(handler),
        )
        with pytest.raises(ProviderError, match="401"):
            client.complete([{"role": "user", "content": "q"}])
        assert len(attempts) == 1

    def test_reasoning_param_passes_through_when_given(self, tmp_path):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_response())

        client = OpenRouterClient(
            model="deepseek/deepseek-chat", api_key="sk-test",
            stream=False, http_client=_client(handler),
        )
        client.complete([{"role": "user", "content": "hello"}],
                        reasoning={"enabled": False})
        assert captured["body"]["reasoning"] == {"enabled": False}
        client.complete([{"role": "user", "content": "hello"}])
        assert "reasoning" not in captured["body"]

    def test_non_stream_mode_sends_plain_body(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_response())

        client = OpenRouterClient(
            api_key="sk-test", stream=False, http_client=_client(handler)
        )
        client.complete([{"role": "user", "content": "q"}])

        assert "stream" not in captured["body"]
        assert captured["body"]["provider"] == {
            "only": ["baidu/fp8"],
            "allow_fallbacks": False,
        }

    def test_provider_error_in_body_raises(self):
        def handler(request):
            return httpx.Response(200, json={"error": "insufficient credits"})

        client = OpenRouterClient(api_key="sk-test", stream=False, http_client=_client(handler))
        with pytest.raises(ProviderError, match="insufficient credits"):
            client.complete([{"role": "user", "content": "q"}])


class TestWireHelpers:
    def test_wire_messages_prepends_system(self):
        msgs = wire_messages_from_prompt("be good", [{"role": "user", "content": "q"}])
        assert msgs[0] == {"role": "system", "content": "be good"}
        assert len(msgs) == 2

    def test_tool_result_message_shape(self):
        m = tool_result_message("call_1", "found it")
        assert m == {"role": "tool", "tool_call_id": "call_1", "content": "found it"}

    def test_assistant_tool_message_shape(self):
        from context_eval.types import ToolCall

        m = assistant_tool_message(
            [ToolCall(id="call_1", name="recall", arguments='{"id":"m1"}')]
        )
        assert m["tool_calls"][0]["function"]["name"] == "recall"
        assert m["content"] is None

    def test_parse_tool_arguments(self):
        assert parse_tool_arguments('{"a": 1}') == {"a": 1}
        assert parse_tool_arguments("") == {}
        assert parse_tool_arguments("not json") == {}
        assert parse_tool_arguments("[1,2]") == {}  # non-dict JSON


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _sse_client(chunks) -> httpx.Client:
    """A mock client serving the given chunks as one SSE response."""
    return _client(lambda request: httpx.Response(200, text=sse_text(*chunks)))


def sse_text(*chunks) -> str:
    """Serialize chunks as an SSE stream, with comments and [DONE]."""
    events = [": OPENROUTER PROCESSING"]
    events += [f"data: {json.dumps(c)}" for c in chunks]
    events.append("data: [DONE]")
    return "\n\n".join(events) + "\n\n"


class TestStreaming:
    def test_stream_happy_path_with_usage_chunk(self):
        chunks = [
            {
                "model": "deepseek/deepseek-v4.1-flash",
                "provider": "baidu/fp8",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "Hel"}}
                ],
            },
            {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 17},
                },
            },
        ]
        client = OpenRouterClient(
            api_key="sk-test", http_client=_sse_client(chunks)
        )
        result = client.complete([{"role": "user", "content": "q"}], max_tokens=64)

        assert result.text == "Hello"
        assert result.finish_reason == "stop"
        assert result.model == "deepseek/deepseek-v4.1-flash"
        assert result.provider == "baidu/fp8"
        assert result.usage.prompt_tokens == 13
        assert result.usage.reasoning_tokens == 17
        assert result.usage.llm_calls == 1

    def test_stream_accumulates_tool_call_deltas(self):
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "recall", "arguments": ""},
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"id":'}}]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ' "m3"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
        ]
        client = OpenRouterClient(
            api_key="sk-test", http_client=_sse_client(chunks)
        )
        result = client.complete([{"role": "user", "content": "q"}])

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert (tc.id, tc.name, tc.arguments) == ("call_1", "recall", '{"id": "m3"}')
        assert result.finish_reason == "tool_calls"

    def test_stream_body_carries_options_and_routing(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            empty = sse_text({"choices": []})
            return httpx.Response(200, text=empty)

        client = OpenRouterClient(api_key="sk-test", http_client=_client(handler))
        client.complete([{"role": "user", "content": "q"}])

        assert captured["body"]["stream"] is True
        assert captured["body"]["stream_options"] == {"include_usage": True}
        assert captured["body"]["model"] == "deepseek/deepseek-v4.1-flash"
        assert captured["body"]["provider"] == {
            "only": ["baidu/fp8"],
            "allow_fallbacks": False,
        }

    def test_stream_retry_on_transient_then_success(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) < 2:
                return httpx.Response(503, text="overloaded")
            done = sse_text({"choices": [{"index": 0, "delta": {"content": "ok"}}]})
            return httpx.Response(200, text=done)

        client = OpenRouterClient(api_key="sk-test", http_client=_client(handler))
        result = client.complete([{"role": "user", "content": "q"}])
        assert result.text == "ok"
        assert len(attempts) == 2

    def test_stream_error_payload_raises_provider_error(self):
        client = OpenRouterClient(
            api_key="sk-test",
            http_client=_client(
                lambda r: httpx.Response(200, text=sse_text({"error": "upstream died"}))
            ),
        )
        with pytest.raises(ProviderError, match="upstream died"):
            client.complete([{"role": "user", "content": "q"}])

    def test_stream_non_sse_body_raises(self):
        # HTTP 200 with a plain JSON error body (not SSE)
        client = OpenRouterClient(
            api_key="sk-test",
            http_client=_client(
                lambda r: httpx.Response(200, json={"error": {"message": "no credits"}})
            ),
        )
        with pytest.raises(ProviderError, match="no SSE data"):
            client.complete([{"role": "user", "content": "q"}])

    def test_stream_malformed_chunk_raises(self):
        body = "data: {not json}\n\n"
        client = OpenRouterClient(
            api_key="sk-test", http_client=_client(lambda r: httpx.Response(200, text=body))
        )
        with pytest.raises(ProviderError, match="malformed SSE"):
            client.complete([{"role": "user", "content": "q"}])

    def test_stream_non_transient_http_error_no_retry(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(401, text="bad key")

        client = OpenRouterClient(
            api_key="sk-bad",
            provider_only=("baidu/fp8",),
            http_client=_client(handler),
        )
        with pytest.raises(ProviderError, match="401"):
            client.complete([{"role": "user", "content": "q"}])
        assert len(attempts) == 1

    def test_stream_midstream_disconnect_retries(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ReadError("connection reset")
            done = sse_text({"choices": [{"index": 0, "delta": {"content": "ok"}}]})
            return httpx.Response(200, text=done)

        client = OpenRouterClient(api_key="sk-test", http_client=_client(handler))
        result = client.complete([{"role": "user", "content": "q"}])
        assert result.text == "ok"
        assert len(attempts) == 2

    def test_stream_missing_usage_still_counts_call(self):
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}
                ]
            }
        ]
        client = OpenRouterClient(api_key="sk-test", http_client=_sse_client(chunks))
        result = client.complete([{"role": "user", "content": "q"}])

        assert result.text == "x"
        assert result.usage.llm_calls == 1
        assert result.usage.prompt_tokens == 0


class TestProviderResolution:
    def _endpoints_client(self, tags_response):
        def handler(request):
            if request.url.path.endswith("/endpoints"):
                return httpx.Response(200, json=tags_response)
            return httpx.Response(200, json=ok_response())

        return OpenRouterClient(
            api_key="k", stream=False, http_client=_client(handler)
        )

    def test_resolves_vetted_fp8_tags(self):
        # vetting drops deepinfra (dev25 per-provider acc 0.57) by default
        client = self._endpoints_client(
            {
                "data": {
                    "endpoints": [
                        {"tag": "baidu/fp8", "quantization": "fp8"},
                        {"tag": "morph/fp8", "quantization": "fp8"},
                        {"tag": "fireworks/fp4", "quantization": "fp4"},
                        {"tag": "deepinfra/fp8", "quantization": "fp8"},
                        {"quantization": "fp8"},  # tagless: ignored
                    ]
                }
            }
        )
        assert client._provider_only == ("baidu/fp8", "morph/fp8")

    def test_unresolvable_falls_back_to_single_pin(self):
        client = self._endpoints_client({"data": {"endpoints": []}})
        assert client._provider_only == ("baidu/fp8",)

    def test_explicit_pin_skips_resolution(self):
        client = OpenRouterClient(
            api_key="k",
            stream=False,
            provider_only=("baidu/fp8",),
            http_client=_client(lambda r: httpx.Response(200, json=ok_response())),
        )
        assert client._provider_only == ("baidu/fp8",)


class TestRequireContent:
    @staticmethod
    def _client_with(responses):
        seq = iter(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            r = next(seq)
            r = r.copy()
            r["model"] = body.get("model", "m")
            return httpx.Response(200, json=r)

        return OpenRouterClient(model="m", api_key="k", stream=False,
                                http_client=_client(handler))

    @staticmethod
    def _resp(text, finish="stop", ptok=10, ctok=5):
        return {"choices": [{"message": {"content": text},
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": ptok, "completion_tokens": ctok}}

    def test_empty_at_length_escalates_until_content(self):
        client = self._client_with([
            self._resp("", finish="length", ctok=1024),
            self._resp("real summary", ctok=200),
        ])
        c = client.complete([{"role": "user", "content": "s"}],
                            max_tokens=1024, require_content=True)
        assert c.text == "real summary"
        assert c.usage.llm_calls == 2      # escalation visible in telemetry
        assert c.usage.completion_tokens == 1224  # both attempts billed

    def test_gives_up_after_two_escalations_returning_last(self):
        client = self._client_with([
            self._resp("", finish="length", ctok=1024),
            self._resp("", finish="length", ctok=2048),
            self._resp("", finish="length", ctok=4096),
        ])
        c = client.complete([{"role": "user", "content": "s"}],
                            max_tokens=1024, require_content=True)
        assert c.text == ""
        assert c.finish_reason == "length"
        assert c.usage.llm_calls == 3

    def test_empty_at_stop_is_returned_without_escalation(self):
        client = self._client_with([self._resp("", finish="stop")])
        c = client.complete([{"role": "user", "content": "s"}],
                            max_tokens=1024, require_content=True)
        assert c.usage.llm_calls == 1  # only length-truncation escalates


class TestProviderRotation:
    def test_offset_pins_single_rotated_tag(self):
        from context_eval.provider import OpenRouterClient

        tags = ("a/fp8", "b/fp8", "c/fp8")
        c0 = OpenRouterClient(provider_only=tags, provider_offset=0)
        c1 = OpenRouterClient(provider_only=tags, provider_offset=1)
        c3 = OpenRouterClient(provider_only=tags, provider_offset=3)  # wraps
        assert c0._provider_only == ("a/fp8",)
        assert c1._provider_only == ("b/fp8",)
        assert c3._provider_only == ("a/fp8",)

    def test_no_offset_keeps_full_list(self):
        from context_eval.provider import OpenRouterClient

        tags = ("a/fp8", "b/fp8")
        c = OpenRouterClient(provider_only=tags)
        assert c._provider_only == tags

    def test_env_offset_respected(self, monkeypatch):
        from context_eval.provider import OpenRouterClient

        monkeypatch.setenv("CONTEXT_EVAL_PROVIDER_OFFSET", "2")
        tags = ("a/fp8", "b/fp8", "c/fp8")
        c = OpenRouterClient(provider_only=tags)
        assert c._provider_only == ("c/fp8",)
        monkeypatch.delenv("CONTEXT_EVAL_PROVIDER_OFFSET")
        c2 = OpenRouterClient(provider_only=tags)
        assert c2._provider_only == tags


class _RotatingFakeTransport:
    """429s whatever provider is pinned first; serves the second."""

    def __init__(self):
        self.seen_pins = []

    def complete(self, client, body):
        pin = body["provider"]["only"][0]
        self.seen_pins.append(pin)
        from context_eval.provider import Completion

        if pin == "a/fp8":
            from context_eval.provider import _TransientProviderError

            raise _TransientProviderError("HTTP 429: provider returned error")
        from context_eval.types import Usage
        return Completion(text="ok", finish_reason="stop", usage=Usage(),
                          provider=pin, tool_calls=(), model="m")


def test_429_rotation_moves_to_next_pinned_provider(monkeypatch):
    from context_eval import provider as P
    from context_eval.provider import OpenRouterClient

    tags = ("a/fp8", "b/fp8", "c/fp8")
    c = OpenRouterClient(provider_only=tags, provider_offset=0, stream=False)
    assert c._provider_only == ("a/fp8",)  # pinned by offset
    fake = _RotatingFakeTransport()
    monkeypatch.setattr(c, "_post", lambda body: fake.complete(c, body))
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    result = c.complete([{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert fake.seen_pins[0] == "a/fp8" and fake.seen_pins[1] == "b/fp8"


def test_unpinned_client_does_not_narrow_on_429(monkeypatch):
    from context_eval import provider as P
    from context_eval.provider import OpenRouterClient

    tags = ("a/fp8", "b/fp8")
    c = OpenRouterClient(provider_only=tags, stream=False)  # no offset: multi-tag
    assert c._provider_only == tags
    calls = []

    def post(body):
        calls.append(list(body["provider"]["only"]))
        from context_eval.provider import _TransientProviderError

        raise _TransientProviderError("HTTP 429: provider returned error")

    monkeypatch.setattr(c, "_post", post)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    try:
        c.complete([{"role": "user", "content": "hi"}])
    except P.ProviderError:
        pass
    assert len(calls) >= 2  # the 429 path ran and retried
    assert all(pins == list(tags) for pins in calls)  # never narrowed to one


class TestVettedAllowlist:
    def _fake_endpoints(self, monkeypatch):
        import context_eval.provider as P

        class _Resp:
            @staticmethod
            def json():
                tags = ["venice/fp8", "deepinfra/fp8", "parasail/fp8", "baseten/fp8"]
                return {"data": {"endpoints": [
                    {"tag": t, "quantization": "fp8"} for t in tags
                ]}}

        class _Client:
            def get(self, url):
                return _Resp()

            def close(self):
                pass

        monkeypatch.setattr(P.httpx, "Client", lambda **kw: _Client())

    def test_default_vets_to_good_tier(self, monkeypatch):
        import context_eval.provider as P

        monkeypatch.delenv("CONTEXT_EVAL_PROVIDER_ALLOWLIST", raising=False)
        self._fake_endpoints(monkeypatch)
        tags = P.resolve_provider_tags()
        assert tags == ("parasail/fp8", "baseten/fp8")
        assert "venice/fp8" not in tags and "deepinfra/fp8" not in tags

    def test_env_overrides_the_tier(self, monkeypatch):
        import context_eval.provider as P

        monkeypatch.setenv("CONTEXT_EVAL_PROVIDER_ALLOWLIST", "venice/fp8")
        self._fake_endpoints(monkeypatch)
        assert P.resolve_provider_tags() == ("venice/fp8",)

    def test_empty_env_disables_vetting(self, monkeypatch):
        import context_eval.provider as P

        monkeypatch.setenv("CONTEXT_EVAL_PROVIDER_ALLOWLIST", " ")
        self._fake_endpoints(monkeypatch)
        tags = P.resolve_provider_tags()
        assert "venice/fp8" in tags and "deepinfra/fp8" in tags


class TestStrictVettedRotation:
    def test_persistent_500_rotates_pin(self, monkeypatch):
        import context_eval.provider as P
        from context_eval.provider import OpenRouterClient

        tags = ("a/fp8", "b/fp8")
        c = OpenRouterClient(provider_only=tags, provider_offset=0, stream=False)
        seen = []

        def post(body):
            from context_eval.provider import Completion, _TransientProviderError
            from context_eval.types import Usage

            pin = body["provider"]["only"][0]
            seen.append(pin)
            if pin == "b/fp8":  # rotated pin succeeds
                return Completion(text="ok", finish_reason="stop",
                                  usage=Usage(), provider=pin,
                                  tool_calls=(), model="m")
            raise _TransientProviderError("HTTP 500: upstream error")

        monkeypatch.setattr(c, "_post", post)
        monkeypatch.setattr(P.time, "sleep", lambda s: None)
        r = c.complete([{"role": "user", "content": "q"}])
        assert r.text == "ok"
        assert seen[0] == "a/fp8" and seen[1] == "a/fp8"  # one in-place retry
        assert seen[2] == "b/fp8"  # rotated on persistent failure

    def test_fallbacks_off_by_default_env_enables(self, monkeypatch):
        from context_eval.provider import OpenRouterClient

        c = OpenRouterClient(provider_only=("a/fp8",))
        assert c._allow_fallbacks is False
        monkeypatch.setenv("CONTEXT_EVAL_ALLOW_FALLBACKS", "1")
        c2 = OpenRouterClient(provider_only=("a/fp8",))
        assert c2._allow_fallbacks is True
