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
            api_key="sk-bad", stream=False, http_client=_client(handler)
        )
        with pytest.raises(ProviderError, match="401"):
            client.complete([{"role": "user", "content": "q"}])
        assert len(attempts) == 1

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

        client = OpenRouterClient(api_key="sk-bad", http_client=_client(handler))
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
