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
    model="deepseek/deepseek-chat-v3-0324",
):
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
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
            model="deepseek/deepseek-chat", api_key="sk-test", http_client=_client(handler)
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

        client = OpenRouterClient(api_key="sk-test", http_client=_client(handler))
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

        client = OpenRouterClient(api_key="sk-test", http_client=_client(handler))
        result = client.complete([{"role": "user", "content": "q"}])
        assert result.text == "hi"
        assert len(attempts) == 3

    def test_non_transient_error_raises_immediately(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(401, text="bad key")

        client = OpenRouterClient(api_key="sk-bad", http_client=_client(handler))
        with pytest.raises(ProviderError, match="401"):
            client.complete([{"role": "user", "content": "q"}])
        assert len(attempts) == 1

    def test_provider_error_in_body_raises(self):
        def handler(request):
            return httpx.Response(200, json={"error": "insufficient credits"})

        client = OpenRouterClient(api_key="sk-test", http_client=_client(handler))
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
