"""Thin OpenAI-compatible chat-completions client for OpenRouter.

Hand-rolled on httpx: no SDK magic, full control of the wire format and of
what we measure. The harness owns the tool loop — this client only performs
single completions and reports usage verbatim.

Streaming is the default: more robust for long reasoning-model generations,
and reasoning-token detail arrives in the final usage chunk. Provider routing
pins the exact serving path (default: Baidu FP8 for deepseek-v4.1-flash,
no fallbacks) for reproducible measurements.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from context_eval.types import ToolCall, Usage

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
# Pin serving for reproducibility (cheap, fast FP8 path); override per client.
DEFAULT_PROVIDER_ONLY: tuple[str, ...] = ("baidu/fp8",)
DEFAULT_ALLOW_FALLBACKS = False

# OpenAI wire-format message dicts. Kept as plain dicts: the provider stays a
# dumb transport and the harness owns transcript <-> wire conversion.
ChatMessage = dict[str, Any]


@dataclass(frozen=True)
class ToolSchema:
    """A JSON-schema tool advertisement (OpenAI function-tool shape)."""

    name: str
    description: str
    parameters: Mapping[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    finish_reason: str
    # The concrete model OpenRouter served (aliases resolve here; we record it
    # so every run is reproducible even when the request used an alias slug).
    model: str
    # The serving provider that actually handled the call (e.g. "baidu/fp8").
    provider: str = ""


class ProviderError(RuntimeError):
    pass


class LLMClient(Protocol):
    """Structural interface for anything that can complete() — the real
    OpenRouterClient satisfies it, tests supply fakes."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSchema] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion: ...


class _TransientProviderError(Exception):
    pass


def _parse_usage(usage: Mapping[str, Any] | None) -> Usage:
    usage = usage or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return Usage(
        prompt_tokens=usage.get("prompt_tokens", 0) or 0,
        completion_tokens=usage.get("completion_tokens", 0) or 0,
        reasoning_tokens=details.get("reasoning_tokens", 0) or 0,
        cached_tokens=prompt_details.get("cached_tokens", 0) or 0,
        cost=usage.get("cost", 0.0) or 0.0,
        llm_calls=1,
    )


class OpenRouterClient:
    """One instance per (model, config); safe to share across strategies."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 180.0,
        http_client: httpx.Client | None = None,
        *,
        stream: bool = True,
        provider_only: tuple[str, ...] = DEFAULT_PROVIDER_ONLY,
        allow_fallbacks: bool = DEFAULT_ALLOW_FALLBACKS,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = http_client
        self._stream = stream
        self._provider_only = tuple(provider_only) if provider_only else None
        self._allow_fallbacks = allow_fallbacks

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSchema] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [t.to_wire() for t in tools]
            body["tool_choice"] = "auto"
        if self._provider_only:
            body["provider"] = {
                "only": list(self._provider_only),
                "allow_fallbacks": self._allow_fallbacks,
            }

        last_error: Exception | None = None
        attempts = 7  # shared-pool rate limits can persist a while
        for attempt in range(attempts):
            try:
                if self._stream:
                    return self._post_stream(body)
                return self._post(body)
            except _TransientProviderError as exc:
                last_error = exc
                if attempt < attempts - 1:
                    # Exponential with jitter, capped: 1s..~45s
                    time.sleep(min(2.0**attempt, 45.0) * (0.5 + random.random()))
        raise ProviderError(f"transient failures exhausted: {last_error}")

    def _request_args(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set; export it or pass api_key="
            )
        return {
            "url": f"{self._base_url}/chat/completions",
            "headers": {"Authorization": f"Bearer {self._api_key}"},
            "json": body,
        }

    def _post(self, body: dict[str, Any]) -> Completion:
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            response = client.post(**self._request_args(body))
        finally:
            if self._client is None:
                client.close()
        return self._parse_plain_response(response)

    def _parse_plain_response(self, response: httpx.Response) -> Completion:
        if response.status_code in (429, 500, 502, 503, 529):
            raise _TransientProviderError(
                f"HTTP {response.status_code}: {response.text[:500]}"
            )
        if response.status_code != 200:
            raise ProviderError(f"HTTP {response.status_code}: {response.text[:500]}")

        data = response.json()
        if "error" in data:
            raise ProviderError(str(data["error"])[:500])

        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = tuple(
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", "") or "",
            )
            for tc in message.get("tool_calls") or ()
        )
        return Completion(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=_parse_usage(data.get("usage")),
            finish_reason=choice.get("finish_reason", "unknown"),
            model=data.get("model", self.model),
            provider=data.get("provider") or data.get("provider_name") or "",
        )

    def _post_stream(self, body: dict[str, Any]) -> Completion:
        body = {
            **body,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            try:
                with client.stream("POST", **self._request_args(body)) as response:
                    if response.status_code in (429, 500, 502, 503, 529):
                        detail = response.read()
                        raise _TransientProviderError(
                            f"HTTP {response.status_code}: {detail[:500]!r}"
                        )
                    if response.status_code != 200:
                        detail = response.read()
                        raise ProviderError(
                            f"HTTP {response.status_code}: {detail[:500]!r}"
                        )
                    parsed = self._consume_sse(response)
            except httpx.HTTPError as exc:  # connection died mid-stream
                raise _TransientProviderError(str(exc)) from exc
        finally:
            if self._client is None:
                client.close()
        return parsed

    def _consume_sse(self, response: httpx.Response) -> Completion:
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason = "unknown"
        model = self.model
        provider = ""
        usage: Usage | None = None
        saw_data = False

        for raw_line in response.iter_lines():
            if not raw_line.startswith("data:"):
                continue  # SSE comments (": OPENROUTER PROCESSING") etc.
            payload = raw_line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            saw_data = True
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"malformed SSE chunk: {payload[:200]!r}") from exc
            if not isinstance(chunk, dict):
                raise ProviderError(f"non-object SSE chunk: {payload[:200]!r}")
            if chunk.get("error"):
                raise ProviderError(str(chunk["error"])[:500])

            model = chunk.get("model") or model
            provider = chunk.get("provider") or chunk.get("provider_name") or provider
            if chunk.get("usage"):
                usage = _parse_usage(chunk["usage"])

            choices = chunk.get("choices") or []
            if not choices:  # usage-only final chunk
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or ():
                slot = tool_calls.setdefault(
                    tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

        if not saw_data:
            # HTTP 200 with a non-SSE body (e.g. a JSON error payload)
            raise ProviderError("stream contained no SSE data events")

        return Completion(
            text="".join(content_parts),
            tool_calls=tuple(
                ToolCall(id=slot["id"], name=slot["name"], arguments=slot["arguments"])
                for _, slot in sorted(tool_calls.items())
            ),
            usage=usage or Usage(llm_calls=1),
            finish_reason=finish_reason,
            model=model,
            provider=provider,
        )


def wire_messages_from_prompt(
    system: str, view: Sequence[ChatMessage]
) -> list[ChatMessage]:
    """Prepend the harness system prompt to an assembled view."""
    return [{"role": "system", "content": system}, *view]


def tool_result_message(tool_call_id: str, content: str) -> ChatMessage:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def assistant_tool_message(tool_calls: Sequence[ToolCall], text: str = "") -> ChatMessage:
    """The assistant turn that requested tool calls, in wire format."""
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ],
    }


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Best-effort parse of model-produced tool arguments; empty on failure."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
