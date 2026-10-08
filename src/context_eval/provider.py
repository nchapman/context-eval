"""Thin OpenAI-compatible chat-completions client for OpenRouter.

Hand-rolled on httpx: no SDK magic, full control of the wire format and of
what we measure. The harness owns the tool loop — this client only performs
single completions and reports usage verbatim.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from context_eval.types import ToolCall, Usage

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-chat"

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


class ProviderError(RuntimeError):
    pass


class OpenRouterClient:
    """One instance per (model, config); safe to share across strategies."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 180.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = http_client

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

        last_error: Exception | None = None
        for attempt in range(4):  # 1 try + 3 retries on transient failures
            try:
                return self._post(body)
            except _TransientProviderError as exc:
                last_error = exc
                time.sleep(min(2.0**attempt, 8.0))
        raise ProviderError(f"transient failures exhausted: {last_error}")

    def _post(self, body: dict[str, Any]) -> Completion:
        if not self._api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set; export it or pass api_key="
            )
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
            )
        finally:
            if self._client is None:
                client.close()

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
        usage = data.get("usage") or {}
        return Completion(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                llm_calls=1,
            ),
            finish_reason=choice.get("finish_reason", "unknown"),
            model=data.get("model", self.model),
        )


class _TransientProviderError(Exception):
    pass


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
