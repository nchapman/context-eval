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
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from context_eval.types import ToolCall, Usage

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
# Pin serving for reproducibility (cheap, fast FP8 path); override per client.
# Routing: pin to every provider serving the model at fp8 quantization
# (resolved from OpenRouter's endpoints API). Falls back to the original
# single-provider pin when resolution is unavailable.
DEFAULT_PROVIDER_QUANT = "fp8"
_FALLBACK_PROVIDER_ONLY: tuple[str, ...] = ("baidu/fp8",)


def resolve_provider_tags(
    model: str = DEFAULT_MODEL,
    quant: str = DEFAULT_PROVIDER_QUANT,
    http_client: httpx.Client | None = None,
) -> tuple[str, ...]:
    """Routing tags (e.g. "baidu/fp8") for endpoints serving ``model`` at
    ``quant``. OpenRouter load-balances across the ``only`` list, so more
    providers means more shared-pool rate-limit headroom. Unresolvable
    (offline, API change) degrades to the single-provider pin."""
    url = f"{BASE_URL}/models/{model}/endpoints"
    own = http_client is None
    client = http_client or httpx.Client(timeout=30.0)
    try:
        payload = client.get(url).json()
        endpoints = payload.get("data", {}).get("endpoints", [])
        tags = sorted(
            {
                e["tag"]
                for e in endpoints
                if e.get("quantization") == quant and e.get("tag")
            }
        )
        return tuple(tags) or _FALLBACK_PROVIDER_ONLY
    except Exception:
        return _FALLBACK_PROVIDER_ONLY
    finally:
        if own:
            client.close()
# Fallbacks ON by default: pinning to the fp8 `only` list with
# allow_fallbacks=False funneled evening traffic onto whichever fp8
# upstream OpenRouter preferred (baidu), whose shared-pool per-key rate
# limits then 429'd sequential single calls. With fallbacks enabled the
# pin stays a *preference* (fp8 pricing when healthy) and OpenRouter can
# route around a congested or throttled upstream.
DEFAULT_ALLOW_FALLBACKS = True

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
    # Reasoning-channel text; non-empty when the model answered entirely in
    # reasoning and left the content channel empty (observed on ~30% of
    # short factual answers with deepseek-v4.1-flash).
    reasoning: str = ""


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
        provider_only: tuple[str, ...] | None = None,
        allow_fallbacks: bool = DEFAULT_ALLOW_FALLBACKS,
        provider_offset: int | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = http_client
        self._stream = stream
        # None = auto: every fp8 endpoint, resolved once at construction.
        if provider_only is None:
            provider_only = resolve_provider_tags(model, http_client=http_client)
        tags = tuple(provider_only) if provider_only else None
        # Sticky per-worker rotation: offset k pins to tags[k % n] only.
        # OpenRouter's own balancing over an `only` list still funnels
        # concurrent traffic onto one warm upstream (the baidu 429
        # incident); one-pinned-provider-per-worker with fallbacks spreads
        # load deterministically. Offset also reads from the environment
        # (CONTEXT_EVAL_PROVIDER_OFFSET) so sharded workers rotate with
        # no code changes.
        offset = provider_offset
        if offset is None:
            env = os.environ.get("CONTEXT_EVAL_PROVIDER_OFFSET")
            offset = int(env) if env and env.strip().lstrip("-").isdigit() else None
        if offset is not None and tags and len(tags) > 1:
            tags = (tags[offset % len(tags)],)
        self._provider_only = tags
        # Full pool kept for 429 rotation (see complete()); the pinned tag
        # is tags[0] of the pool when rotation enabled.
        self._rotation_pool = tuple(provider_only) if provider_only else ()
        self._allow_fallbacks = allow_fallbacks

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSchema] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning: Mapping[str, Any] | None = None,
        require_content: bool = False,
    ) -> Completion:
        """require_content=True treats an empty-content completion truncated
        at the cap (finish=length) as retryable: reasoning models can burn
        the entire max_tokens thinking and emit nothing (measured: 8192
        reasoning tokens, zero content). Up to two escalations double the
        cap; usage accumulates across attempts because every attempt bills
        (usage.llm_calls exposes the escalation count)."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning:
            # e.g. {"enabled": False} for utility calls (summarization):
            # validated against flash via OpenRouter -- honored end-to-end
            # (reasoning_tokens drop to 0, content lands, clean stop).
            body["reasoning"] = dict(reasoning)
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
        ladder_start = time.monotonic()
        billed = Usage()
        escalations = 0
        result: Completion | None = None
        rotation: tuple[str, ...] = ()  # pins to try after the current one
        if self._rotation_pool and len(self._rotation_pool) > 1:
            cur = self._provider_only or ()
            i = self._rotation_pool.index(cur[0]) if cur and cur[0] in self._rotation_pool else 0
            rotation = self._rotation_pool[i + 1 :] + self._rotation_pool[: i + 1]
        for attempt in range(attempts):
            try:
                if self._stream:
                    result = self._post_stream(body)
                else:
                    result = self._post(body)
            except _TransientProviderError as exc:
                last_error = exc
                # Total ladder budget: 7 x (600s stream deadline + backoff)
                # can reach ~78 min on a wedged upstream -- a stuck run that
                # looks hung (measured: 22 min at 3s CPU, mid-ladder).
                if time.monotonic() - ladder_start > 900:
                    raise ProviderError(
                        f"transient ladder exceeded 900s wall budget: {last_error}"
                    ) from exc
                # A pin that stays throttled blocks OpenRouter's own
                # rerouting (only-listed upstream accepts then fails), so
                # persistent 429s rotate to the next pinned provider
                # instead of re-hitting the same pool for minutes.
                pinned = bool(self._provider_only) and len(self._provider_only) == 1
                if rotation and pinned and str(exc).find("429") != -1:
                    body["provider"]["only"] = [rotation[0]]
                    rotation = rotation[1:]
                    print(
                        f"[provider] pin throttled; rotating to {body['provider']['only'][0]}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if attempt < attempts - 1:
                    # Exponential with jitter, capped: 1s..~45s
                    wait = min(2.0**attempt, 45.0) * (0.5 + random.random())
                    print(
                        f"[provider] attempt {attempt + 1}/{attempts} failed "
                        f"({type(exc).__name__}: {str(exc)[:80]}); "
                        f"retrying in {wait:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(wait)
                continue
            billed.add(result.usage)
            if (
                require_content
                and not result.text.strip()
                and result.finish_reason == "length"
            ):
                escalations += 1
                if escalations <= 2:
                    body["max_tokens"] = min(body["max_tokens"] * 2, 32_768)
                    last_error = _TransientProviderError(
                        f"empty content at cap; escalating max_tokens to "
                        f"{body['max_tokens']}"
                    )
                    continue
            return Completion(
                text=result.text,
                tool_calls=result.tool_calls,
                usage=billed,
                finish_reason=result.finish_reason,
                model=result.model,
                provider=result.provider,
                reasoning=result.reasoning,
            )
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
            reasoning=message.get("reasoning_content") or "",
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
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason = "unknown"
        model = self.model
        provider = ""
        usage: Usage | None = None
        saw_data = False
        started = time.monotonic()

        for raw_line in response.iter_lines():
            # Wall-clock guards: keepalive comments reset httpx read
            # timeouts forever, so a stalled stream would otherwise hang a
            # run indefinitely. Retryable -- the retry ladder takes over.
            elapsed = time.monotonic() - started
            if elapsed > 600:
                raise _TransientProviderError(
                    f"stream wall-clock deadline exceeded ({elapsed:.0f}s)"
                )
            if not saw_data and elapsed > 240:
                raise _TransientProviderError(
                    f"no stream data within {elapsed:.0f}s"
                )
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
            # DeepSeek-style reasoning: some responses carry the whole
            # answer here with an empty content channel (observed on ~30%
            # of short factual answers with v4.1-flash).
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
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
            reasoning="".join(reasoning_parts),
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
