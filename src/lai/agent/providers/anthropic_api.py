"""Anthropic Messages API provider.

Speaks the wire protocol directly over httpx rather than through an SDK, because
the same code then serves every Anthropic-compatible endpoint — api.anthropic.com,
z.ai/GLM, and self-hosted gateways — with only a base URL and auth header change.
"""

from __future__ import annotations

import json
import time

import httpx

from ...errors import ProviderError
from .base import (
    ImageBlock,
    Message,
    StreamCallback,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    TurnResult,
    Usage,
)

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
MAX_RETRIES = 4


class AnthropicProvider:
    """Anthropic-compatible chat completion with tool use and streaming."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        thinking_budget: int = 0,
        timeout: float = 180.0,
        name: str = "anthropic",
        extra_headers: dict | None = None,
        prompt_cache: bool = True,
    ) -> None:
        if not api_key:
            raise ProviderError(f"{name}: no API key configured")
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.prompt_cache = prompt_cache
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            # Send both auth styles: api.anthropic.com wants x-api-key, most
            # compatible gateways (z.ai included) want a bearer token.
            "x-api-key": api_key,
            "authorization": f"Bearer {api_key}",
            **(extra_headers or {}),
        }
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # -- request ---------------------------------------------------------

    def _payload(self, messages: list[Message], system: str, tools: list[dict] | None) -> dict:
        encoded = [_encode_message(m) for m in messages]
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": encoded,
        }
        if system:
            # A list block lets us mark it cacheable on providers that support it.
            payload["system"] = [{"type": "text", "text": system}]
        if tools:
            payload["tools"] = list(tools)
        if self.prompt_cache:
            _mark_cacheable(payload)
        if self.thinking_budget > 0:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            # Extended thinking requires the default temperature.
        else:
            payload["temperature"] = self.temperature
        return payload

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[dict] | None = None,
        stream: StreamCallback | None = None,
    ) -> TurnResult:
        payload = self._payload(messages, system, tools)
        if stream is not None:
            payload["stream"] = True
            return self._stream(payload, stream)
        return self._once(payload)

    def _once(self, payload: dict) -> TurnResult:
        data = self._request_json("/v1/messages", payload)
        return _decode_response(data)

    def _request_json(self, path: str, payload: dict) -> dict:
        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.post(path, json=payload)
            except httpx.RequestError as exc:
                last_error = f"network error: {exc}"
                _backoff(attempt)
                continue
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise ProviderError(
                        f"{self.name}: malformed JSON response", detail=str(exc)
                    ) from exc
            last_error = _error_detail(response)
            if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES - 1:
                _backoff(attempt, retry_after=response.headers.get("retry-after"))
                continue
            raise ProviderError(
                f"{self.name}: HTTP {response.status_code} from {self.base_url}", detail=last_error
            )
        raise ProviderError(f"{self.name}: request failed after {MAX_RETRIES} attempts", detail=last_error)

    def _stream(self, payload: dict, callback: StreamCallback) -> TurnResult:
        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                with self._client.stream("POST", "/v1/messages", json=payload) as response:
                    if response.status_code != 200:
                        response.read()
                        last_error = _error_detail(response)
                        if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES - 1:
                            _backoff(attempt, retry_after=response.headers.get("retry-after"))
                            continue
                        raise ProviderError(
                            f"{self.name}: HTTP {response.status_code} from {self.base_url}",
                            detail=last_error,
                        )
                    return _consume_stream(response.iter_lines(), callback)
            except httpx.RequestError as exc:
                last_error = f"network error: {exc}"
                if attempt < MAX_RETRIES - 1:
                    _backoff(attempt)
                    continue
                raise ProviderError(f"{self.name}: streaming failed", detail=last_error) from exc
        raise ProviderError(f"{self.name}: streaming failed after retries", detail=last_error)


# -- wire encoding -------------------------------------------------------


def _encode_message(message: Message) -> dict:
    blocks: list[dict] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text:
                blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.media_type,
                        "data": block.base64,
                    },
                }
            )
        elif isinstance(block, ThinkingBlock):
            if block.signature:
                blocks.append(
                    {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
                )
        elif isinstance(block, ToolCall):
            blocks.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
        elif isinstance(block, ToolResultBlock):
            content: list[dict] = []
            if block.content:
                content.append({"type": "text", "text": block.content})
            for image in block.images:
                import base64 as b64  # noqa: PLC0415

                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64.b64encode(image).decode("ascii"),
                        },
                    }
                )
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": content or [{"type": "text", "text": "(no output)"}],
                    "is_error": block.is_error,
                }
            )
    if not blocks:
        blocks = [{"type": "text", "text": "(empty)"}]
    return {"role": message.role, "content": blocks}


def _decode_response(data: dict) -> TurnResult:
    blocks: list = []
    for raw in data.get("content", []):
        kind = raw.get("type")
        if kind == "text":
            blocks.append(TextBlock(raw.get("text", "")))
        elif kind == "thinking":
            blocks.append(ThinkingBlock(raw.get("thinking", ""), raw.get("signature", "")))
        elif kind == "tool_use":
            blocks.append(
                ToolCall(id=raw.get("id", ""), name=raw.get("name", ""), input=raw.get("input") or {})
            )
    usage_raw = data.get("usage") or {}
    return TurnResult(
        message=Message("assistant", blocks),
        stop_reason=data.get("stop_reason") or "end_turn",
        usage=Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0)),
            output_tokens=int(usage_raw.get("output_tokens", 0)),
            cache_read_tokens=int(usage_raw.get("cache_read_input_tokens", 0)),
            cache_write_tokens=int(usage_raw.get("cache_creation_input_tokens", 0)),
        ),
        model=data.get("model", ""),
        raw=data,
    )


def _consume_stream(lines, callback: StreamCallback) -> TurnResult:
    """Reassemble SSE deltas into a TurnResult."""
    blocks: list = []
    partial: dict[int, dict] = {}
    stop_reason = "end_turn"
    usage = Usage()
    model = ""

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            event = json.loads(chunk)
        except json.JSONDecodeError:
            continue

        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message") or {}
            model = message.get("model", model)
            raw_usage = message.get("usage") or {}
            usage = Usage(
                input_tokens=int(raw_usage.get("input_tokens", 0)),
                output_tokens=int(raw_usage.get("output_tokens", 0)),
                cache_read_tokens=int(raw_usage.get("cache_read_input_tokens", 0)),
                cache_write_tokens=int(raw_usage.get("cache_creation_input_tokens", 0)),
            )
        elif kind == "content_block_start":
            index = event.get("index", 0)
            block = event.get("content_block") or {}
            partial[index] = {
                "type": block.get("type"),
                "text": block.get("text", ""),
                "thinking": block.get("thinking", ""),
                "signature": block.get("signature", ""),
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "json": "",
            }
            if block.get("type") == "tool_use":
                callback("tool", block.get("name", ""))
        elif kind == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta") or {}
            slot = partial.setdefault(
                index,
                {"type": delta.get("type"), "text": "", "thinking": "", "json": "",
                 "id": "", "name": "", "signature": ""},
            )
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text", "")
                slot["text"] += text
                callback("text", text)
            elif dtype == "thinking_delta":
                text = delta.get("thinking", "")
                slot["thinking"] += text
                callback("thinking", text)
            elif dtype == "signature_delta":
                slot["signature"] += delta.get("signature", "")
            elif dtype == "input_json_delta":
                slot["json"] += delta.get("partial_json", "")
        elif kind == "content_block_stop":
            index = event.get("index", 0)
            slot = partial.pop(index, None)
            if slot:
                blocks.append(_finish_block(slot))
        elif kind == "message_delta":
            delta = event.get("delta") or {}
            stop_reason = delta.get("stop_reason") or stop_reason
            raw_usage = event.get("usage") or {}
            if raw_usage:
                usage = Usage(
                    input_tokens=usage.input_tokens or int(raw_usage.get("input_tokens", 0)),
                    output_tokens=int(raw_usage.get("output_tokens", usage.output_tokens)),
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                )
        elif kind == "error":
            error = event.get("error") or {}
            raise ProviderError(
                f"stream error: {error.get('type', 'unknown')}", detail=error.get("message", "")
            )

    blocks.extend(_finish_block(slot) for slot in partial.values())

    return TurnResult(
        message=Message("assistant", [b for b in blocks if b is not None]),
        stop_reason=stop_reason,
        usage=usage,
        model=model,
    )


def _finish_block(slot: dict):
    kind = slot.get("type")
    if kind == "text":
        return TextBlock(slot.get("text", ""))
    if kind == "thinking":
        return ThinkingBlock(slot.get("thinking", ""), slot.get("signature", ""))
    if kind == "tool_use":
        raw = slot.get("json") or ""
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        return ToolCall(id=slot.get("id", ""), name=slot.get("name", ""), input=parsed)
    text = slot.get("text", "")
    return TextBlock(text) if text else None


def _error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return (response.text or "")[:500]
    error = data.get("error")
    if isinstance(error, dict):
        return f"{error.get('type', '')}: {error.get('message', '')}".strip(": ")
    return json.dumps(data)[:500]


def _backoff(attempt: int, *, retry_after: str | None = None) -> None:
    if retry_after:
        try:
            time.sleep(min(float(retry_after), 30.0))
            return
        except (TypeError, ValueError):
            pass
    time.sleep(min(1.5 * (2**attempt), 20.0))


# Anthropic-style prompt caching. The prefix of a request — tools, then system,
# then the conversation so far — is identical from one turn to the next, and an
# agent loop re-sends all of it every single time. Marking the end of that
# prefix means the far end charges a fraction for it and answers sooner.
#
# Four breakpoints are allowed; three are used, in the order the API concatenates
# them, because a marker only caches what comes before it:
#
#   tools → system → the transcript up to the previous turn
#
# The newest message is deliberately left unmarked: it is different every time,
# so marking it would write a cache entry that is never read.
CACHE_CONTROL = {"type": "ephemeral"}
MIN_CACHEABLE_MESSAGES = 3
"""Below this the prefix is too small to be worth a cache write."""


def _mark_cacheable(payload: dict) -> None:
    """Add cache breakpoints to the stable prefix of a request."""
    tools = payload.get("tools")
    if tools:
        tools[-1] = {**tools[-1], "cache_control": CACHE_CONTROL}

    system = payload.get("system")
    if system:
        system[-1] = {**system[-1], "cache_control": CACHE_CONTROL}

    messages = payload.get("messages") or []
    if len(messages) < MIN_CACHEABLE_MESSAGES:
        return
    # The turn before the newest one: everything up to it is settled history.
    for message in reversed(messages[:-1]):
        blocks = message.get("content")
        if isinstance(blocks, list) and blocks:
            blocks[-1] = {**blocks[-1], "cache_control": CACHE_CONTROL}
            return
