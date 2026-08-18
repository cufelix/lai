"""OpenAI-compatible chat-completions provider.

Covers OpenAI, OpenRouter, Groq, DeepSeek, z.ai's OpenAI endpoint and local
Ollama — anything exposing ``/chat/completions`` with function tools. Vision is
sent as ``image_url`` data URIs where the model supports it.
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

RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_RETRIES = 4


class OpenAIProvider:
    """Chat-completions with function tools."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 8192,
        temperature: float = 1.0,
        timeout: float = 180.0,
        name: str = "openai",
        supports_vision: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.supports_vision = supports_vision
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[dict] | None = None,
        stream: StreamCallback | None = None,
    ) -> TurnResult:
        wire: list[dict] = []
        if system:
            wire.append({"role": "system", "content": system})
        for message in messages:
            wire.extend(_encode_message(message, supports_vision=self.supports_vision))

        payload: dict = {
            "model": self.model,
            "messages": wire,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = self._request("/chat/completions", payload)
        result = _decode_response(data)
        if stream is not None and result.text:
            stream("text", result.text)
        return result

    def _request(self, path: str, payload: dict) -> dict:
        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.post(path, json=payload)
            except httpx.RequestError as exc:
                last_error = f"network error: {exc}"
                time.sleep(min(1.5 * (2**attempt), 20.0))
                continue
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise ProviderError(f"{self.name}: malformed JSON", detail=str(exc)) from exc
            last_error = (response.text or "")[:500]
            if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES - 1:
                time.sleep(min(1.5 * (2**attempt), 20.0))
                continue
            raise ProviderError(
                f"{self.name}: HTTP {response.status_code} from {self.base_url}", detail=last_error
            )
        raise ProviderError(f"{self.name}: failed after {MAX_RETRIES} attempts", detail=last_error)


def _encode_message(message: Message, *, supports_vision: bool) -> list[dict]:
    """One neutral Message can become several OpenAI messages (tool results split out)."""
    out: list[dict] = []
    parts: list[dict] = []
    tool_calls: list[dict] = []

    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            if supports_vision:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.media_type};base64,{block.base64}"},
                    }
                )
            else:
                parts.append({"type": "text", "text": "[screenshot omitted: model has no vision]"})
        elif isinstance(block, ThinkingBlock):
            continue  # not representable on this API
        elif isinstance(block, ToolCall):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
        elif isinstance(block, ToolResultBlock):
            content = block.content or "(no output)"
            if block.images and supports_vision:
                content += f"\n[{len(block.images)} screenshot(s) attached in the next message]"
            out.append({"role": "tool", "tool_call_id": block.tool_use_id, "content": content})
            if block.images and supports_vision:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + __import__("base64").b64encode(image).decode("ascii")
                                },
                            }
                            for image in block.images
                        ],
                    }
                )

    if parts or tool_calls:
        entry: dict = {"role": message.role}
        if parts:
            entry["content"] = parts if len(parts) > 1 or parts[0]["type"] != "text" else parts[0]["text"]
        elif tool_calls:
            entry["content"] = None
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.insert(0, entry) if message.role == "assistant" and out else out.append(entry)
    return out


def _decode_response(data: dict) -> TurnResult:
    choices = data.get("choices") or [{}]
    choice = choices[0]
    raw_message = choice.get("message") or {}
    blocks: list = []

    content = raw_message.get("content")
    if isinstance(content, str) and content:
        blocks.append(TextBlock(content))
    elif isinstance(content, list):
        blocks.extend(
            TextBlock(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )

    for index, call in enumerate(raw_message.get("tool_calls") or []):
        function = call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            parsed = {}
        blocks.append(
            ToolCall(id=call.get("id") or f"call_{index}", name=function.get("name", ""), input=parsed)
        )

    usage_raw = data.get("usage") or {}
    finish = choice.get("finish_reason") or "end_turn"
    return TurnResult(
        message=Message("assistant", blocks),
        stop_reason="tool_use" if finish == "tool_calls" else finish,
        usage=Usage(
            input_tokens=int(usage_raw.get("prompt_tokens", 0)),
            output_tokens=int(usage_raw.get("completion_tokens", 0)),
        ),
        model=data.get("model", ""),
        raw=data,
    )
