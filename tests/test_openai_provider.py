"""OpenAI-compatible wire format (covers OpenAI, OpenRouter, Ollama)."""

from __future__ import annotations

import base64
import json

import pytest

from lai.agent.providers.base import (
    Message,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
)
from lai.agent.providers.openai_api import OpenAIProvider, _decode_response, _encode_message

PNG = b"\x89PNGfake"


def encode(message: Message, *, vision: bool = True) -> list[dict]:
    return _encode_message(message, supports_vision=vision)


def test_plain_text_message_uses_a_string_body():
    out = encode(Message.user("hello"))
    assert out == [{"role": "user", "content": "hello"}]


def test_image_message_becomes_a_parts_array():
    out = encode(Message.user("look", images=[PNG]))
    parts = out[0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    payload = parts[1]["image_url"]["url"]
    assert payload.startswith("data:image/png;base64,")
    assert base64.b64decode(payload.split(",", 1)[1]) == PNG


def test_images_are_described_when_the_model_has_no_vision():
    parts = encode(Message.user("look", images=[PNG]), vision=False)[0]["content"]
    assert all(part["type"] == "text" for part in parts)
    assert any("screenshot omitted" in part["text"] for part in parts)


def test_tool_calls_are_emitted_as_function_calls():
    out = encode(Message("assistant", [ToolCall("c1", "ui_click", {"ref": 4})]))
    call = out[0]["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["function"]["name"] == "ui_click"
    assert json.loads(call["function"]["arguments"]) == {"ref": 4}


def test_tool_results_become_tool_role_messages():
    out = encode(Message("user", [ToolResultBlock("c1", "clicked ok")]))
    assert out[0] == {"role": "tool", "tool_call_id": "c1", "content": "clicked ok"}


def test_empty_tool_result_still_carries_content():
    assert encode(Message("user", [ToolResultBlock("c1", "")]))[0]["content"] == "(no output)"


def test_tool_result_images_are_split_into_a_following_user_message():
    out = encode(Message("user", [ToolResultBlock("c1", "shot taken", images=(PNG,))]))
    assert out[0]["role"] == "tool"
    assert out[1]["role"] == "user"
    assert out[1]["content"][0]["type"] == "image_url"


def test_thinking_blocks_are_dropped():
    # The chat-completions API has no representation for these.
    assert encode(Message("assistant", [ThinkingBlock("pondering", "sig")])) == []


def test_decode_text_response():
    turn = _decode_response({
        "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        "model": "gpt-x",
    })
    assert turn.text == "hi there"
    assert turn.usage.input_tokens == 11
    assert turn.model == "gpt-x"
    assert not turn.wants_tools


def test_decode_tool_calls_and_normalise_the_stop_reason():
    turn = _decode_response({
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "window_list", "arguments": '{"match":"x"}'}}
                ],
            },
            "finish_reason": "tool_calls",
        }],
    })
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].name == "window_list"
    assert turn.tool_calls[0].input == {"match": "x"}


def test_decode_tolerates_malformed_tool_arguments():
    turn = _decode_response({
        "choices": [{
            "message": {"tool_calls": [{"id": "c", "function": {"name": "x", "arguments": "{broken"}}]},
            "finish_reason": "tool_calls",
        }],
    })
    assert turn.tool_calls[0].input == {}


def test_decode_generates_an_id_when_the_server_omits_one():
    turn = _decode_response({
        "choices": [{"message": {"tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]},
                     "finish_reason": "tool_calls"}],
    })
    assert turn.tool_calls[0].id


def test_decode_list_style_content():
    turn = _decode_response({
        "choices": [{"message": {"content": [{"type": "text", "text": "part"}]}, "finish_reason": "stop"}]
    })
    assert turn.text == "part"


def test_decode_empty_response():
    assert _decode_response({}).text == ""


def test_ollama_style_construction_needs_no_real_key():
    provider = OpenAIProvider(
        api_key="ollama", model="qwen3-vl:2b", base_url="http://127.0.0.1:11434/v1",
        name="ollama", supports_vision=True,
    )
    try:
        assert provider.name == "ollama"
        assert provider.base_url == "http://127.0.0.1:11434/v1"
    finally:
        provider.close()


def test_trailing_slash_is_stripped_from_the_base_url():
    provider = OpenAIProvider(api_key="k", model="m", base_url="https://x.example/v1/")
    try:
        assert provider.base_url == "https://x.example/v1"
    finally:
        provider.close()


@pytest.mark.parametrize("status", [401, 400])
def test_non_retryable_http_errors_raise(monkeypatch, status):

    from lai.errors import ProviderError

    provider = OpenAIProvider(api_key="k", model="m", base_url="https://x.example/v1")

    class FakeResponse:
        status_code = status
        text = "denied"

        def json(self):
            raise ValueError

    monkeypatch.setattr(provider._client, "post", lambda *a, **kw: FakeResponse())
    try:
        with pytest.raises(ProviderError, match=str(status)):
            provider.complete([Message.user("hi")])
    finally:
        provider.close()


def test_system_prompt_and_tools_are_included(monkeypatch):
    captured: dict = {}
    provider = OpenAIProvider(api_key="k", model="m", base_url="https://x.example/v1")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    def fake_post(path, json=None, **kwargs):
        captured.update(json or {})
        return FakeResponse()

    monkeypatch.setattr(provider._client, "post", fake_post)
    try:
        provider.complete(
            [Message.user("go")],
            system="you are LAI",
            tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        )
        assert captured["messages"][0] == {"role": "system", "content": "you are LAI"}
        assert captured["tool_choice"] == "auto"
        assert captured["model"] == "m"
    finally:
        provider.close()


def test_cached_prompt_tokens_are_reported_separately():
    """`prompt_tokens` here already contains the cached ones, where Anthropic
    reports them apart. Usage has to mean one thing whoever answered, or the
    cost line double-counts on exactly one of the two."""
    turn = _decode_response({
        "choices": [{"message": {"content": "hi"}}],
        "usage": {
            "prompt_tokens": 16090,
            "completion_tokens": 99,
            "prompt_tokens_details": {"cached_tokens": 5504},
        },
    })
    assert turn.usage.cache_read_tokens == 5504
    assert turn.usage.input_tokens == 16090 - 5504
    assert turn.usage.input_tokens + turn.usage.cache_read_tokens == 16090


def test_a_backend_that_does_not_cache_reports_nothing_cached():
    turn = _decode_response({
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    })
    assert turn.usage.input_tokens == 11
    assert turn.usage.cache_read_tokens == 0


def test_a_null_details_block_is_not_a_crash():
    turn = _decode_response({
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 11, "prompt_tokens_details": None},
    })
    assert turn.usage.input_tokens == 11
