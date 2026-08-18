"""Provider wire formats. Pure translation tests — no network."""

from __future__ import annotations

import base64
import json

import pytest

from lai.agent.providers.anthropic_api import (
    _consume_stream,
    _decode_response,
    _encode_message,
    _finish_block,
)
from lai.agent.providers.base import (
    Message,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
)
from lai.agent.providers.registry import (
    Credential,
    _extract,
    build_provider,
    discover_credentials,
)
from lai.config import ProviderConfig
from lai.errors import ProviderError

PNG = b"\x89PNG\r\n\x1a\nfakeimagedata"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ZAI_API_KEY",
        "Z_AI_API_KEY", "GLM_API_KEY", "BIGMODEL_API_KEY", "OPENAI_API_KEY",
        "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
        "GLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


# -- encoding ------------------------------------------------------------


def test_encode_text_message():
    wire = _encode_message(Message.user("hello"))
    assert wire == {"role": "user", "content": [{"type": "text", "text": "hello"}]}


def test_encode_image_message():
    wire = _encode_message(Message.user("look", images=[PNG]))
    kinds = [block["type"] for block in wire["content"]]
    assert kinds == ["text", "image"]
    image = wire["content"][1]
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == PNG


def test_encode_tool_use():
    wire = _encode_message(Message("assistant", [ToolCall("c1", "ui_click", {"ref": 3})]))
    assert wire["content"][0] == {
        "type": "tool_use", "id": "c1", "name": "ui_click", "input": {"ref": 3}
    }


def test_encode_tool_result_with_images():
    wire = _encode_message(
        Message("user", [ToolResultBlock("c1", "clicked", images=(PNG,), is_error=False)])
    )
    block = wire["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "c1"
    assert block["is_error"] is False
    kinds = [part["type"] for part in block["content"]]
    assert kinds == ["text", "image"]


def test_encode_tool_result_error_flag():
    wire = _encode_message(Message("user", [ToolResultBlock("c1", "boom", is_error=True)]))
    assert wire["content"][0]["is_error"] is True


def test_encode_empty_tool_result_still_has_content():
    wire = _encode_message(Message("user", [ToolResultBlock("c1", "")]))
    assert wire["content"][0]["content"], "an empty content array is rejected by the API"


def test_encode_empty_message_is_not_empty_on_the_wire():
    wire = _encode_message(Message("user", []))
    assert wire["content"] == [{"type": "text", "text": "(empty)"}]


def test_unsigned_thinking_block_is_dropped():
    # Replaying a thinking block without its signature is rejected by the API.
    wire = _encode_message(Message("assistant", [ThinkingBlock("pondering", "")]))
    assert wire["content"] == [{"type": "text", "text": "(empty)"}]


def test_signed_thinking_block_is_kept():
    wire = _encode_message(Message("assistant", [ThinkingBlock("pondering", "sig123")]))
    assert wire["content"][0]["type"] == "thinking"


# -- decoding ------------------------------------------------------------


def test_decode_text_response():
    turn = _decode_response({
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "model": "m1",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    })
    assert turn.text == "hi"
    assert turn.stop_reason == "end_turn"
    assert turn.usage.input_tokens == 7
    assert turn.model == "m1"
    assert not turn.wants_tools


def test_decode_tool_use_response():
    turn = _decode_response({
        "content": [
            {"type": "text", "text": "let me look"},
            {"type": "tool_use", "id": "c9", "name": "ui_snapshot", "input": {"scope": "focused"}},
        ],
        "stop_reason": "tool_use",
    })
    assert turn.wants_tools
    assert turn.tool_calls[0].name == "ui_snapshot"
    assert turn.tool_calls[0].input == {"scope": "focused"}
    assert turn.text == "let me look"


def test_decode_cache_usage():
    turn = _decode_response({
        "content": [],
        "usage": {"input_tokens": 1, "output_tokens": 2,
                  "cache_read_input_tokens": 30, "cache_creation_input_tokens": 40},
    })
    assert turn.usage.cache_read_tokens == 30
    assert turn.usage.cache_write_tokens == 40


def test_decode_empty_response():
    turn = _decode_response({})
    assert turn.text == "" and turn.stop_reason == "end_turn"


# -- streaming -----------------------------------------------------------


def sse(events: list[dict]) -> list[str]:
    return [f"data: {json.dumps(event)}" for event in events]


def test_stream_reassembles_text_and_tool_input():
    lines = sse([
        {"type": "message_start", "message": {"model": "m1", "usage": {"input_tokens": 12, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "check."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "c1", "name": "ui_click"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"ref"'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ': 42, "prefer'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '_action": true}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 25}},
        {"type": "message_stop"},
    ])
    seen: list[tuple[str, str]] = []
    turn = _consume_stream(lines, lambda kind, payload: seen.append((kind, payload)))

    assert turn.text == "Let me check."
    assert turn.stop_reason == "tool_use"
    call = turn.tool_calls[0]
    assert call.name == "ui_click"
    assert call.input == {"ref": 42, "prefer_action": True}
    assert [payload for kind, payload in seen if kind == "text"] == ["Let me ", "check."]
    assert ("tool", "ui_click") in seen
    assert turn.usage.input_tokens == 12 and turn.usage.output_tokens == 25


def test_stream_handles_thinking_deltas():
    lines = sse([
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
    ])
    seen: list[str] = []
    turn = _consume_stream(lines, lambda kind, payload: seen.append(kind))
    blocks = [b for b in turn.message.content if isinstance(b, ThinkingBlock)]
    assert blocks and blocks[0].thinking == "hmm" and blocks[0].signature == "sig"
    assert "thinking" in seen


def test_stream_ignores_noise_and_done():
    lines = ["", ": keepalive", "data: [DONE]", "data: not-json",
             *sse([{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "ok"}},
                   {"type": "content_block_stop", "index": 0}])]
    assert _consume_stream(lines, lambda *_: None).text == "ok"


def test_stream_error_event_raises():
    lines = sse([{"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}])
    with pytest.raises(ProviderError, match="overloaded_error"):
        _consume_stream(lines, lambda *_: None)


def test_stream_closes_unterminated_blocks():
    lines = sse([
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "cut off"}},
    ])
    assert _consume_stream(lines, lambda *_: None).text == "cut off"


def test_finish_block_tolerates_malformed_tool_json():
    call = _finish_block({"type": "tool_use", "id": "c1", "name": "x", "json": '{"broken'})
    assert isinstance(call, ToolCall) and call.input == {}


def test_finish_block_empty_text_is_dropped():
    assert _finish_block({"type": "unknown", "text": ""}) is None


# -- usage ---------------------------------------------------------------


def test_usage_arithmetic():
    total = Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40)
    assert (total.input_tokens, total.output_tokens) == (11, 22)
    assert (total.cache_read_tokens, total.cache_write_tokens) == (33, 44)
    assert total.total == 33
    assert set(total.to_dict()) == {
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"
    }


# -- credential discovery ------------------------------------------------


def test_extract_reads_shell_assignments():
    script = (
        "#!/bin/bash\n"
        "export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic\n"
        'export ANTHROPIC_AUTH_TOKEN="tok-123"\n'
        "export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-4.6\n"
    )
    assert _extract(script, "ANTHROPIC_BASE_URL") == "https://api.z.ai/api/anthropic"
    assert _extract(script, "ANTHROPIC_AUTH_TOKEN") == "tok-123"
    assert _extract(script, "ANTHROPIC_DEFAULT_SONNET_MODEL") == "glm-4.6"
    assert _extract(script, "NOT_PRESENT") == ""


def test_anthropic_key_discovered(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    found = [c for c in discover_credentials() if c.provider == "anthropic"]
    assert found and found[0].api_key == "sk-ant-test"
    assert "anthropic.com" in found[0].base_url


def test_compatible_gateway_is_classified_as_zai(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-abc")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "glm-4.6")
    found = [c for c in discover_credentials() if c.provider == "zai"]
    assert found and found[0].model == "glm-4.6"
    assert found[0].api_key == "tok-abc"


def test_anthropic_key_with_foreign_base_url_is_not_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "tok")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
    assert not [c for c in discover_credentials() if c.provider == "anthropic"]


def test_zai_named_key_discovered(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "glm-key")
    found = [c for c in discover_credentials() if c.provider == "zai"]
    assert found and found[0].api_key == "glm-key"


def test_openai_and_openrouter_discovered(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    providers = {c.provider for c in discover_credentials()}
    assert {"openai", "openrouter"} <= providers


def test_credential_describe_never_shows_the_key():
    cred = Credential("zai", "supersecret", "https://x", "glm-4.6", "env")
    assert "supersecret" not in cred.describe()
    assert "zai" in cred.describe() and "glm-4.6" in cred.describe()


def test_unknown_provider_rejected():
    with pytest.raises(ProviderError, match="unknown provider"):
        build_provider(ProviderConfig(name="pigeon-post"))


def test_explicit_provider_without_a_key_fails_clearly():
    with pytest.raises(ProviderError, match="no API key"):
        build_provider(ProviderConfig(name="openai", model="gpt-4o"))


def test_explicit_config_beats_discovery(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    provider = build_provider(
        ProviderConfig(name="anthropic", model="explicit-model", api_key="explicit-key")
    )
    try:
        assert provider.model == "explicit-model"
    finally:
        provider.close()
