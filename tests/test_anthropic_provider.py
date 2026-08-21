

# -- prompt caching ------------------------------------------------------


def _provider(**kwargs):
    from lai.agent.providers.anthropic_api import AnthropicProvider

    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("model", "glm-5")
    kwargs.setdefault("name", "zai")
    return AnthropicProvider(**kwargs)


def _conversation(turns: int = 3):
    from lai.agent.providers.base import Message

    messages = []
    for index in range(turns):
        messages.append(Message.user(f"question {index}"))
        messages.append(Message.assistant(f"answer {index}"))
    return messages[:turns]


TOOLS = [
    {"name": "a", "description": "does a", "input_schema": {"type": "object"}},
    {"name": "b", "description": "does b", "input_schema": {"type": "object"}},
]


def _marked(blocks) -> list[int]:
    return [i for i, block in enumerate(blocks) if "cache_control" in block]


def test_the_unchanging_prefix_is_marked_cacheable():
    """Tools, system and settled history are identical every turn; an agent
    loop re-sends all of it, and that is the largest saving available."""
    payload = _provider()._payload(_conversation(3), "system text", TOOLS)
    assert _marked(payload["tools"]) == [len(TOOLS) - 1], "the last tool caches all of them"
    assert _marked(payload["system"]) == [0]


def test_the_settled_transcript_is_cached_but_not_the_newest_turn():
    """Marking the newest message writes a cache entry nothing will ever read."""
    payload = _provider()._payload(_conversation(4), "system", TOOLS)
    marked = [
        index for index, message in enumerate(payload["messages"])
        if any("cache_control" in block for block in message["content"])
    ]
    assert marked == [len(payload["messages"]) - 2]


def test_a_short_conversation_is_not_worth_a_cache_write():
    payload = _provider()._payload(_conversation(2), "system", TOOLS)
    assert not any(
        "cache_control" in block
        for message in payload["messages"]
        for block in message["content"]
    )


def test_caching_can_be_turned_off():
    payload = _provider(prompt_cache=False)._payload(_conversation(4), "system", TOOLS)
    assert not any("cache_control" in tool for tool in payload["tools"])
    assert not any("cache_control" in block for block in payload["system"])


def test_marking_does_not_mutate_the_caller_s_tools():
    """The registry hands out one list of schemas; scribbling on it would leak."""
    tools = [dict(tool) for tool in TOOLS]
    _provider()._payload(_conversation(3), "system", tools)
    assert not any("cache_control" in tool for tool in tools)


def test_a_request_with_no_tools_or_system_still_works():
    payload = _provider()._payload(_conversation(3), "", None)
    assert "tools" not in payload and "system" not in payload


def test_the_setting_comes_from_config(tmp_path, monkeypatch):
    from lai.config import load_config

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.delenv("LAI_PROMPT_CACHE", raising=False)
    assert load_config().provider.prompt_cache is True
    (tmp_path / "config.toml").write_text(
        "[provider]\nprompt_cache = false\n", encoding="utf-8"
    )
    assert load_config().provider.prompt_cache is False
