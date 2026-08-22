"""Tool calls a model wrote out as text.

Native function calling is an API feature, and plenty of capable models do not
have it — Hermes, Qwen, most things served straight off Ollama. They write the
call instead. Read as prose it looks like the model claiming it already acted,
and the run stalls with the model insisting the thing is done.
"""

from __future__ import annotations

from lai.agent.providers.toolcall_text import describe, parse

# -- the tagged form -----------------------------------------------------


def test_a_hermes_call_is_read_as_a_call():
    text, calls = parse(
        'Let me look.\n<tool_call>\n{"name": "window_list", "arguments": {}}\n</tool_call>'
    )
    assert text == "Let me look."
    assert [(c.name, c.input) for c in calls] == [("window_list", {})]


def test_several_calls_in_one_reply():
    text, calls = parse(
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    assert text == ""
    assert [c.name for c in calls] == ["a", "b"]
    assert [c.id for c in calls] == ["call_0", "call_1"]


def test_ids_continue_from_the_native_ones():
    """A turn carrying both must not end up with two call_0."""
    _, calls = parse('<tool_call>{"name": "a", "arguments": {}}</tool_call>', start=2)
    assert calls[0].id == "call_2"


def test_a_list_inside_one_element():
    _, calls = parse('<tool_call>[{"name": "a", "arguments": {}}, {"name": "b"}]</tool_call>')
    assert [c.name for c in calls] == ["a", "b"]


def test_arguments_given_as_a_json_string():
    _, calls = parse('<tool_call>{"name": "a", "arguments": "{\\"x\\": 2}"}</tool_call>')
    assert calls[0].input == {"x": 2}


def test_the_nested_shape_some_fine_tunes_use():
    _, calls = parse(
        '<tool_call>{"function": {"name": "a", "arguments": {"x": 1}}}</tool_call>'
    )
    assert [(c.name, c.input) for c in calls] == [("a", {"x": 1})]


# -- other vendors' spellings -------------------------------------------


def test_the_mistral_marker():
    text, calls = parse('[TOOL_CALLS] [{"name": "ui_click", "arguments": {"ref": 3}}]')
    assert text == ""
    assert [(c.name, c.input) for c in calls] == [("ui_click", {"ref": 3})]


def test_the_llama_tag():
    _, calls = parse('<|python_tag|>{"name": "a", "arguments": {}}<|eom_id|>')
    assert [c.name for c in calls] == ["a"]


# -- what must NOT be read as a call ------------------------------------


def test_ordinary_prose_is_left_alone():
    assert parse("I will list the windows next.") == ("I will list the windows next.", [])


def test_a_model_describing_a_call_has_not_made_one():
    """A bare JSON object in prose is a model explaining itself."""
    text, calls = parse('I would call {"name": "window_list", "arguments": {}} here.')
    assert calls == []
    assert "window_list" in text


def test_malformed_json_is_not_a_call():
    text, calls = parse("<tool_call>{not json at all}</tool_call>")
    assert calls == []
    assert "not json" in text, "the text is kept so the model can see its own mistake"


def test_a_call_with_no_name_is_not_a_call():
    _, calls = parse('<tool_call>{"arguments": {"x": 1}}</tool_call>')
    assert calls == []


def test_empty_input():
    assert parse("") == ("", [])


# -- asking for them -----------------------------------------------------


def test_the_prompt_block_carries_the_schemas_and_the_shape():
    block = describe([{"name": "window_list", "description": "list windows"}])
    assert "<tools>" in block and "window_list" in block
    assert "<tool_call>" in block
    assert "did not happen" in block, "a described call is not a call"


def test_no_tools_means_no_block():
    assert describe([]) == ""
