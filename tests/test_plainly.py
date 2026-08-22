"""What the agent just did, in words anybody can read.

`ui_click {"ref": 114}` is precise for whoever is debugging the tool call, and
useless to the person watching their own computer being used.
"""

from __future__ import annotations

import pytest

from lai.agent.plainly import describe


@pytest.mark.parametrize("name,args,expected", [
    ("app_open", {"name": "Calculator"}, "Opened “Calculator”"),
    ("ui_click", {"name": "3"}, "Clicked “3”"),
    ("computer_key", {"key": "ctrl+l"}, "Pressed ctrl+l"),
    ("ui_type", {"text": "hello"}, "Typed “hello”"),
    ("desktop_wait", {"seconds": 3}, "Waited 3 seconds"),
    ("ocr_read", {"scope": "focused"}, "Read the words on screen"),
    ("window_list", {}, "Checked which windows are open"),
    ("task_complete", {"summary": "x"}, "Finished"),
])
def test_the_common_actions_read_as_sentences(name, args, expected):
    assert describe(name, args) == expected


def test_a_click_with_only_a_ref_does_not_expose_the_ref():
    """A ref is an index into a tree the reader cannot see."""
    said = describe("ui_click", {"ref": 114})
    assert "114" not in said
    assert said == "Clicked an item on screen"


def test_a_long_value_is_cut_rather_than_wrapping_the_line():
    said = describe("ui_type", {"text": "x" * 300})
    assert len(said) < 80 and said.endswith("”")


def test_newlines_do_not_break_the_line():
    assert "\n" not in describe("ui_type", {"text": "one\ntwo"})


def test_a_service_tool_names_the_service():
    assert describe("mcp__github__create_pull_request") == "Used github to create pull request"
    assert describe("mcp__token-optimizer__smart_logs") == "Used token optimizer to smart logs"


def test_an_unknown_tool_is_still_readable():
    assert describe("some_new_tool", {}) == "Some new tool"


def test_no_arguments_at_all_is_fine():
    for name in ("ui_click", "app_open", "ui_type", "desktop_wait", "shell_exec"):
        said = describe(name)
        assert said and said[0].isupper()


def test_every_tool_the_agent_has_a_sentence_written_for_it():
    """The fallback — the tool name with its underscores taken out — is never
    wrong and always reads like a function. A new tool should get a sentence,
    and this is what says so."""
    from lai.agent.plainly import _HANDLERS
    from lai.tools import build_registry

    registry = build_registry()
    missing = [spec.name for spec in registry.specs() if spec.name not in _HANDLERS]
    assert missing == [], f"no plain-language sentence for: {missing}"


def test_the_fallback_is_still_readable():
    assert describe("some_brand_new_tool", {}) == "Some brand new tool"
