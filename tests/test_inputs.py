"""Tests for lai.osl.inputs.

normalize_key() and validation are pure logic and never touch xdotool. The
InputController validation tests use a monkeypatched ``_run`` that raises if
called at all, proving that every validation error is raised before any
subprocess would be spawned.
"""

from __future__ import annotations

import pytest

from lai.errors import InvalidToolInput
from lai.osl.inputs import BUTTONS, InputController, normalize_key

# -- normalize_key ------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("enter", "Return"),
        ("Enter", "Return"),
        ("return", "Return"),
        ("esc", "Escape"),
        ("escape", "Escape"),
        ("cmd", "super"),
        ("command", "super"),
        ("win", "super"),
        ("meta", "super"),
    ],
)
def test_normalize_key_aliases(spec, expected):
    assert normalize_key(spec) == expected


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("ENTER", "Return"),
        ("Esc", "Escape"),
        ("CMD", "super"),
        ("Tab", "Tab"),
        ("TAB", "Tab"),
    ],
)
def test_normalize_key_case_insensitive(spec, expected):
    assert normalize_key(spec) == expected


def test_normalize_key_chord():
    assert normalize_key("Ctrl+Shift+t") == "ctrl+shift+t"


def test_normalize_key_chord_with_spaces_stripped():
    assert normalize_key("Ctrl + Alt + Delete") == "ctrl+alt+Delete"


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("f5", "F5"),
        ("F5", "F5"),
        ("F12", "F12"),
        ("f1", "F1"),
    ],
)
def test_normalize_key_function_keys(spec, expected):
    assert normalize_key(spec) == expected


def test_normalize_key_passthrough_unknown_keysym():
    # Not an alias, not a single char, not an F-key -> passed through verbatim.
    assert normalize_key("Num_Lock") == "Num_Lock"


@pytest.mark.parametrize("spec", ["", "   ", "\t"])
def test_normalize_key_empty_or_whitespace_raises(spec):
    with pytest.raises(InvalidToolInput):
        normalize_key(spec)


# -- InputController._button_code ------------------------------------------------------


def test_button_code_valid_buttons():
    assert InputController._button_code("left") == BUTTONS["left"]
    assert InputController._button_code("Right") == BUTTONS["right"]
    assert InputController._button_code("WHEEL_UP") == BUTTONS["wheel_up"]


def test_button_code_invalid_raises_invalid_tool_input():
    with pytest.raises(InvalidToolInput):
        InputController._button_code("not-a-button")


# -- validation happens before any xdotool subprocess ------------------------------------------------------


@pytest.fixture
def guarded_controller():
    """An InputController whose ``_run`` fails the test if it's ever invoked."""
    controller = InputController()

    def _boom(*args, **kwargs):
        raise AssertionError(f"_run should not have been called, got args={args!r}")

    controller._run = _boom  # type: ignore[method-assign]
    return controller


def test_click_count_too_low_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.click(count=0)


def test_click_count_too_high_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.click(count=6)


def test_click_bad_button_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.click(button="not-a-button")


def test_scroll_amount_too_low_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.scroll(amount=0)


def test_scroll_amount_too_high_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.scroll(amount=51)


def test_scroll_bad_direction_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.scroll(direction="diagonal")


def test_type_text_non_str_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.type_text(12345)  # type: ignore[arg-type]


def test_type_text_empty_string_returns_zero_length_without_calling_run(guarded_controller):
    result = guarded_controller.type_text("")
    assert result.action == "type"
    assert result.detail == {"length": 0}


def test_key_count_too_low_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.key("a", count=0)


def test_key_count_too_high_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.key("a", count=51)


def test_key_bad_spec_raises_before_run(guarded_controller):
    with pytest.raises(InvalidToolInput):
        guarded_controller.key("   ")


def test_move_never_uses_the_hanging_sync_flag(monkeypatch):
    """Regression: `xdotool mousemove --sync` blocks until timeout when the
    pointer is already at the target, turning a 5 ms action into a 10 s failure."""
    from lai.osl.geometry import Point
    from lai.osl.inputs import InputController

    controller = InputController()
    seen: list[tuple[str, ...]] = []

    def fake_run(*args: str) -> str:
        seen.append(args)
        if args[0] == "getmouselocation":
            return "X=100\nY=200\nSCREEN=0\nWINDOW=1"
        return ""

    monkeypatch.setattr(controller, "_run", fake_run)
    result = controller.move(Point(100, 200))
    move_args = next(a for a in seen if a[0] == "mousemove")
    assert "--sync" not in move_args
    assert result.to_dict()["x"] == 100


def test_move_reports_where_the_pointer_actually_landed(monkeypatch):
    from lai.osl.geometry import Point
    from lai.osl.inputs import InputController

    controller = InputController()

    def fake_run(*args: str) -> str:
        if args[0] == "getmouselocation":
            return "X=5\nY=5\nSCREEN=0\nWINDOW=1"  # clamped somewhere else
        return ""

    monkeypatch.setattr(controller, "_run", fake_run)
    data = controller.move(Point(900, 900)).to_dict()
    assert data["x"] == 5 and data["requested"] == {"x": 900, "y": 900}


def test_move_can_skip_verification(monkeypatch):
    from lai.osl.geometry import Point
    from lai.osl.inputs import InputController

    controller = InputController()
    calls: list[str] = []
    monkeypatch.setattr(controller, "_run", lambda *args: calls.append(args[0]) or "")
    controller.move(Point(1, 2), verify=False)
    assert calls == ["mousemove"], "verification should cost nothing when disabled"


# -- a display that xdotool cannot open -----------------------------------


def test_a_missing_display_is_named_rather_than_quoted_back(monkeypatch):
    """xdotool says "Can't open display: (null)", which tells the model that
    something is null. Five of those in this machine's logs, and none of them
    said the one thing that would have helped: DISPLAY is not set."""
    import subprocess

    from lai.errors import BackendUnavailable
    from lai.osl.inputs import InputController

    def dead(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Error: Can't open display: (null)\nFailed creating new xdo instance"
        )

    controller = InputController()
    monkeypatch.setattr(controller, "_xdotool", "/usr/bin/xdotool")
    monkeypatch.setattr(subprocess, "run", dead)
    monkeypatch.delenv("DISPLAY", raising=False)

    with pytest.raises(BackendUnavailable, match="DISPLAY is not set"):
        controller._run("key", "a")


def test_a_display_that_stopped_answering_says_which_one(monkeypatch):
    import subprocess

    from lai.errors import BackendUnavailable
    from lai.osl.inputs import InputController

    def dead(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Error: Can't open display: :91")

    controller = InputController()
    monkeypatch.setattr(controller, "_xdotool", "/usr/bin/xdotool")
    monkeypatch.setattr(subprocess, "run", dead)
    monkeypatch.setenv("DISPLAY", ":91")

    with pytest.raises(BackendUnavailable) as raised:
        controller._run("key", "a")
    assert ":91" in str(raised.value) + str(raised.value.detail)


def test_an_ordinary_xdotool_failure_is_still_reported_as_itself(monkeypatch):
    import subprocess

    from lai.errors import BackendUnavailable
    from lai.osl.inputs import InputController

    def refused(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unknown key name: qq")

    controller = InputController()
    monkeypatch.setattr(controller, "_xdotool", "/usr/bin/xdotool")
    monkeypatch.setattr(subprocess, "run", refused)
    monkeypatch.setenv("DISPLAY", ":0")

    with pytest.raises(BackendUnavailable, match="unknown key name"):
        controller._run("key", "qq")
