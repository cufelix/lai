"""Integration tests for lai.osl.desktop.Desktop.

These are the only tests in the suite that drive a real application. They use
exclusively a GNOME Calculator window that the test itself launches, and the
`calculator` fixture *always* terminates that process in teardown (even on
failure) via `desktop.apps.terminate(pid)`. No pre-existing window (Firefox,
Chrome, terminals, Desktop) is ever touched, moved, focused or closed.
"""

from __future__ import annotations

import pytest

from lai.errors import LaiError
from lai.osl.apps import LaunchResult
from lai.osl.desktop import Desktop, Observation
from lai.osl.geometry import Point

pytestmark = [pytest.mark.x11, pytest.mark.slow]


@pytest.fixture
def calculator(desktop: Desktop):
    """Launch GNOME Calculator and unconditionally terminate it afterwards."""
    result = desktop.open_app("Calculator", timeout=30)
    try:
        yield desktop, result
    finally:
        if result.pid:
            desktop.apps.terminate(result.pid)


def test_launch_returns_launch_result_with_calculator_window(calculator):
    _desktop, result = calculator
    assert isinstance(result, LaunchResult)
    assert result.window is not None, f"calculator window never appeared; note={result.note!r}"
    assert "alculator" in result.window.wm_class


def test_observe_focused_returns_calculator_with_many_elements(calculator):
    desktop, _result = calculator
    obs = desktop.observe(scope="focused")
    assert isinstance(obs, Observation)
    assert obs.active_window is not None
    assert "alculator" in obs.active_window.wm_class
    assert obs.elements is not None
    assert len(obs.elements) > 10


def test_observation_summary_is_nonempty_and_mentions_focused(calculator):
    desktop, _result = calculator
    obs = desktop.observe(scope="focused")
    summary = obs.summary()
    assert isinstance(summary, str)
    assert summary.strip() != ""
    assert "FOCUSED" in summary


def test_find_one_seven_button_has_positive_area(calculator):
    desktop, _result = calculator
    obs = desktop.observe(scope="focused")
    assert obs.elements is not None
    element = obs.elements.find_one("7", role="push button")
    assert element.bounds.area > 0


def test_click_element_returns_dict_with_method_key(calculator):
    desktop, _result = calculator
    obs = desktop.observe(scope="focused")
    assert obs.elements is not None
    element = obs.elements.find_one("7", role="push button")
    click_result = desktop.click_element(element.ref)
    assert "method" in click_result


def test_wait_settle_returns_dict_with_settled_key(calculator):
    desktop, _result = calculator
    settled = desktop.wait_settle(timeout=2.0)
    assert "settled" in settled


def test_validate_point_raises_lai_error_for_offscreen_point(desktop: Desktop):
    # Pure validation against screen bounds — doesn't need the calculator running.
    with pytest.raises(LaiError):
        desktop._validate_point(Point(99999, 99999))


# -- known bug: see WindowManager.close in src/lai/osl/windows.py ------------------------------------------------------


def test_desktop_close_does_not_raise():
    instance = Desktop()
    instance.windows.list_windows()  # force the X11 connection open
    instance.close()  # should close cleanly with no arguments
