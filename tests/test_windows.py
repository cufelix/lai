"""Tests for lai.osl.windows.

Pure tests build WindowInfo directly. The @pytest.mark.x11 tests only read the
live window tree via EWMH — they never focus, move, resize or close any window
that they didn't create themselves (they create none).
"""

from __future__ import annotations

import pytest

from lai.errors import WindowNotFound
from lai.osl.geometry import Point, Rect
from lai.osl.windows import WindowInfo, WindowManager


def _make_window(**overrides) -> WindowInfo:
    defaults = dict(
        id=291,  # 0x123
        title="My Window - App",
        wm_class="MyApp",
        instance="myapp",
        pid=1234,
        bounds=Rect(0, 0, 800, 600),
        desktop=0,
        states=(),
        active=False,
    )
    defaults.update(overrides)
    return WindowInfo(**defaults)


# -- WindowInfo.matches ------------------------------------------------------


def test_matches_by_title_substring():
    w = _make_window(title="Untitled - GIMP")
    assert w.matches("gimp") is True
    assert w.matches("GIMP") is True


def test_matches_by_wm_class():
    w = _make_window(wm_class="Gnome-calculator")
    assert w.matches("calculator") is True


def test_matches_by_instance():
    w = _make_window(instance="gnome-calculator")
    assert w.matches("gnome-calculator") is True


def test_matches_by_hex_id():
    w = _make_window(id=291)  # -> 0x00000123
    assert w.matches("0x00000123") is True


def test_matches_by_decimal_id():
    w = _make_window(id=291)
    assert w.matches("291") is True


def test_matches_is_case_insensitive():
    w = _make_window(title="MixedCase Title", wm_class="MixedClass")
    assert w.matches("mixedcase title") is True
    assert w.matches("MIXEDCLASS") is True


def test_matches_empty_query_is_false():
    w = _make_window(title="Anything")
    assert w.matches("") is False
    assert w.matches("   ") is False


def test_matches_no_match_returns_false():
    w = _make_window(title="Firefox", wm_class="Firefox", instance="firefox")
    assert w.matches("nonexistent-app-xyz") is False


# -- WindowInfo.hex_id ------------------------------------------------------


@pytest.mark.parametrize(
    "window_id,expected",
    [
        (0, "0x00000000"),
        (291, "0x00000123"),
        (4294967295, "0xffffffff"),
    ],
)
def test_hex_id_formatting(window_id, expected):
    w = _make_window(id=window_id)
    assert w.hex_id == expected


# -- WindowInfo.visible ------------------------------------------------------


def test_visible_true_when_no_hidden_state_and_positive_area():
    w = _make_window(states=(), bounds=Rect(0, 0, 100, 100))
    assert w.visible is True


def test_visible_false_when_hidden_state_present():
    w = _make_window(states=("hidden",), bounds=Rect(0, 0, 100, 100))
    assert w.visible is False


def test_visible_false_when_area_is_zero():
    w = _make_window(states=(), bounds=Rect(0, 0, 0, 100))
    assert w.visible is False


def test_visible_true_with_other_non_hidden_states():
    w = _make_window(states=("above", "sticky"), bounds=Rect(0, 0, 10, 10))
    assert w.visible is True


# -- WindowInfo.to_dict ------------------------------------------------------


def test_to_dict_shape():
    w = _make_window(
        id=291,
        title="Title",
        wm_class="App",
        instance="app",
        pid=42,
        bounds=Rect(1, 2, 3, 4),
        desktop=1,
        states=("above",),
        active=True,
    )
    data = w.to_dict()
    assert data == {
        "id": 291,
        "hex_id": "0x00000123",
        "title": "Title",
        "app": "App",
        "instance": "app",
        "pid": 42,
        "bounds": {"x": 1, "y": 2, "width": 3, "height": 4},
        "desktop": 1,
        "states": ["above"],
        "active": True,
        "visible": True,
    }


# -- x11: live window tree (read-only) ------------------------------------------------------


@pytest.fixture
def window_manager():
    wm = WindowManager()
    try:
        yield wm
    finally:
        wm.close()


@pytest.mark.x11
def test_list_windows_non_empty(window_manager):
    windows = window_manager.list_windows()
    assert len(windows) > 0


@pytest.mark.x11
def test_every_window_has_positive_area_rect(window_manager):
    windows = window_manager.list_windows()
    assert len(windows) > 0
    for w in windows:
        assert isinstance(w.bounds, Rect)
        assert w.bounds.area > 0


@pytest.mark.x11
def test_active_window_returns_window_info_or_none(window_manager):
    active = window_manager.active_window()
    assert active is None or isinstance(active, WindowInfo)


@pytest.mark.x11
def test_get_bogus_id_raises_window_not_found(window_manager):
    with pytest.raises(WindowNotFound):
        window_manager.get(999999999)


@pytest.mark.x11
def test_find_nonsense_query_returns_empty_list(window_manager):
    assert window_manager.find("zzzz-nonsense-window-query-12345") == []


@pytest.mark.x11
def test_find_one_nonsense_query_raises_window_not_found(window_manager):
    with pytest.raises(WindowNotFound):
        window_manager.find_one("zzzz-nonsense-window-query-12345")


@pytest.mark.x11
def test_window_at_point_inside_known_window_returns_a_window(window_manager):
    windows = window_manager.list_windows()
    assert len(windows) > 0
    target = windows[0]
    point = target.bounds.center
    result = window_manager.window_at(Point(point.x, point.y))
    assert result is not None
    assert isinstance(result, WindowInfo)
    # The point we chose is inside `target`'s own bounds by construction.
    assert result.bounds.contains(point)


# -- connection lifecycle ------------------------------------------------
# Regression guard: `close(self)` (the X11 connection) was once shadowed by a
# `close(self, window_id)` defined later in the same class body, which made
# Desktop.close() raise TypeError. The window-closing overload is now
# `close_window`.


@pytest.mark.x11
def test_window_manager_close_releases_display_connection_cleanly():
    wm = WindowManager()
    wm.list_windows()  # force a connection to open
    wm.close()  # should close cleanly with no arguments
    assert wm._display is None
