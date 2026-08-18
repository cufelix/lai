"""Desktop tools against the real display.

Every test here drives only a window this module launched itself. The fixture
always terminates it, so a failing test cannot leave a stray window behind.
"""

from __future__ import annotations

import time

import pytest

from lai.config import Config
from lai.osl.desktop import Desktop
from lai.tools import build_registry
from lai.tools.base import ToolContext

pytestmark = [pytest.mark.x11, pytest.mark.slow]

TEST_APP = "Calculator"


@pytest.fixture(scope="module")
def registry():
    return build_registry()


@pytest.fixture(scope="module")
def desktop():
    instance = Desktop()
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture(scope="module")
def app(desktop, registry):
    """Launch the test application; always clean it up."""
    ctx = ToolContext(desktop=desktop, config=Config())
    result = registry.call("app_open", {"name": TEST_APP, "timeout": 30}, ctx)
    window = (result.data or {}).get("window")
    if not result.ok or not window:
        pytest.skip(f"could not launch {TEST_APP}: {result.content[:200]}")
    desktop.wait_settle(timeout=4)
    try:
        yield {"ctx": ctx, "window": window, "pid": window["pid"]}
    finally:
        try:
            desktop.apps.terminate(int(window["pid"]))
        except Exception:
            pass


# -- app tools -----------------------------------------------------------


def test_app_list_finds_installed_applications(desktop, registry):
    ctx = ToolContext(desktop=desktop, config=Config())
    result = registry.call("app_list", {"limit": 20}, ctx)
    assert result.ok and result.data["apps"]
    assert all(entry["name"] for entry in result.data["apps"])


def test_app_list_query_filters(desktop, registry):
    ctx = ToolContext(desktop=desktop, config=Config())
    result = registry.call("app_list", {"query": "calcul"}, ctx)
    assert result.ok
    assert any("alcul" in entry["name"].lower() for entry in result.data["apps"])


def test_app_list_with_no_match(desktop, registry):
    ctx = ToolContext(desktop=desktop, config=Config())
    result = registry.call("app_list", {"query": "zzz-not-an-app-zzz"}, ctx)
    assert "No application" in result.content


def test_app_open_produced_a_real_window(app):
    window = app["window"]
    assert "alculator" in f"{window['app']}{window['title']}".lower()
    assert window["bounds"]["width"] > 0 and window["bounds"]["height"] > 0
    assert window["pid"]


def test_app_close_requires_a_target(desktop, registry):
    ctx = ToolContext(desktop=desktop, config=Config())
    assert registry.call("app_close", {}, ctx).ok is False


# -- window tools --------------------------------------------------------


def test_window_list_includes_our_window(app, registry):
    result = registry.call("window_list", {}, app["ctx"])
    assert result.ok
    ids = {entry["id"] for entry in result.data["windows"]}
    assert app["window"]["id"] in ids


def test_window_list_match_filter(app, registry):
    result = registry.call("window_list", {"match": "alculator"}, app["ctx"])
    assert result.ok and result.data["windows"]


def test_window_list_with_no_match(app, registry):
    result = registry.call("window_list", {"match": "zzz-nothing-zzz"}, app["ctx"])
    assert "No matching windows" in result.content


def test_window_focus_by_id(app, registry):
    result = registry.call("window_focus", {"id": app["window"]["id"]}, app["ctx"])
    assert result.ok
    assert "alculator" in result.content.lower()


def test_window_focus_unknown_id_fails_cleanly(app, registry):
    result = registry.call("window_focus", {"id": 999_999_999}, app["ctx"])
    assert result.ok is False


def test_window_arrange_moves_and_resizes(app, registry, desktop):
    """It must report the bounds the window actually has, not the ones asked for.

    A window has a minimum size and a window manager has opinions, so the
    request is a request. Reporting success with the requested geometry is how
    an agent ends up clicking at coordinates that were never on screen.
    """
    target = {"x": 120, "y": 140, "width": 600, "height": 480}
    result = registry.call(
        "window_arrange", {"id": app["window"]["id"], "bounds": target}, app["ctx"]
    )
    assert result.ok
    time.sleep(0.4)
    moved = desktop.windows.get(app["window"]["id"])
    reported = result.data["bounds"]

    assert abs(reported["width"] - moved.bounds.width) <= 8, "the report must match reality"
    assert abs(reported["height"] - moved.bounds.height) <= 8

    refused = abs(moved.bounds.height - target["height"]) > 8
    if refused:
        assert "not what was asked for" in result.content
        assert "height" in result.content


def test_window_arrange_needs_a_state_or_bounds(app, registry):
    result = registry.call("window_arrange", {"id": app["window"]["id"]}, app["ctx"])
    assert result.ok is False and "state" in result.content


def test_window_arrange_rejects_incomplete_bounds(app, registry):
    result = registry.call(
        "window_arrange", {"id": app["window"]["id"], "bounds": {"x": 1}}, app["ctx"]
    )
    assert result.ok is False


def test_window_arrange_maximize_then_restore(app, registry, desktop):
    assert registry.call(
        "window_arrange", {"id": app["window"]["id"], "state": "maximized"}, app["ctx"]
    ).ok
    time.sleep(0.4)
    assert "maximized_horz" in desktop.windows.get(app["window"]["id"]).states
    assert registry.call(
        "window_arrange", {"id": app["window"]["id"], "state": "restore"}, app["ctx"]
    ).ok


# -- computer tools ------------------------------------------------------


def test_screenshot_of_the_whole_desktop(app, registry):
    result = registry.call("computer_screenshot", {"scope": "desktop"}, app["ctx"])
    assert result.ok and len(result.images) == 1
    assert result.images[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert result.data["scale"] <= 1.0
    assert "maps to screen" in result.content


def test_screenshot_of_the_focused_window(app, registry):
    registry.call("window_focus", {"id": app["window"]["id"]}, app["ctx"])
    result = registry.call("computer_screenshot", {"scope": "focused"}, app["ctx"])
    assert result.ok and result.images


def test_screenshot_of_an_explicit_region(app, registry):
    region = {"x": 0, "y": 0, "width": 300, "height": 200}
    result = registry.call("computer_screenshot", {"scope": "region", "region": region}, app["ctx"])
    assert result.ok
    assert result.data["region"] == region


def test_screenshot_region_needs_full_geometry(app, registry):
    result = registry.call("computer_screenshot", {"scope": "region", "region": {"x": 0}}, app["ctx"])
    assert result.ok is False


def test_screenshot_of_a_monitor_and_a_bad_index(app, registry):
    assert registry.call("computer_screenshot", {"scope": "monitor", "monitor": 0}, app["ctx"]).ok
    bad = registry.call("computer_screenshot", {"scope": "monitor", "monitor": 99}, app["ctx"])
    assert bad.ok is False and "out of range" in bad.content


def test_screenshot_max_edge_is_honoured(app, registry):
    result = registry.call(
        "computer_screenshot", {"scope": "desktop", "max_edge": 640}, app["ctx"]
    )
    assert max(result.data["image_size"].values()) <= 640


def test_cursor_reports_a_position(app, registry):
    result = registry.call("computer_cursor", {}, app["ctx"])
    assert result.ok
    assert isinstance(result.data["x"], int) and isinstance(result.data["y"], int)


def test_reported_bounds_are_on_screen(app, desktop):
    """GTK CSD windows must report their visible rectangle, not the shadow box."""
    bounds = desktop.windows.get(app["window"]["id"]).bounds
    screen = desktop.screen.virtual_bounds()
    assert bounds.x >= screen.x and bounds.y >= screen.y
    assert bounds.width > 0 and bounds.height > 0
    assert screen.contains(bounds.center)


def test_move_and_click_within_our_window(app, registry, desktop):
    registry.call("window_focus", {"id": app["window"]["id"]}, app["ctx"])
    bounds = desktop.windows.get(app["window"]["id"]).bounds
    # Aim at the window's centre — inside our own app, harmless.
    x, y = bounds.center.x, bounds.center.y
    assert registry.call("computer_move", {"x": x, "y": y}, app["ctx"]).ok
    position = registry.call("computer_cursor", {}, app["ctx"])
    assert abs(position.data["x"] - x) <= 2 and abs(position.data["y"] - y) <= 2
    under = position.data.get("window")
    assert under is None or under["id"] == app["window"]["id"]


def test_click_off_screen_is_rejected(app, registry):
    result = registry.call("computer_click", {"x": 99_999, "y": 99_999}, app["ctx"])
    assert result.ok is False and "outside the screen" in result.content


def test_invalid_button_is_rejected(app, registry):
    result = registry.call("computer_click", {"x": 10, "y": 10, "button": "pinky"}, app["ctx"])
    assert result.ok is False


def test_scroll_requires_a_valid_direction(app, registry):
    assert registry.call("computer_scroll", {"direction": "sideways"}, app["ctx"]).ok is False


def test_key_normalisation_is_reported(app, registry):
    registry.call("window_focus", {"id": app["window"]["id"]}, app["ctx"])
    result = registry.call("computer_key", {"key": "Escape"}, app["ctx"])
    assert result.ok and result.data["key"] == "Escape"


# -- ui tools ------------------------------------------------------------


def test_ui_snapshot_reads_the_accessibility_tree(app, registry):
    registry.call("window_focus", {"id": app["window"]["id"]}, app["ctx"])
    result = registry.call("ui_snapshot", {"max_elements": 200}, app["ctx"])
    assert result.ok
    if result.data["count"] == 0:
        pytest.skip("this build of the app publishes no a11y tree")
    assert result.data["count"] > 5
    assert "push button" in result.content


def test_ui_find_locates_a_named_button(app, registry):
    result = registry.call("ui_find", {"query": "7", "role": "push button"}, app["ctx"])
    if "No element" in result.content:
        pytest.skip("a11y tree unavailable for this app")
    assert result.ok and result.data["matches"]
    assert result.data["matches"][0]["bounds"]["width"] > 0


def test_ui_find_reports_available_roles_when_nothing_matches(app, registry):
    result = registry.call("ui_find", {"query": "zzz-no-such-element"}, app["ctx"])
    assert "No element matching" in result.content


def test_ui_click_by_name_uses_the_accessible_action(app, registry):
    registry.call("window_focus", {"id": app["window"]["id"]}, app["ctx"])
    snapshot = registry.call("ui_snapshot", {"max_elements": 200}, app["ctx"])
    if snapshot.data["count"] == 0:
        pytest.skip("a11y tree unavailable for this app")
    result = registry.call("ui_click", {"name": "7", "role": "push button"}, app["ctx"])
    assert result.ok
    assert result.data["method"] in ("a11y_action", "pointer")


def test_ui_click_needs_a_ref_or_a_name(app, registry):
    result = registry.call("ui_click", {}, app["ctx"])
    assert result.ok is False and "ref" in result.content


def test_ui_click_unknown_name_fails_cleanly(app, registry):
    result = registry.call("ui_click", {"name": "zzz-not-a-button"}, app["ctx"])
    assert result.ok is False


def test_ui_read_on_an_element_without_text(app, registry):
    snapshot = registry.call("ui_snapshot", {"max_elements": 200}, app["ctx"])
    if snapshot.data["count"] == 0:
        pytest.skip("a11y tree unavailable for this app")
    result = registry.call("ui_read", {"name": "7", "role": "push button"}, app["ctx"])
    assert result.ok  # either text or an explanatory message, but never a crash


def test_ui_wait_for_times_out_cleanly(app, registry):
    result = registry.call("ui_wait_for", {"name": "zzz-never-appears", "timeout": 1}, app["ctx"])
    assert result.ok is False


# -- core tools ----------------------------------------------------------


def test_desktop_observe_returns_a_summary_and_an_image(app, registry):
    result = registry.call("desktop_observe", {"scope": "focused"}, app["ctx"])
    assert result.ok and "FOCUSED" in result.content
    assert len(result.images) == 1


def test_desktop_observe_without_a_screenshot(app, registry):
    result = registry.call("desktop_observe", {"screenshot": False}, app["ctx"])
    assert result.ok and result.images == []


def test_desktop_observe_can_annotate(app, registry):
    result = registry.call(
        "desktop_observe", {"scope": "focused", "annotate": True}, app["ctx"]
    )
    assert result.ok


def test_desktop_wait_settles(app, registry):
    result = registry.call("desktop_wait", {"timeout": 2}, app["ctx"])
    assert result.ok and "settled" in result.data


def test_desktop_wait_fixed_sleep(app, registry):
    started = time.monotonic()
    result = registry.call("desktop_wait", {"seconds": 0.3}, app["ctx"])
    assert result.ok and time.monotonic() - started >= 0.25


# -- clipboard -----------------------------------------------------------


def test_clipboard_roundtrip(app, registry):
    text = "LAI clipboard test ✓ ěščř"
    write = registry.call("clipboard_write", {"text": text}, app["ctx"])
    assert write.ok
    read = registry.call("clipboard_read", {}, app["ctx"])
    assert read.content == text


def test_clipboard_primary_selection(app, registry):
    write = registry.call(
        "clipboard_write", {"text": "primary value", "selection": "primary"}, app["ctx"]
    )
    assert write.ok
    read = registry.call("clipboard_read", {"selection": "primary"}, app["ctx"])
    assert "primary value" in read.content
