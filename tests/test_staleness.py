"""References that stopped being true.

`ui_snapshot` hands back numbered elements, and every number belongs to the
snapshot it came from. Open a window, close one, switch focus, press a key that
opens a menu — and every one of them is now pointing at something else, or at
nothing.

The model has no way to know that. It sees a list of refs it was given a moment
ago and uses one, and the run spends a full turn on `element_not_found` before
anybody learns anything. Twelve of those in this machine's logs.
"""

from __future__ import annotations

from lai.agent.staleness import Staleness


def test_a_fresh_snapshot_leaves_refs_usable():
    stale = Staleness()
    stale.record("ui_snapshot", ok=True)
    assert stale.warning("ui_click", {"ref": 3}) == ""


def test_opening_a_window_invalidates_them():
    stale = Staleness()
    stale.record("ui_snapshot", ok=True)
    stale.record("app_open", ok=True)
    warning = stale.warning("ui_click", {"ref": 3})
    assert "ui_snapshot" in warning
    assert "changed" in warning.lower()


def test_the_warning_names_what_changed_it():
    """"Something moved" is not actionable; "you opened an application" is."""
    stale = Staleness()
    stale.record("ui_snapshot", ok=True)
    stale.record("window_focus", ok=True)
    assert "window_focus" in stale.warning("ui_read", {"ref": 9})


def test_it_is_said_once_not_on_every_call():
    stale = Staleness()
    stale.record("ui_snapshot", ok=True)
    stale.record("app_open", ok=True)
    assert stale.warning("ui_click", {"ref": 3}) != ""
    assert stale.warning("ui_click", {"ref": 4}) == ""


def test_a_new_snapshot_clears_it():
    stale = Staleness()
    stale.record("app_open", ok=True)
    stale.record("ui_snapshot", ok=True)
    assert stale.warning("ui_click", {"ref": 3}) == ""


def test_acting_by_name_is_not_affected():
    """A name survives a window moving. A number does not."""
    stale = Staleness()
    stale.record("app_open", ok=True)
    assert stale.warning("ui_click", {"name": "Save"}) == ""


def test_a_tool_that_takes_no_ref_is_not_warned_about():
    stale = Staleness()
    stale.record("app_open", ok=True)
    assert stale.warning("computer_screenshot", {}) == ""
    assert stale.warning("window_list", {}) == ""


def test_a_change_that_failed_did_not_change_anything():
    stale = Staleness()
    stale.record("ui_snapshot", ok=True)
    stale.record("app_open", ok=False)
    assert stale.warning("ui_click", {"ref": 3}) == ""


def test_before_any_snapshot_there_is_nothing_to_go_stale():
    """A ref used before a snapshot is the model's own invention, and the
    element_not_found it earns says so more clearly than a guess would."""
    stale = Staleness()
    assert stale.warning("ui_click", {"ref": 3}) == ""


def test_every_tool_that_moves_the_tree_is_covered():
    from lai.agent.staleness import DISTURBS

    for tool in ("app_open", "app_close", "window_focus", "window_close",
                 "window_arrange", "workspace_switch", "computer_key"):
        assert tool in DISTURBS, tool
