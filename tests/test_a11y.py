"""Tests for lai.osl.a11y.

Pure logic tests build Element/Snapshot objects by hand — no AT-SPI bus needed.
The @pytest.mark.x11 tests only read the live AT-SPI registry (applications())
and never act on anything.
"""

from __future__ import annotations

import pytest

from lai.errors import ElementNotFound
from lai.osl.a11y import A11yTree, Element, Snapshot
from lai.osl.geometry import Point, Rect


def _make_element(**overrides) -> Element:
    defaults = dict(
        ref=1,
        role="push button",
        name="Save",
        bounds=Rect(10, 10, 40, 20),
        states=("enabled", "sensitive"),
        actions=("click",),
        description="",
        value=None,
        path=(),
        app="TestApp",
        pid=1234,
        depth=0,
    )
    defaults.update(overrides)
    return Element(**defaults)


# -- Element.interactive ------------------------------------------------------


def test_interactive_true_for_known_interactive_role():
    e = _make_element(role="push button", actions=())
    assert e.interactive is True


def test_interactive_true_when_has_actions_even_if_role_is_not_interactive():
    e = _make_element(role="label", actions=("click",))
    assert e.interactive is True


def test_interactive_false_for_non_interactive_role_and_no_actions():
    e = _make_element(role="label", actions=())
    assert e.interactive is False


# -- Element.enabled ------------------------------------------------------


def test_enabled_true_when_enabled_state_present():
    e = _make_element(states=("enabled",))
    assert e.enabled is True


def test_enabled_true_when_sensitive_state_present():
    e = _make_element(states=("sensitive",))
    assert e.enabled is True


def test_enabled_false_when_neither_state_present():
    e = _make_element(states=("focused", "visible"))
    assert e.enabled is False


def test_enabled_false_when_no_states():
    e = _make_element(states=())
    assert e.enabled is False


# -- Element.center ------------------------------------------------------


def test_center_matches_bounds_center():
    e = _make_element(bounds=Rect(0, 0, 10, 10))
    assert e.center == Point(5, 5)


# -- Element.to_line ------------------------------------------------------


def test_to_line_basic_format():
    e = _make_element(ref=7, role="push button", name="Save", bounds=Rect(10, 10, 20, 20), states=("enabled",))
    line = e.to_line()
    assert line.startswith("[7] push button")
    assert '"Save"' in line
    assert "@20,20" in line  # center of Rect(10,10,20,20) is (20,20)


def test_to_line_includes_value_when_present():
    e = _make_element(role="entry", name="", value="hello world")
    line = e.to_line()
    assert "value='hello world'" in line


def test_to_line_includes_state_flags():
    e = _make_element(states=("enabled", "sensitive", "focused", "checked"))
    line = e.to_line()
    assert "(focused,checked)" in line


def test_to_line_includes_disabled_flag_when_not_enabled():
    e = _make_element(states=())
    line = e.to_line()
    assert "disabled" in line


def test_to_line_no_name_omits_quotes():
    e = _make_element(name="")
    line = e.to_line()
    assert '""' not in line


# -- Element.matches ------------------------------------------------------


def test_matches_role_filter_case_insensitive():
    e = _make_element(role="push button", name="Save")
    assert e.matches("", role="Push Button") is True
    assert e.matches("", role="push button") is True


def test_matches_role_filter_mismatch_returns_false():
    e = _make_element(role="push button", name="Save")
    assert e.matches("", role="label") is False


def test_matches_empty_query_with_matching_role_is_true():
    e = _make_element(role="push button")
    assert e.matches("", role="push button") is True


def test_matches_by_name_substring():
    e = _make_element(name="Save As...")
    assert e.matches("save") is True


def test_matches_by_description():
    e = _make_element(name="", description="Saves the current document")
    assert e.matches("saves the") is True


def test_matches_by_value():
    e = _make_element(name="", value="42")
    assert e.matches("42") is True


def test_matches_no_match_returns_false():
    e = _make_element(name="Save", description="", value=None)
    assert e.matches("zzzznotfound") is False


# -- Snapshot.get / handle ------------------------------------------------------


def test_snapshot_get_raises_for_unknown_ref():
    snapshot = Snapshot(elements=[_make_element(ref=1)])
    with pytest.raises(ElementNotFound):
        snapshot.get(999)


def test_snapshot_get_returns_matching_element():
    e = _make_element(ref=5)
    snapshot = Snapshot(elements=[e])
    assert snapshot.get(5) is e


def test_snapshot_handle_raises_for_unknown_ref():
    snapshot = Snapshot(elements=[_make_element(ref=1)])
    with pytest.raises(ElementNotFound):
        snapshot.handle(1)  # no live handle registered for this hand-built snapshot


def test_snapshot_handle_returns_registered_handle():
    snapshot = Snapshot(elements=[_make_element(ref=1)])
    sentinel = object()
    snapshot._handles[1] = sentinel
    assert snapshot.handle(1) is sentinel


def test_snapshot_len_and_iter():
    elements = [_make_element(ref=1), _make_element(ref=2)]
    snapshot = Snapshot(elements=elements)
    assert len(snapshot) == 2
    assert list(snapshot) == elements


# -- Snapshot.find / find_one ------------------------------------------------------


def test_find_filters_by_query_and_role():
    a = _make_element(ref=1, role="push button", name="Save")
    b = _make_element(ref=2, role="label", name="Save state")
    snapshot = Snapshot(elements=[a, b])
    assert snapshot.find("save", role="push button") == [a]


def test_find_interactive_filter():
    interactive = _make_element(ref=1, role="push button", name="Save", actions=("click",))
    static = _make_element(ref=2, role="label", name="Save", actions=())
    snapshot = Snapshot(elements=[interactive, static])
    assert snapshot.find("save", interactive=True) == [interactive]
    assert snapshot.find("save", interactive=False) == [static]


def test_find_one_prefers_exact_name_match():
    save_as = _make_element(ref=1, role="push button", name="Save As", bounds=Rect(0, 0, 5, 5))
    save = _make_element(ref=2, role="push button", name="Save", bounds=Rect(0, 0, 1000, 1000))
    snapshot = Snapshot(elements=[save_as, save])
    result = snapshot.find_one("Save")
    assert result.ref == 2  # exact name wins even though it's the larger element


def test_find_one_prefers_interactive_over_smaller_non_interactive():
    interactive = _make_element(
        ref=1, role="push button", name="Save", actions=("click",), bounds=Rect(0, 0, 100, 100)
    )
    non_interactive_tiny = _make_element(
        ref=2, role="label", name="Save", actions=(), bounds=Rect(0, 0, 1, 1)
    )
    snapshot = Snapshot(elements=[non_interactive_tiny, interactive])
    result = snapshot.find_one("Save")
    assert result.ref == 1


def test_find_one_smallest_area_tiebreak_among_interactive():
    big = _make_element(ref=1, role="push button", name="Save", bounds=Rect(0, 0, 200, 200))
    small = _make_element(ref=2, role="push button", name="Save", bounds=Rect(0, 0, 10, 10))
    snapshot = Snapshot(elements=[big, small])
    result = snapshot.find_one("Save")
    assert result.ref == 2


def test_find_one_raises_when_nothing_matches():
    snapshot = Snapshot(elements=[_make_element(name="Save")])
    with pytest.raises(ElementNotFound):
        snapshot.find_one("zzzznotfound")


# -- Snapshot.render ------------------------------------------------------


def test_render_empty_snapshot_returns_no_elements_message():
    snapshot = Snapshot(elements=[])
    result = snapshot.render()
    assert "no accessible elements" in result


def test_render_respects_limit_and_appends_truncation_line():
    elements = [_make_element(ref=i, name=f"Item {i}") for i in range(10)]
    snapshot = Snapshot(elements=elements)
    result = snapshot.render(limit=3)
    lines = result.splitlines()
    assert len(lines) == 4  # 3 element lines + 1 truncation line
    assert "10 elements total, truncated" in lines[-1]


def test_render_no_truncation_line_when_under_limit():
    elements = [_make_element(ref=i, name=f"Item {i}") for i in range(3)]
    snapshot = Snapshot(elements=elements)
    result = snapshot.render(limit=10)
    assert "truncated" not in result


def test_render_interactive_only():
    interactive = _make_element(ref=1, role="push button", name="Click me")
    static = _make_element(ref=2, role="label", name="Just text", actions=())
    snapshot = Snapshot(elements=[interactive, static])
    result = snapshot.render(interactive_only=True)
    assert "Click me" in result
    assert "Just text" not in result


# -- Snapshot.to_dict ------------------------------------------------------


def test_to_dict_shape():
    e = _make_element(ref=1)
    snapshot = Snapshot(elements=[e], app="TestApp", truncated=False)
    data = snapshot.to_dict()
    assert data["app"] == "TestApp"
    assert data["count"] == 1
    assert data["truncated"] is False
    assert data["elements"] == [e.to_dict()]


# -- x11: live AT-SPI registry (read-only) ------------------------------------------------------


@pytest.mark.x11
def test_a11y_tree_available_is_true():
    tree = A11yTree()
    assert tree.available is True


@pytest.mark.x11
def test_applications_returns_non_empty_list_of_tuples():
    tree = A11yTree()
    apps = tree.applications()
    assert len(apps) > 0
    for name, pid, accessible in apps:
        assert isinstance(name, str)
        assert pid is None or isinstance(pid, int)
        assert accessible is not None


def test_a_role_filter_that_finds_nothing_says_what_the_roles_actually_are():
    """"no element matching 'Calculator' with role 'text'" is a dead end when
    there *is* a Calculator and its role is something else. The next thing the
    model needs is which role to ask for."""
    from lai.errors import ElementNotFound
    from lai.osl.a11y import Snapshot

    snapshot = Snapshot(
        elements=[
            Element(ref=1, role="frame", name="Calculator", bounds=Rect(0, 0, 10, 10)),
            Element(ref=2, role="push button", name="Calculator", bounds=Rect(0, 0, 5, 5)),
        ],
        app="Calculator",
    )
    try:
        snapshot.find_one("Calculator", role="text")
    except ElementNotFound as exc:
        message = f"{exc} {getattr(exc, 'detail', '')}"
        assert "frame" in message and "push button" in message, message
    else:
        raise AssertionError("expected ElementNotFound")


def test_a_name_that_matches_nothing_at_all_is_reported_plainly():
    from lai.errors import ElementNotFound
    from lai.osl.a11y import Snapshot

    snapshot = Snapshot(
        elements=[Element(ref=1, role="frame", name="Calculator", bounds=Rect(0, 0, 10, 10))],
        app="Calculator",
    )
    try:
        snapshot.find_one("Spreadsheet")
    except ElementNotFound as exc:
        assert "Spreadsheet" in str(exc)
