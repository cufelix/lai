"""Exhaustive pure-logic tests for lai.osl.geometry (no display required)."""

from __future__ import annotations

import dataclasses

import pytest

from lai.osl.geometry import Monitor, Point, Rect

# -- Point ----------------------------------------------------------------


def test_point_as_tuple():
    assert Point(3, 4).as_tuple() == (3, 4)


def test_point_offset_positive_and_negative():
    p = Point(10, 20)
    assert p.offset(5, -5) == Point(15, 15)
    assert p.offset(0, 0) == Point(10, 20)
    assert p.offset(-10, -20) == Point(0, 0)


def test_point_is_frozen():
    p = Point(1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.x = 99  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.y = 99  # type: ignore[misc]


# -- Rect: basic geometry ---------------------------------------------------


def test_rect_right_and_bottom():
    r = Rect(10, 20, 30, 40)
    assert r.right == 40
    assert r.bottom == 60


def test_rect_area():
    assert Rect(0, 0, 10, 5).area == 50
    assert Rect(0, 0, 0, 5).area == 0
    assert Rect(0, 0, 10, 0).area == 0


def test_rect_area_negative_dimensions_clamped_to_zero():
    # area must never go negative even if width/height are (defensively) negative
    assert Rect(0, 0, -10, 5).area == 0
    assert Rect(0, 0, 10, -5).area == 0


@pytest.mark.parametrize(
    "rect,expected",
    [
        (Rect(0, 0, 4, 4), Point(2, 2)),  # even size
        (Rect(0, 0, 5, 5), Point(2, 2)),  # odd size (integer division truncates)
        (Rect(10, 10, 5, 5), Point(12, 12)),  # odd size, offset origin
        (Rect(-10, -10, 4, 4), Point(-8, -8)),  # negative origin, even size
    ],
)
def test_rect_center_odd_and_even_sizes(rect, expected):
    assert rect.center == expected


# -- Rect.contains ------------------------------------------------------


def test_rect_contains_inside_point():
    r = Rect(0, 0, 100, 100)
    assert r.contains(Point(50, 50)) is True


def test_rect_contains_top_left_corner_is_inclusive():
    r = Rect(10, 10, 100, 100)
    assert r.contains(Point(10, 10)) is True


def test_rect_contains_bottom_right_edge_is_exclusive():
    r = Rect(0, 0, 100, 100)
    assert r.contains(Point(100, 50)) is False  # x == right
    assert r.contains(Point(50, 100)) is False  # y == bottom
    assert r.contains(Point(99, 99)) is True


def test_rect_contains_point_outside():
    r = Rect(0, 0, 100, 100)
    assert r.contains(Point(-1, 50)) is False
    assert r.contains(Point(50, -1)) is False
    assert r.contains(Point(200, 200)) is False


# -- Rect.intersects ------------------------------------------------------


def test_rect_intersects_overlapping():
    a = Rect(0, 0, 100, 100)
    b = Rect(50, 50, 100, 100)
    assert a.intersects(b) is True
    assert b.intersects(a) is True


def test_rect_intersects_one_contains_other():
    a = Rect(0, 0, 100, 100)
    b = Rect(25, 25, 10, 10)
    assert a.intersects(b) is True
    assert b.intersects(a) is True


def test_rect_intersects_touching_edges_is_false():
    # Sharing only an edge (zero-width overlap) does not count as intersecting.
    a = Rect(0, 0, 100, 100)
    b = Rect(100, 0, 50, 50)  # starts exactly where a ends
    assert a.intersects(b) is False
    assert b.intersects(a) is False


def test_rect_intersects_disjoint_is_false():
    a = Rect(0, 0, 10, 10)
    b = Rect(100, 100, 10, 10)
    assert a.intersects(b) is False
    assert b.intersects(a) is False


# -- Rect.intersection ------------------------------------------------------


def test_rect_intersection_overlapping_returns_common_area():
    a = Rect(0, 0, 100, 100)
    b = Rect(50, 50, 100, 100)
    result = a.intersection(b)
    assert result == Rect(50, 50, 50, 50)
    assert result.area == 2500


def test_rect_intersection_non_overlapping_has_zero_area():
    a = Rect(0, 0, 10, 10)
    b = Rect(100, 100, 10, 10)
    result = a.intersection(b)
    assert result.area == 0


def test_rect_intersection_touching_edges_has_zero_area():
    a = Rect(0, 0, 100, 100)
    b = Rect(100, 0, 50, 50)
    result = a.intersection(b)
    assert result.area == 0


def test_rect_intersection_is_symmetric_in_area():
    a = Rect(10, 10, 40, 40)
    b = Rect(30, 5, 40, 40)
    assert a.intersection(b).area == b.intersection(a).area


# -- Rect.clamp ------------------------------------------------------


def test_rect_clamp_point_already_inside():
    r = Rect(0, 0, 100, 100)
    assert r.clamp(Point(50, 50)) == Point(50, 50)


def test_rect_clamp_point_outside_each_direction():
    r = Rect(10, 10, 100, 100)  # spans x:[10,110), y:[10,110)
    assert r.clamp(Point(-5, 50)) == Point(10, 50)  # left of rect
    assert r.clamp(Point(500, 50)) == Point(109, 50)  # right of rect -> right - 1
    assert r.clamp(Point(50, -5)) == Point(50, 10)  # above rect
    assert r.clamp(Point(50, 500)) == Point(50, 109)  # below rect -> bottom - 1


def test_rect_clamp_corner_case():
    r = Rect(0, 0, 10, 10)
    assert r.clamp(Point(1000, 1000)) == Point(9, 9)


# -- Rect.scaled ------------------------------------------------------


def test_rect_scaled_up():
    r = Rect(10, 20, 30, 40)
    scaled = r.scaled(2.0)
    assert scaled == Rect(20, 40, 60, 80)


def test_rect_scaled_down_rounds():
    r = Rect(1, 1, 3, 3)
    scaled = r.scaled(0.5)
    # round(0.5) == 0, round(1.5) == 2 (banker's rounding) — assert against Python's
    # actual round() semantics rather than hardcoding, so this stays correct.
    assert scaled == Rect(round(0.5), round(0.5), round(1.5), round(1.5))


def test_rect_scaled_identity():
    r = Rect(5, 6, 7, 8)
    assert r.scaled(1.0) == r


# -- Rect.to_dict / from_dict ------------------------------------------------------


def test_rect_to_dict():
    r = Rect(1, 2, 3, 4)
    assert r.to_dict() == {"x": 1, "y": 2, "width": 3, "height": 4}


def test_rect_from_dict_roundtrip():
    original = Rect(7, 8, 9, 10)
    restored = Rect.from_dict(original.to_dict())
    assert restored == original


def test_rect_from_dict_coerces_numeric_strings_and_floats():
    data = {"x": "1", "y": 2.9, "width": "30", "height": 40}
    restored = Rect.from_dict(data)
    assert restored == Rect(1, 2, 30, 40)


def test_rect_as_tuple():
    assert Rect(1, 2, 3, 4).as_tuple() == (1, 2, 3, 4)


def test_rect_is_frozen():
    r = Rect(0, 0, 1, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.width = 100  # type: ignore[misc]


# -- Monitor ------------------------------------------------------


def test_monitor_to_dict_merges_bounds():
    m = Monitor(name="eDP-1", bounds=Rect(0, 0, 1920, 1080), primary=True)
    assert m.to_dict() == {
        "name": "eDP-1",
        "primary": True,
        "x": 0,
        "y": 0,
        "width": 1920,
        "height": 1080,
    }


def test_monitor_defaults_not_primary():
    m = Monitor(name="HDMI-1", bounds=Rect(1920, 0, 1920, 1080))
    assert m.primary is False


def test_monitor_is_frozen():
    m = Monitor(name="eDP-1", bounds=Rect(0, 0, 1, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.name = "other"  # type: ignore[misc]
