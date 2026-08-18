"""Tests for lai.osl.screen.

Pure tests build Screenshot instances directly with a tiny in-memory PNG (no
display required). The @pytest.mark.x11 tests exercise the real mss/X11 backed
ScreenCapture and are read-only: they never move, close or resize anything.
"""

from __future__ import annotations

import base64
import io
import time

import pytest
from PIL import Image

from lai.errors import DisplayError
from lai.osl.geometry import Point, Rect
from lai.osl.screen import MIN_MAX_EDGE, ScreenCapture, Screenshot, annotate


def _make_png(width: int = 10, height: int = 8, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# -- pure: Screenshot coordinate mapping ------------------------------------


def test_to_screen_maps_image_point_back_to_absolute_coords():
    shot = Screenshot(
        png=_make_png(),
        region=Rect(100, 200, 800, 600),
        size=(400, 300),
        scale=0.5,
        captured_at=time.time(),
    )
    # image-space (50, 40) at scale 0.5 -> real-space (100, 80) offset by region
    result = shot.to_screen(Point(50, 40))
    assert result == Point(100 + 100, 200 + 80)


def test_to_image_maps_screen_point_into_image_space():
    shot = Screenshot(
        png=_make_png(),
        region=Rect(100, 200, 800, 600),
        size=(400, 300),
        scale=0.5,
        captured_at=time.time(),
    )
    result = shot.to_image(Point(300, 280))
    assert result == Point(round((300 - 100) * 0.5), round((280 - 200) * 0.5))


def test_to_screen_to_image_roundtrip_at_scale_other_than_one():
    shot = Screenshot(
        png=_make_png(),
        region=Rect(50, 60, 1000, 800),
        size=(250, 200),
        scale=0.25,
        captured_at=time.time(),
    )
    original = Point(80, 120)  # chosen to divide evenly at scale 0.25
    screen_point = shot.to_screen(original)
    back = shot.to_image(screen_point)
    assert back == original


def test_to_screen_identity_at_scale_one():
    shot = Screenshot(
        png=_make_png(),
        region=Rect(0, 0, 1920, 1080),
        size=(1920, 1080),
        scale=1.0,
        captured_at=time.time(),
    )
    assert shot.to_screen(Point(42, 84)) == Point(42, 84)
    assert shot.to_image(Point(42, 84)) == Point(42, 84)


def test_base64_property_matches_manual_encoding():
    png = _make_png()
    shot = Screenshot(png=png, region=Rect(0, 0, 10, 8), size=(10, 8), scale=1.0, captured_at=time.time())
    assert shot.base64 == base64.b64encode(png).decode("ascii")


def test_to_dict_shape():
    png = _make_png()
    shot = Screenshot(
        png=png,
        region=Rect(1, 2, 300, 200),
        size=(150, 100),
        scale=0.5,
        captured_at=1234.5,
    )
    data = shot.to_dict()
    assert data == {
        "region": {"x": 1, "y": 2, "width": 300, "height": 200},
        "image_size": {"width": 150, "height": 100},
        "scale": 0.5,
        "bytes": len(png),
    }


def test_to_dict_scale_is_rounded_to_4_decimals():
    shot = Screenshot(
        png=_make_png(), region=Rect(0, 0, 3, 3), size=(1, 1), scale=1 / 3, captured_at=0.0
    )
    assert shot.to_dict()["scale"] == round(1 / 3, 4)


# -- x11: real ScreenCapture -------------------------------------------------


@pytest.fixture
def screen_capture():
    sc = ScreenCapture()
    try:
        yield sc
    finally:
        sc.close()


@pytest.mark.x11
def test_monitors_non_empty(screen_capture):
    monitors = screen_capture.monitors()
    assert len(monitors) > 0
    for monitor in monitors:
        assert monitor.bounds.width > 0
        assert monitor.bounds.height > 0
        assert isinstance(monitor.name, str) and monitor.name


@pytest.mark.x11
def test_virtual_bounds_is_sane(screen_capture):
    bounds = screen_capture.virtual_bounds()
    assert bounds.width > 0
    assert bounds.height > 0


@pytest.mark.x11
def test_grab_honours_max_edge(screen_capture):
    # Use a limit above MIN_MAX_EDGE so it isn't clamped up by the floor.
    limit = 500
    assert limit > MIN_MAX_EDGE
    shot = screen_capture.grab(max_edge=limit)
    assert max(shot.size) <= limit


@pytest.mark.x11
def test_grab_explicit_region_returns_that_region(screen_capture):
    region = Rect(0, 0, 300, 200)
    shot = screen_capture.grab(region)
    assert shot.region == region


@pytest.mark.x11
def test_grab_invalid_region_raises_display_error(screen_capture):
    with pytest.raises(DisplayError):
        screen_capture.grab(Rect(0, 0, 0, 100))
    with pytest.raises(DisplayError):
        screen_capture.grab(Rect(0, 0, 100, -5))


@pytest.mark.x11
def test_annotate_returns_same_size_screenshot_with_different_bytes(screen_capture):
    shot = screen_capture.grab(max_edge=400)
    boxes = [(Rect(shot.region.x + 5, shot.region.y + 5, 40, 40), "1")]
    annotated = annotate(shot, boxes)
    assert annotated.size == shot.size
    assert annotated.png != shot.png
    assert annotated.region == shot.region
    assert annotated.scale == shot.scale
