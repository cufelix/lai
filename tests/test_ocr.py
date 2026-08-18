"""Tests for lai.osl.ocr and the ocr_* tools in lai.tools.perception.

Coordinate mapping and ranking are pure arithmetic and are tested directly,
with a hand-built Screenshot — no tesseract binary involved. The
``available``/``read`` failure path is exercised by forcing
``OCREngine._tesseract_path`` to ``None``, which is exactly what happens on a
machine without tesseract installed (this repo's dev environment included).
The @pytest.mark.x11 tests are skipped outright when the ``tesseract``
binary is missing, since they need a real recognition pass.
"""

from __future__ import annotations

import io
import shutil

import pytest

from lai.config import Config
from lai.errors import BackendUnavailable
from lai.osl.geometry import Rect
from lai.osl.ocr import (
    DEFAULT_MIN_CONFIDENCE,
    OCREngine,
    OCRWord,
    _map_word_bounds,
    _preprocess,
    _rank_words,
)
from lai.osl.screen import Screenshot
from lai.tools.base import ToolContext, ToolRegistry
from lai.tools.perception import register as register_perception

HAS_TESSERACT = shutil.which("tesseract") is not None


def _shot(*, region=Rect(1000, 500, 200, 60), scale=1.0, size=(200, 60)) -> Screenshot:
    from PIL import Image

    image = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Screenshot(png=buf.getvalue(), region=region, size=size, scale=scale, captured_at=0.0)


def context(**kwargs) -> ToolContext:
    kwargs.setdefault("config", Config())
    kwargs.setdefault("extra", {})
    return ToolContext(**kwargs)


# -- _map_word_bounds: pure coordinate mapping ------------------------------------------------------


def test_map_word_bounds_no_scaling_offsets_by_region_origin():
    shot = _shot(region=Rect(1000, 500, 200, 60), scale=1.0)
    rect = _map_word_bounds(shot, x=10, y=5, w=40, h=20, extra_scale=1.0)
    assert rect == Rect(1010, 505, 40, 20)


def test_map_word_bounds_undoes_screenshot_downscale():
    # Screenshot.scale = 0.5 means the delivered image is half the size of
    # the real screen region, so a box measured on the image must be doubled
    # to land back on real screen pixels.
    shot = _shot(region=Rect(0, 0, 400, 400), scale=0.5, size=(200, 200))
    rect = _map_word_bounds(shot, x=10, y=10, w=20, h=10, extra_scale=1.0)
    assert rect == Rect(20, 20, 40, 20)


def test_map_word_bounds_undoes_our_own_upscale():
    # We upscaled the image 2x ourselves (extra_scale) before handing it to
    # tesseract; that must be undone before the Screenshot's own scale is
    # applied, not conflated with it.
    shot = _shot(region=Rect(0, 0, 100, 100), scale=1.0, size=(100, 100))
    rect = _map_word_bounds(shot, x=20, y=20, w=40, h=20, extra_scale=2.0)
    assert rect == Rect(10, 10, 20, 10)


def test_map_word_bounds_combines_region_scale_and_upscale():
    shot = _shot(region=Rect(500, 200, 100, 100), scale=0.5, size=(50, 50))
    # image pixel (10,10) with a 2x upscale -> processed-image (10,10) undoes
    # to image (5,5) -> undoes Screenshot scale (0.5) -> real (10,10) ->
    # offset by region origin -> (510, 210).
    rect = _map_word_bounds(shot, x=10, y=10, w=10, h=10, extra_scale=2.0)
    assert rect.x == 510 and rect.y == 210
    assert rect.width == 10 and rect.height == 10


def test_map_word_bounds_minimum_size_is_one_pixel():
    shot = _shot(region=Rect(0, 0, 10, 10), scale=1.0, size=(10, 10))
    rect = _map_word_bounds(shot, x=0, y=0, w=0, h=0, extra_scale=1.0)
    assert rect.width >= 1 and rect.height >= 1


# -- _preprocess: greyscale + conditional upscale ------------------------------------------------------


def test_preprocess_upscales_small_images_towards_the_target():
    from PIL import Image

    small = Image.new("RGB", (100, 30), "white")
    processed, factor = _preprocess(small)
    assert factor > 1.0
    assert processed.mode == "L"
    assert max(processed.size) > max(small.size)


def test_preprocess_leaves_large_images_alone():
    from PIL import Image

    large = Image.new("RGB", (1200, 800), "white")
    processed, factor = _preprocess(large)
    assert factor == 1.0
    assert processed.size == large.size


def test_preprocess_upscale_is_capped():
    from PIL import Image

    from lai.osl.ocr import MAX_UPSCALE

    tiny = Image.new("RGB", (10, 10), "white")
    _processed, factor = _preprocess(tiny)
    assert factor <= MAX_UPSCALE


# -- _rank_words: find_text ranking ------------------------------------------------------


def _word(text: str, confidence: float = 90.0) -> OCRWord:
    return OCRWord(text=text, bounds=Rect(0, 0, 10, 10), confidence=confidence)


def test_rank_words_exact_match_beats_prefix_and_substring():
    words = [_word("Settings"), _word("Set"), _word("Reset")]
    ranked = _rank_words(words, "Set")
    assert [w.text for w in ranked] == ["Set", "Settings", "Reset"]


def test_rank_words_is_case_insensitive():
    words = [_word("SAVE")]
    assert _rank_words(words, "save") == words


def test_rank_words_ties_broken_by_confidence():
    words = [_word("Cancel", confidence=40.0), _word("Cancel", confidence=95.0)]
    ranked = _rank_words(words, "cancel")
    assert ranked[0].confidence == 95.0


def test_rank_words_excludes_non_matches():
    words = [_word("Save"), _word("Load")]
    assert [w.text for w in _rank_words(words, "sav")] == ["Save"]


def test_rank_words_empty_query_returns_nothing():
    assert _rank_words([_word("Save")], "") == []


# -- OCREngine.available / missing-binary failure path ------------------------------------------------------


def test_available_is_false_without_the_tesseract_binary(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    engine = OCREngine()
    assert engine.available is False


def test_read_raises_backend_unavailable_with_apt_install_hint(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    engine = OCREngine()
    with pytest.raises(BackendUnavailable, match="tesseract"):
        engine.read(Rect(0, 0, 10, 10))


def test_read_error_detail_names_the_apt_package(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    engine = OCREngine()
    try:
        engine.read(Rect(0, 0, 10, 10))
        pytest.fail("expected BackendUnavailable")
    except BackendUnavailable as exc:
        assert "apt install tesseract-ocr" in (exc.detail or "")


def test_languages_raises_when_tesseract_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    engine = OCREngine()
    with pytest.raises(BackendUnavailable):
        engine.languages()


# -- OCREngine.read: confidence filter + empty-word filter (faked backend) ------------------------------------------------------


class _FakeOutput:
    DICT = "DICT"


class _FakePytesseract:
    """Stands in for the pytesseract module, no tesseract binary involved."""

    Output = _FakeOutput

    def __init__(self, data: dict):
        self._data = data

    def image_to_data(self, _image, output_type, timeout):
        assert output_type is self.Output.DICT
        return self._data


def _engine_with_fake_backend(monkeypatch, data: dict) -> OCREngine:
    engine = OCREngine()
    monkeypatch.setattr(engine, "_tesseract_path", "/usr/bin/tesseract")
    monkeypatch.setattr(engine, "_backend", lambda: _FakePytesseract(data))
    monkeypatch.setattr("lai.osl.ocr._preprocess", lambda image: (image, 1.0))
    return engine


def test_read_filters_empty_whitespace_and_low_confidence_words(monkeypatch):
    data = {
        "text": ["", "Hello", "  ", "world", "faint"],
        "conf": ["-1", "95", "-1", "55", "10"],
        "left": [0, 10, 0, 60, 120],
        "top": [0, 5, 0, 5, 5],
        "width": [0, 40, 0, 40, 40],
        "height": [0, 20, 0, 20, 20],
    }
    engine = _engine_with_fake_backend(monkeypatch, data)
    result = engine.read(_shot(region=Rect(1000, 500, 200, 60)))

    assert [w.text for w in result.words] == ["Hello", "world"]
    assert result.text == "Hello world"
    # Confidence below DEFAULT_MIN_CONFIDENCE (40) was dropped.
    assert all(w.confidence >= DEFAULT_MIN_CONFIDENCE for w in result.words)
    # And the surviving words carry absolute screen coordinates.
    assert result.words[0].bounds == Rect(1010, 505, 40, 20)


def test_read_honours_an_explicit_min_confidence_override(monkeypatch):
    data = {
        "text": ["ok"],
        "conf": ["50"],
        "left": [0],
        "top": [0],
        "width": [10],
        "height": [10],
    }
    engine = _engine_with_fake_backend(monkeypatch, data)
    result = engine.read(_shot(), min_confidence=60)
    assert result.words == ()


def test_find_text_uses_read_and_ranks(monkeypatch):
    data = {
        "text": ["Save", "Cancel"],
        "conf": ["90", "90"],
        "left": [0, 50],
        "top": [0, 0],
        "width": [20, 30],
        "height": [10, 10],
    }
    engine = _engine_with_fake_backend(monkeypatch, data)
    matches = engine.find_text("save", _shot())
    assert [m.text for m in matches] == ["Save"]


# -- ocr_read / ocr_find tools: failure path when tesseract is missing ------------------------------------------------------


class _FakeScreen:
    def virtual_bounds(self) -> Rect:
        return Rect(0, 0, 1920, 1080)


class _FakeWindows:
    def active_window(self):
        return None


class _FakeDesktop:
    def __init__(self) -> None:
        self.screen = _FakeScreen()
        self.windows = _FakeWindows()


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_perception(reg)
    return reg


def test_ocr_read_tool_fails_clearly_without_tesseract(monkeypatch, registry):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    ctx = context(desktop=_FakeDesktop())
    result = registry.call("ocr_read", {"scope": "desktop"}, ctx)
    assert result.ok is False
    assert "tesseract" in result.content.lower()


def test_ocr_find_tool_fails_clearly_without_tesseract(monkeypatch, registry):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    ctx = context(desktop=_FakeDesktop())
    result = registry.call("ocr_find", {"query": "Save", "scope": "desktop"}, ctx)
    assert result.ok is False
    assert "tesseract" in result.content.lower()


def test_ocr_read_tool_rejects_a_malformed_region(registry):
    ctx = context(desktop=_FakeDesktop())
    result = registry.call("ocr_read", {"scope": "region", "region": {"x": 1}}, ctx)
    assert result.ok is False
    assert "region" in result.content


def test_ocr_engines_are_isolated_per_context(registry):
    """ctx.extra must not leak an engine between two independent contexts."""
    ctx1 = context(desktop=_FakeDesktop())
    ctx2 = context(desktop=_FakeDesktop())
    registry.call("ocr_read", {"scope": "desktop"}, ctx1)
    assert "ocr_engine" not in ctx2.extra


# -- x11: real recognition (skipped without a tesseract binary) ------------------------------------------------------


@pytest.mark.x11
@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract binary is not installed")
def test_real_ocr_reads_something_from_the_live_screen():
    engine = OCREngine()
    result = engine.read(Rect(0, 0, 1920, 1080))
    assert isinstance(result.text, str)
    assert result.duration >= 0
