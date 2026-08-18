"""OCR: reading text from pixels rather than the accessibility tree.

Why this module exists: plenty of real apps expose no accessibility tree at
all — Chromium/Electron apps not started with ``--force-renderer-
accessibility``, games, and anything rendering inside a VM window. For those,
the only way the agent can "read" the screen is to look at the pixels. This
gives it that fallback, but tags every recognised word with its *absolute
screen* coordinates (not just its text), so a word OCR reads can also be
clicked — the same contract :mod:`lai.tools.computer` already uses for
pointer coordinates.

Non-obvious design decision: word boxes are always mapped back through
``Screenshot.region``/``scale`` (and, when we did our own upscaling for
legibility, through that too) before being returned. A :class:`Screenshot`
may be downscaled (``screen.py`` caps images at ~1400px on the long edge to
keep vision-token cost down) or cropped to a region; if we handed back raw
pixel coordinates from the image tesseract actually saw, they would be
meaningless anywhere except that specific, possibly-shrunk image.
"""

from __future__ import annotations

import io
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass

from ..errors import BackendUnavailable
from .geometry import Point, Rect
from .screen import ScreenCapture, Screenshot

DEFAULT_MIN_CONFIDENCE = 40.0
DEFAULT_TIMEOUT = 10.0
DEFAULT_LIMIT = 20

# screen.py's default cap (~1400px) is tuned for vision-model token cost, not
# OCR accuracy — small text needs close to full resolution to be legible to
# tesseract at all, so we grab far less aggressively when OCR is the point.
OCR_CAPTURE_MAX_EDGE = 6000

# Tesseract's accuracy falls off a cliff once glyphs get much below ~30px
# tall. We can't know the actual glyph height up front, but a decent proxy
# is "how big is the captured region overall": normal UI text (12-14px
# fonts) comfortably clears the 30px line-height threshold once the image's
# longest edge reaches a few hundred pixels. Upscaling small crops (a single
# button, a small dialog) towards this target measurably improves accuracy.
SMALL_EDGE_TARGET = 900
MAX_UPSCALE = 4.0


@dataclass(frozen=True, slots=True)
class OCRWord:
    """One recognised word, with its box in absolute screen coordinates."""

    text: str
    bounds: Rect
    confidence: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bounds": self.bounds.to_dict(),
            "confidence": round(self.confidence, 1),
        }


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    words: tuple[OCRWord, ...]
    region: Rect
    duration: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "region": self.region.to_dict(),
            "word_count": len(self.words),
            "words": [w.to_dict() for w in self.words],
            "duration": round(self.duration, 3),
        }


class OCREngine:
    """pytesseract-backed OCR that reports word boxes in screen space."""

    def __init__(
        self,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        capture: ScreenCapture | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self._capture = capture
        self._tesseract_path = shutil.which("tesseract")
        self._pytesseract = None

    @property
    def available(self) -> bool:
        return self._tesseract_path is not None

    def _require(self) -> None:
        if not self.available:
            raise BackendUnavailable(
                "tesseract is not installed",
                detail="install it with: sudo apt install tesseract-ocr",
            )

    def _backend(self):
        if self._pytesseract is None:
            try:
                import pytesseract  # noqa: PLC0415 - optional heavy import
            except ImportError as exc:  # pragma: no cover - env dependent
                raise BackendUnavailable(
                    "pytesseract is not installed", detail="install it with: pip install pytesseract"
                ) from exc
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_path
            self._pytesseract = pytesseract
        return self._pytesseract

    def _screen(self) -> ScreenCapture:
        if self._capture is None:
            self._capture = ScreenCapture()
        return self._capture

    def languages(self) -> list[str]:
        self._require()
        pytesseract = self._backend()
        try:
            return list(pytesseract.get_languages(config=""))
        except Exception as exc:  # pragma: no cover - depends on install
            raise BackendUnavailable("failed to query tesseract languages", detail=str(exc)) from exc

    def read(
        self,
        source: Screenshot | Rect,
        *,
        min_confidence: float | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> OCRResult:
        """Recognise text in ``source`` (a fresh :class:`Screenshot` or a screen
        :class:`Rect` to capture), returning words with screen-space boxes."""
        self._require()
        pytesseract = self._backend()
        started = time.monotonic()

        if isinstance(source, Screenshot):
            shot = source
        else:
            shot = self._screen().grab(source, max_edge=OCR_CAPTURE_MAX_EDGE)

        from PIL import Image  # noqa: PLC0415 - optional heavy import

        image = Image.open(io.BytesIO(shot.png))
        processed, extra_scale = _preprocess(image)

        threshold = self.min_confidence if min_confidence is None else min_confidence
        try:
            data = pytesseract.image_to_data(
                processed, output_type=pytesseract.Output.DICT, timeout=timeout
            )
        except RuntimeError as exc:
            # pytesseract raises RuntimeError (or its RunTimeoutError
            # subclass) when the subprocess is killed for exceeding
            # ``timeout`` — treat it the same as any other backend failure.
            raise BackendUnavailable(f"OCR timed out after {timeout}s", detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - depends on install
            raise BackendUnavailable("tesseract failed to run", detail=str(exc)) from exc

        words: list[OCRWord] = []
        texts: list[str] = []
        count = len(data.get("text", []))
        for i in range(count):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][i])
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < threshold:
                continue
            bounds = _map_word_bounds(
                shot, data["left"][i], data["top"][i], data["width"][i], data["height"][i], extra_scale
            )
            words.append(OCRWord(text=text, bounds=bounds, confidence=confidence))
            texts.append(text)

        return OCRResult(
            text=" ".join(texts),
            words=tuple(words),
            region=shot.region,
            duration=time.monotonic() - started,
        )

    def find_text(
        self,
        query: str,
        source: Screenshot | Rect,
        *,
        min_confidence: float | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[OCRWord]:
        """Case-insensitive substring search over OCR'd words, best match first."""
        result = self.read(source, min_confidence=min_confidence)
        return _rank_words(result.words, query)[:limit]


def _preprocess(image) -> tuple:
    """Greyscale (tesseract ignores colour anyway) plus a conditional upscale.

    Returns ``(image, extra_scale)`` where ``extra_scale`` is how much larger
    the returned image is than ``image`` on entry — callers need it to map
    word boxes back to the original pixel space.
    """
    from PIL import Image  # noqa: PLC0415

    grey = image.convert("L")
    longest = max(grey.size)
    if longest >= SMALL_EDGE_TARGET:
        return grey, 1.0
    factor = min(MAX_UPSCALE, SMALL_EDGE_TARGET / max(1, longest))
    resized = grey.resize(
        (max(1, round(grey.width * factor)), max(1, round(grey.height * factor))), Image.LANCZOS
    )
    return resized, factor


def _map_word_bounds(shot: Screenshot, x: int, y: int, w: int, h: int, extra_scale: float) -> Rect:
    """Map a tesseract word box (processed-image pixels) to absolute screen space.

    Pure arithmetic, deliberately factored out of :meth:`OCREngine.read` so it
    can be unit tested without a tesseract binary: undo our own upscale
    (``extra_scale``), then undo the Screenshot's capture scale/region via
    ``Screenshot.to_screen``.
    """
    image_x, image_y = x / extra_scale, y / extra_scale
    image_w, image_h = w / extra_scale, h / extra_scale
    top_left = shot.to_screen(Point(round(image_x), round(image_y)))
    width = max(1, round(image_w / shot.scale))
    height = max(1, round(image_h / shot.scale))
    return Rect(top_left.x, top_left.y, width, height)


def _rank_words(words: Iterable[OCRWord], query: str) -> list[OCRWord]:
    """Case-insensitive substring ranking: exact > prefix > contains, ties
    broken by OCR confidence."""
    needle = query.strip().lower()
    if not needle:
        return []
    scored: list[tuple[int, float, OCRWord]] = []
    for word in words:
        hay = word.text.lower()
        if needle == hay:
            score = 3
        elif hay.startswith(needle):
            score = 2
        elif needle in hay:
            score = 1
        else:
            continue
        scored.append((score, word.confidence, word))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [word for _, _, word in scored]
