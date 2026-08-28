"""Screen capture: the agent's eyes.

Captures the virtual screen, a single monitor or an arbitrary rectangle, then
downscales to a model-friendly size and reports the ``scale`` so callers can map
model-space coordinates back to real screen coordinates.
"""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Iterable
from dataclasses import dataclass

from ..errors import BackendUnavailable, DisplayError
from .geometry import Monitor, Point, Rect

# Claude's vision sweet spot: keep the long edge at or below this so a full-HD
# desktop costs ~1.1k tokens instead of ~2.5k, without losing UI legibility.
DEFAULT_MAX_EDGE = 1400
MIN_MAX_EDGE = 320


@dataclass(frozen=True, slots=True)
class Screenshot:
    """A captured image plus everything needed to reason about coordinates."""

    png: bytes
    region: Rect
    """Region of the real screen that was captured (absolute coords)."""
    size: tuple[int, int]
    """Pixel size of the delivered image (after downscaling)."""
    scale: float
    """delivered_pixels / real_pixels. Multiply real coords by this to get image coords."""
    captured_at: float

    @property
    def base64(self) -> str:
        return base64.b64encode(self.png).decode("ascii")

    def to_screen(self, point: Point) -> Point:
        """Map a point in image space back to absolute screen coordinates."""
        return Point(
            self.region.x + round(point.x / self.scale),
            self.region.y + round(point.y / self.scale),
        )

    def to_image(self, point: Point) -> Point:
        """Map an absolute screen point into image space."""
        return Point(
            round((point.x - self.region.x) * self.scale),
            round((point.y - self.region.y) * self.scale),
        )

    def to_dict(self) -> dict:
        return {
            "region": self.region.to_dict(),
            "image_size": {"width": self.size[0], "height": self.size[1]},
            "scale": round(self.scale, 4),
            "bytes": len(self.png),
        }


def _no_display_message() -> str:
    """Why a capture failed, in terms somebody can act on."""
    import os  # noqa: PLC0415

    display = os.environ.get("DISPLAY")
    if not display:
        return (
            "cannot capture the screen: DISPLAY is not set, so there is no session to "
            "capture. This usually means LAI was started by something that strips the "
            "environment — a service, cron, or an MCP client."
        )
    return (
        f"cannot capture the screen on DISPLAY={display!r} — the X server is not "
        f"answering. If this was the agent's own screen, it has been shut down."
    )


class ScreenCapture:
    """mss-backed capture with lazy, per-thread connections."""

    def __init__(self, max_edge: int = DEFAULT_MAX_EDGE) -> None:
        self.max_edge = max(MIN_MAX_EDGE, int(max_edge))
        self._mss = None

    def _backend(self):
        if self._mss is None:
            try:
                import mss  # noqa: PLC0415 - optional heavy import
            except ImportError as exc:  # pragma: no cover - env dependent
                raise BackendUnavailable("mss is not installed", detail=str(exc)) from exc
            try:
                self._mss = mss.mss()
            except Exception as exc:  # pragma: no cover - env dependent
                raise DisplayError("cannot open display for capture", detail=str(exc)) from exc
        return self._mss

    def close(self) -> None:
        if self._mss is not None:
            try:
                self._mss.close()
            finally:
                self._mss = None

    def monitors(self) -> list[Monitor]:
        raw = self._backend().monitors
        out: list[Monitor] = []
        for index, mon in enumerate(raw[1:], start=1):
            out.append(
                Monitor(
                    name=str(mon.get("name") or mon.get("output") or f"monitor-{index}"),
                    bounds=Rect(mon["left"], mon["top"], mon["width"], mon["height"]),
                    primary=bool(mon.get("is_primary", index == 1)),
                )
            )
        return out

    def virtual_bounds(self) -> Rect:
        """Bounding box covering every monitor."""
        raw = self._backend().monitors[0]
        return Rect(raw["left"], raw["top"], raw["width"], raw["height"])

    def primary_bounds(self) -> Rect:
        for monitor in self.monitors():
            if monitor.primary:
                return monitor.bounds
        return self.virtual_bounds()

    def grab(self, region: Rect | None = None, *, max_edge: int | None = None) -> Screenshot:
        """Capture ``region`` (default: the whole virtual screen)."""
        bounds = region or self.virtual_bounds()
        if bounds.width <= 0 or bounds.height <= 0:
            raise DisplayError(f"invalid capture region {bounds.as_tuple()}")

        backend = self._backend()
        try:
            raw = backend.grab(
                {"left": bounds.x, "top": bounds.y, "width": bounds.width, "height": bounds.height}
            )
        except AssertionError as exc:
            # mss carries `assert self._monitors is not None`, which is what
            # fires when it cannot reach a display. Passed through, the model is
            # handed "raised AssertionError:" — an empty string it can do
            # nothing with, when the actual problem has a name and a fix.
            raise DisplayError(_no_display_message()) from exc
        from PIL import Image  # noqa: PLC0415 - optional heavy import

        image = Image.frombytes("RGB", raw.size, raw.rgb)
        limit = max(MIN_MAX_EDGE, int(max_edge or self.max_edge))
        scale = 1.0
        longest = max(image.size)
        if longest > limit:
            scale = limit / longest
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return Screenshot(
            png=buffer.getvalue(),
            region=bounds,
            size=image.size,
            scale=scale,
            captured_at=time.time(),
        )

    def grab_monitor(self, index: int = 0, **kwargs) -> Screenshot:
        mons = self.monitors()
        if not mons:
            raise DisplayError("no monitors detected")
        if not 0 <= index < len(mons):
            raise DisplayError(f"monitor {index} out of range (have {len(mons)})")
        return self.grab(mons[index].bounds, **kwargs)

    def monitor_for(self, point: Point) -> Monitor | None:
        for monitor in self.monitors():
            if monitor.bounds.contains(point):
                return monitor
        return None


def annotate(
    shot: Screenshot,
    boxes: Iterable[tuple[Rect, str]],
    *,
    color: str = "#ff2d55",
    outline_width: int = 2,
) -> Screenshot:
    """Draw labelled boxes (given in *screen* coords) onto a screenshot.

    Used to render numbered element refs so the model can say "click ref 12"
    instead of guessing pixels.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    image = Image.open(io.BytesIO(shot.png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for rect, label in boxes:
        local = Rect(
            round((rect.x - shot.region.x) * shot.scale),
            round((rect.y - shot.region.y) * shot.scale),
            max(1, round(rect.width * shot.scale)),
            max(1, round(rect.height * shot.scale)),
        )
        draw.rectangle(
            [local.x, local.y, local.right, local.bottom], outline=color, width=outline_width
        )
        if label:
            text_y = max(0, local.y - 12)
            draw.rectangle([local.x, text_y, local.x + 8 * len(label) + 4, text_y + 12], fill=color)
            draw.text((local.x + 2, text_y), label, fill="#ffffff")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return Screenshot(
        png=buffer.getvalue(),
        region=shot.region,
        size=image.size,
        scale=shot.scale,
        captured_at=shot.captured_at,
    )
