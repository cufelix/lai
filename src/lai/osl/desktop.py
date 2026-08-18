"""The Desktop facade — one object the agent's tools drive.

Fuses the four perception sources (pixels, a11y tree, window tree, process
tree) and the two actuation paths (semantic AT-SPI actions, raw input events)
into the small set of operations an agent actually needs.

The important policy lives here: **prefer semantic actions, fall back to pixels.**
Clicking a button by name is deterministic; clicking coordinates is a guess.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..errors import ElementNotFound, LaiError, WindowNotFound
from .a11y import A11yTree, Element, Snapshot
from .apps import AppEntry, AppLauncher, LaunchResult
from .clipboard import Clipboard
from .geometry import Monitor, Point
from .inputs import InputController, InputResult
from .screen import ScreenCapture, Screenshot, annotate
from .session import ensure_display
from .windows import WindowInfo, WindowManager

SETTLE_POLL = 0.12
SETTLE_SAMPLE_EDGE = 320
# How long a UI snapshot stays trustworthy enough to resolve a name against.
SNAPSHOT_REUSE_SECONDS = 3.0


@dataclass(slots=True)
class Observation:
    """Everything the agent knows about the screen at one instant."""

    active_window: WindowInfo | None
    windows: list[WindowInfo] = field(default_factory=list)
    elements: Snapshot | None = None
    screenshot: Screenshot | None = None
    monitors: list[Monitor] = field(default_factory=list)
    captured_at: float = field(default_factory=time.time)

    def summary(self, *, max_windows: int = 12, max_elements: int = 60) -> str:
        """Text rendering fed to the model alongside the screenshot."""
        lines: list[str] = []
        if self.active_window:
            w = self.active_window
            lines.append(
                f"FOCUSED: {w.title!r} [{w.wm_class}] at {w.bounds.as_tuple()}"
                + (f" states={','.join(w.states)}" if w.states else "")
            )
        else:
            lines.append("FOCUSED: (none)")

        if self.windows:
            lines.append(f"\nWINDOWS ({len(self.windows)}):")
            for w in self.windows[:max_windows]:
                marker = "*" if w.active else " "
                lines.append(
                    f" {marker} {w.hex_id} {w.wm_class or '?'}: {w.title[:70]!r} {w.bounds.as_tuple()}"
                )
            if len(self.windows) > max_windows:
                lines.append(f"   ... and {len(self.windows) - max_windows} more")

        if self.elements is not None:
            count = len(self.elements)
            lines.append(f"\nUI ELEMENTS ({count} accessible):")
            lines.append(self.elements.render(limit=max_elements))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "active_window": self.active_window.to_dict() if self.active_window else None,
            "windows": [w.to_dict() for w in self.windows],
            "elements": self.elements.to_dict() if self.elements else None,
            "screenshot": self.screenshot.to_dict() if self.screenshot else None,
            "monitors": [m.to_dict() for m in self.monitors],
        }


class Desktop:
    """High-level desktop control. Every tool in :mod:`lai.tools` goes through this."""

    def __init__(
        self,
        *,
        max_edge: int = 1400,
        a11y_timeout_ms: int = 800,
        display: str | None = None,
    ) -> None:
        # Resolve the X session first: LAI is frequently launched by something
        # that strips DISPLAY (MCP clients, systemd, cron, ssh).
        self.display = ensure_display(display or "")
        self.screen = ScreenCapture(max_edge=max_edge)
        self.input = InputController()
        self.windows = WindowManager(display)
        self.a11y = A11yTree(timeout_ms=a11y_timeout_ms)
        self.apps = AppLauncher(self.windows)
        self.clipboard = Clipboard()
        self._snapshot: Snapshot | None = None

    def close(self) -> None:
        self.screen.close()
        self.windows.close()

    # -- perception ------------------------------------------------------

    def observe(
        self,
        *,
        screenshot: bool = True,
        elements: bool = True,
        scope: str = "focused",
        max_elements: int = 220,
        annotate_elements: bool = False,
    ) -> Observation:
        """Capture the current desktop state.

        ``scope="focused"`` restricts the a11y walk and the screenshot to the
        focused window — far cheaper and usually what matters. ``scope="desktop"``
        covers everything.
        """
        active = self._safe(lambda: self.windows.active_window())
        window_list = self._safe(lambda: self.windows.list_windows()) or []
        monitors = self._safe(lambda: self.screen.monitors()) or []

        snapshot: Snapshot | None = None
        if elements and self.a11y.available:
            snapshot = self._safe(
                lambda: self.a11y.snapshot(
                    pid=active.pid if (scope == "focused" and active) else None,
                    within=active.bounds if (scope == "focused" and active) else None,
                    max_elements=max_elements,
                )
            )
            if snapshot is not None:
                self._snapshot = snapshot

        shot: Screenshot | None = None
        if screenshot:
            region = None
            if scope == "focused" and active and active.bounds.area > 0:
                region = active.bounds.intersection(self.screen.virtual_bounds())
                if region.area <= 0:
                    region = None
            shot = self._safe(lambda: self.screen.grab(region))
            if shot and annotate_elements and snapshot:
                boxes = [
                    (e.bounds, str(e.ref))
                    for e in snapshot.elements
                    if e.interactive and e.bounds.area > 0
                ]
                if boxes:
                    shot = self._safe(lambda: annotate(shot, boxes)) or shot

        return Observation(
            active_window=active,
            windows=window_list,
            elements=snapshot,
            screenshot=shot,
            monitors=monitors,
        )

    def snapshot(self, **kwargs) -> Snapshot:
        """Fresh a11y snapshot (defaults to the focused window's process)."""
        if "pid" not in kwargs and "app_name" not in kwargs:
            active = self._safe(lambda: self.windows.active_window())
            if active:
                kwargs.setdefault("pid", active.pid)
                kwargs.setdefault("within", active.bounds)
        snapshot = self.a11y.snapshot(**kwargs)
        self._snapshot = snapshot
        return snapshot

    @property
    def last_snapshot(self) -> Snapshot | None:
        return self._snapshot

    def resolve(self, ref: int) -> tuple[Element, object]:
        """Look up an element ref from the most recent snapshot."""
        if self._snapshot is None:
            raise ElementNotFound("no UI snapshot taken yet; call ui_snapshot first")
        return self._snapshot.get(ref), self._snapshot.handle(ref)

    def find_element(
        self,
        query: str,
        *,
        role: str | None = None,
        fresh: bool = True,
        reuse_within: float = SNAPSHOT_REUSE_SECONDS,
    ) -> Element:
        """Locate an element by name, preferring the snapshot the model just saw.

        Always re-snapshotting looks safer but is not: it reads whatever happens
        to be focused *now*, so a focus change between the model's `ui_snapshot`
        and its `ui_click` silently searches a different application. A recent
        snapshot is both cheaper and truer to what the model was looking at.
        """
        recent = self._snapshot
        if recent is not None and (time.time() - recent.captured_at) <= reuse_within:
            try:
                return recent.find_one(query, role=role)
            except ElementNotFound:
                pass  # not in the old view — fall through and look again

        if not fresh and recent is not None:
            return recent.find_one(query, role=role)

        snapshot = self.snapshot()
        try:
            return snapshot.find_one(query, role=role)
        except ElementNotFound as exc:
            active = self._safe(lambda: self.windows.active_window())
            where = f"{active.wm_class} ({active.title!r})" if active else "the focused window"
            raise ElementNotFound(
                f"{exc.message} in {where}",
                detail=(
                    f"{len(snapshot)} accessible element(s) were visible. "
                    "Take a fresh ui_snapshot to see what is actually on screen — "
                    "the window may have changed since you last looked."
                ),
            ) from exc

    # -- actuation: semantic ---------------------------------------------

    def click_element(
        self, target: int | str, *, role: str | None = None, prefer_action: bool = True
    ) -> dict:
        """Click by ref or by name. Uses the accessible action when possible."""
        if isinstance(target, int):
            element, handle = self.resolve(target)
        else:
            element = self.find_element(target, role=role)
            handle = self._snapshot.handle(element.ref)  # type: ignore[union-attr]

        if not element.enabled:
            raise LaiError(f"element {element.name!r} ({element.role}) is disabled")

        if prefer_action and element.actions:
            try:
                action = self.a11y.do_action(handle)
                return {
                    "method": "a11y_action",
                    "action": action,
                    "element": element.to_dict(),
                }
            except Exception:
                pass  # fall through to a real pointer click

        center = element.center
        if not self.screen.virtual_bounds().contains(center):
            raise LaiError(
                f"element {element.name!r} centre {center.as_tuple()} is off-screen; "
                "scroll it into view first"
            )
        self.input.click(center)
        return {"method": "pointer", "element": element.to_dict(), "point": center.as_tuple()}

    def set_element_text(self, target: int | str, text: str, *, clear: bool = True) -> dict:
        """Type into a field, semantically where possible."""
        if isinstance(target, int):
            element, handle = self.resolve(target)
        else:
            element = self.find_element(target)
            handle = self._snapshot.handle(element.ref)  # type: ignore[union-attr]

        try:
            self.a11y.set_text(handle, text)
            return {"method": "a11y_set_text", "element": element.to_dict(), "length": len(text)}
        except Exception:
            pass

        # Fallback: focus it, select-all, type over.
        self.a11y.focus(handle)
        self.input.click(element.center)
        time.sleep(0.08)
        if clear:
            self.input.key("ctrl+a")
            self.input.key("Delete")
        self.input.type_text(text)
        return {"method": "typed", "element": element.to_dict(), "length": len(text)}

    def focus_element(self, target: int | str) -> dict:
        if isinstance(target, int):
            element, handle = self.resolve(target)
        else:
            element = self.find_element(target)
            handle = self._snapshot.handle(element.ref)  # type: ignore[union-attr]
        ok = self.a11y.focus(handle)
        if not ok:
            self.input.click(element.center)
        return {"focused": element.to_dict(), "method": "a11y" if ok else "pointer"}

    # -- actuation: raw --------------------------------------------------

    def click(self, x: int, y: int, *, button: str = "left", count: int = 1) -> InputResult:
        point = self._validate_point(Point(int(x), int(y)))
        return self.input.click(point, button=button, count=count)

    def move(self, x: int, y: int) -> InputResult:
        return self.input.move(self._validate_point(Point(int(x), int(y))))

    def drag(self, x1: int, y1: int, x2: int, y2: int, *, button: str = "left") -> InputResult:
        return self.input.drag(
            self._validate_point(Point(int(x1), int(y1))),
            self._validate_point(Point(int(x2), int(y2))),
            button=button,
        )

    def type_text(self, text: str) -> InputResult:
        return self.input.type_text(text)

    def key(self, spec: str, *, count: int = 1) -> InputResult:
        return self.input.key(spec, count=count)

    def scroll(
        self, direction: str = "down", *, amount: int = 3,
        x: int | None = None, y: int | None = None,
    ) -> InputResult:
        point = None
        if x is not None and y is not None:
            point = self._validate_point(Point(int(x), int(y)))
        return self.input.scroll(direction, amount=amount, point=point)

    def _validate_point(self, point: Point) -> Point:
        bounds = self.screen.virtual_bounds()
        if not bounds.contains(point):
            raise LaiError(
                f"point {point.as_tuple()} is outside the screen {bounds.as_tuple()}"
            )
        return point

    # -- windows & apps --------------------------------------------------

    def focus_window(self, target: int | str) -> WindowInfo:
        window = self.windows.get(target) if isinstance(target, int) else self.windows.find_one(target)
        return self.windows.focus(window.id)

    def close_window(self, target: int | str) -> dict:
        window = self.windows.get(target) if isinstance(target, int) else self.windows.find_one(target)
        self.windows.close_window(window.id)
        time.sleep(0.35)
        still_open = any(w.id == window.id for w in self.windows.list_windows(include_invisible=True))
        return {"closed": window.to_dict(), "gone": not still_open}

    def open_app(self, query: str, **kwargs) -> LaunchResult:
        return self.apps.launch(query, **kwargs)

    def list_apps(self, query: str = "", *, limit: int = 40) -> list[AppEntry]:
        return self.apps.find(query, limit=limit) if query else self.apps.apps()[:limit]

    # -- synchronisation -------------------------------------------------

    def wait_settle(self, *, timeout: float = 3.0, stable_for: float = 0.35, threshold: float = 0.004) -> dict:
        """Block until the screen stops changing.

        Cheap perceptual diff on a heavily downscaled grab — this is what stops
        the agent from acting on a half-drawn window.
        """
        try:
            from PIL import Image, ImageChops  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            time.sleep(min(timeout, 0.5))
            return {"settled": False, "reason": "pillow unavailable"}

        import io

        def sample():
            shot = self.screen.grab(max_edge=SETTLE_SAMPLE_EDGE)
            return Image.open(io.BytesIO(shot.png)).convert("L")

        deadline = time.monotonic() + timeout
        previous = sample()
        stable_since: float | None = None
        checks = 0
        while time.monotonic() < deadline:
            time.sleep(SETTLE_POLL)
            current = sample()
            checks += 1
            if current.size != previous.size:
                diff_ratio = 1.0
            else:
                diff = ImageChops.difference(current, previous)
                histogram = diff.histogram()
                changed = sum(histogram[12:])  # ignore tiny anti-aliasing noise
                total = current.width * current.height
                diff_ratio = changed / max(1, total)
            previous = current
            if diff_ratio <= threshold:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= stable_for:
                    return {"settled": True, "checks": checks, "last_diff": round(diff_ratio, 5)}
            else:
                stable_since = None
        return {"settled": False, "checks": checks, "reason": "timeout"}

    def wait_for_window(self, query: str, *, timeout: float = 20.0) -> WindowInfo:
        window = self.windows.wait_for(lambda w: w.matches(query), timeout=timeout)
        if window is None:
            raise WindowNotFound(f"no window matching {query!r} appeared within {timeout}s")
        return window

    def wait_for_element(
        self, query: str, *, role: str | None = None, timeout: float = 15.0, poll: float = 0.4
    ) -> Element:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.find_element(query, role=role)
            except Exception as exc:
                last_error = exc
                time.sleep(poll)
        raise ElementNotFound(
            f"element {query!r} did not appear within {timeout}s",
            detail=str(last_error) if last_error else None,
        )

    # -- misc ------------------------------------------------------------

    @staticmethod
    def _safe(fn):
        """Perception must never crash the loop: degrade to None."""
        try:
            return fn()
        except Exception:
            return None
