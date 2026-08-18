"""Idle detection via the X11 MIT-SCREEN-SAVER extension.

Why this matters: an autonomous agent that drives the real mouse and
keyboard must not fight the human for them. Before taking over the pointer
for a multi-step UI flow, a caller should check :meth:`IdleMonitor.user_active`
— this is X's own notion of "how long since the last real input event", the
same signal screensavers and lock screens use, so it agrees with what the
user actually experiences as "away".

MIT-SCREEN-SAVER via python-xlib rather than shelling out to ``xprintidle``
or ``xssstate``: no subprocess per poll, and the raw ``XScreenSaverQueryInfo``
idle time comes back as milliseconds straight from the X server — nothing to
parse, nothing that can hang like a wedged subprocess.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ..errors import BackendUnavailable, DisplayError
from .geometry import Point

DEFAULT_ACTIVE_THRESHOLD = 3.0
DEFAULT_WAIT_TIMEOUT = 60.0
POLL_INTERVAL = 0.25


@dataclass(frozen=True, slots=True)
class IdleState:
    idle_seconds: float
    active: bool
    threshold: float

    def to_dict(self) -> dict:
        return {
            "idle_seconds": round(self.idle_seconds, 2),
            "active": self.active,
            "threshold": self.threshold,
        }


class IdleMonitor:
    """Reads X server idle time via the MIT-SCREEN-SAVER extension."""

    def __init__(self, display_name: str | None = None) -> None:
        self._display_name = display_name or os.environ.get("DISPLAY")
        self._display = None
        self._root = None
        self._ext_ok: bool | None = None

    # -- connection --------------------------------------------------------

    def _conn(self):
        if self._display is None:
            try:
                from Xlib import display as xdisplay  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - env dependent
                raise BackendUnavailable("python-xlib is not installed", detail=str(exc)) from exc
            if not self._display_name:
                raise DisplayError("DISPLAY is not set; no X11 session to attach to")
            try:
                self._display = xdisplay.Display(self._display_name)
            except Exception as exc:  # pragma: no cover - env dependent
                raise DisplayError(
                    f"cannot connect to X display {self._display_name!r}", detail=str(exc)
                ) from exc
            self._root = self._display.screen().root
            try:
                from Xlib.ext import screensaver  # noqa: PLC0415,F401 - registers the extension
            except ImportError as exc:  # pragma: no cover - env dependent
                raise BackendUnavailable(
                    "python-xlib's screensaver extension module is missing", detail=str(exc)
                ) from exc
        return self._display

    def close(self) -> None:
        if self._display is not None:
            try:
                self._display.close()
            finally:
                self._display = None
                self._root = None
                self._ext_ok = None

    @property
    def available(self) -> bool:
        if self._ext_ok is None:
            try:
                display = self._conn()
                self._ext_ok = bool(display.has_extension("MIT-SCREEN-SAVER"))
            except Exception:
                self._ext_ok = False
        return self._ext_ok

    # -- reading -------------------------------------------------------------

    def idle_seconds(self) -> float:
        if not self.available:
            raise BackendUnavailable(
                "MIT-SCREEN-SAVER extension is not available on this X server"
            )
        try:
            info = self._root.screensaver_query_info()
        except Exception as exc:
            raise DisplayError("screensaver query failed", detail=str(exc)) from exc
        return info.idle / 1000.0

    def user_active(self, threshold: float = DEFAULT_ACTIVE_THRESHOLD) -> bool:
        return self.idle_seconds() < threshold

    def state(self, threshold: float = DEFAULT_ACTIVE_THRESHOLD) -> IdleState:
        seconds = self.idle_seconds()
        return IdleState(idle_seconds=seconds, active=seconds < threshold, threshold=threshold)

    def wait_for_idle(
        self, seconds: float, *, timeout: float = DEFAULT_WAIT_TIMEOUT, poll: float = POLL_INTERVAL
    ) -> bool:
        """Block until the user has been idle for at least ``seconds``.

        Returns ``True`` once satisfied, ``False`` on timeout — a timeout is
        an ordinary outcome (the human just didn't step away), not a failure,
        so the caller decides what to do next rather than us raising.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self.idle_seconds() >= seconds:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)

    def pointer_moved_by_user(self, since_position: Point, *, tolerance: int = 3) -> bool:
        """True if the pointer is no longer near ``since_position``.

        A crude but effective proxy for "did the human grab the mouse":
        compare against a position the caller recorded earlier (e.g. right
        before starting an automated action). Not idle-time based, so it also
        catches a user nudging the mouse without otherwise typing.
        """
        self._conn()
        try:
            pointer = self._root.query_pointer()
        except Exception as exc:
            raise DisplayError("pointer query failed", detail=str(exc)) from exc
        current = Point(pointer.root_x, pointer.root_y)
        return abs(current.x - since_position.x) > tolerance or abs(current.y - since_position.y) > tolerance
