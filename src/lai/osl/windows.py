"""Window management via X11 / EWMH.

This is the desktop's equivalent of a browser's tab list: enumerate windows,
know which one is focused, and be able to raise, move, resize or close them.
Implemented on python-xlib directly (no subprocess per query) so the agent can
poll the window tree cheaply between actions.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ..errors import BackendUnavailable, DisplayError, InvalidToolInput, WindowNotFound
from .geometry import Point, Rect

_STATE_ATOMS = {
    "maximized_vert": "_NET_WM_STATE_MAXIMIZED_VERT",
    "maximized_horz": "_NET_WM_STATE_MAXIMIZED_HORZ",
    "fullscreen": "_NET_WM_STATE_FULLSCREEN",
    "hidden": "_NET_WM_STATE_HIDDEN",
    "above": "_NET_WM_STATE_ABOVE",
    "below": "_NET_WM_STATE_BELOW",
    "sticky": "_NET_WM_STATE_STICKY",
    "skip_taskbar": "_NET_WM_STATE_SKIP_TASKBAR",
    "modal": "_NET_WM_STATE_MODAL",
}

_ACTION_REMOVE, _ACTION_ADD, _ACTION_TOGGLE = 0, 1, 2
_SOURCE_APP = 1


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """A snapshot of one top-level window."""

    id: int
    title: str
    wm_class: str
    instance: str
    pid: int | None
    bounds: Rect
    desktop: int | None
    states: tuple[str, ...] = field(default=())
    active: bool = False

    @property
    def hex_id(self) -> str:
        return f"0x{self.id:08x}"

    @property
    def visible(self) -> bool:
        return "hidden" not in self.states and self.bounds.area > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hex_id": self.hex_id,
            "title": self.title,
            "app": self.wm_class,
            "instance": self.instance,
            "pid": self.pid,
            "bounds": self.bounds.to_dict(),
            "desktop": self.desktop,
            "states": list(self.states),
            "active": self.active,
            "visible": self.visible,
        }

    def matches(self, query: str) -> bool:
        needle = query.lower().strip()
        if not needle:
            return False
        return (
            needle in self.title.lower()
            or needle in self.wm_class.lower()
            or needle in self.instance.lower()
            or needle == self.hex_id
            or needle == str(self.id)
        )


class WindowManager:
    """EWMH client. Lazily connects so import never touches the display."""

    def __init__(self, display_name: str | None = None) -> None:
        self._display_name = display_name or os.environ.get("DISPLAY")
        self._display = None
        self._root = None
        self._atoms: dict[str, int] = {}

    # -- connection ------------------------------------------------------

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
        return self._display

    def close(self) -> None:
        if self._display is not None:
            try:
                self._display.close()
            finally:
                self._display = None
                self._root = None
                self._atoms.clear()

    def _atom(self, name: str) -> int:
        if name not in self._atoms:
            self._atoms[name] = self._conn().intern_atom(name)
        return self._atoms[name]

    def _prop(self, window, name: str, kind: int = 0):
        try:
            prop = window.get_full_property(self._atom(name), kind)
        except Exception:
            return None
        return prop.value if prop else None

    # -- reading ---------------------------------------------------------

    def list_windows(self, *, include_invisible: bool = False) -> list[WindowInfo]:
        self._conn()
        from Xlib import Xatom  # noqa: PLC0415

        ids = self._prop(self._root, "_NET_CLIENT_LIST", Xatom.WINDOW) or []
        active = self.active_window_id()
        out: list[WindowInfo] = []
        for wid in ids:
            info = self._describe(int(wid), active_id=active)
            if info and (include_invisible or info.visible):
                out.append(info)
        return out

    def active_window_id(self) -> int | None:
        self._conn()
        from Xlib import Xatom  # noqa: PLC0415

        value = self._prop(self._root, "_NET_ACTIVE_WINDOW", Xatom.WINDOW)
        if not value:
            return None
        wid = int(value[0])
        return wid or None

    def active_window(self) -> WindowInfo | None:
        wid = self.active_window_id()
        return self._describe(wid, active_id=wid) if wid else None

    def get(self, window_id: int) -> WindowInfo:
        info = self._describe(int(window_id), active_id=self.active_window_id())
        if info is None:
            raise WindowNotFound(f"window {window_id} does not exist")
        return info

    def find(self, query: str, *, include_invisible: bool = False) -> list[WindowInfo]:
        return [w for w in self.list_windows(include_invisible=include_invisible) if w.matches(query)]

    def find_one(self, query: str) -> WindowInfo:
        matches = self.find(query)
        if not matches:
            raise WindowNotFound(f"no window matching {query!r}")
        # Prefer the active window, then the largest — the most likely main window.
        matches.sort(key=lambda w: (w.active, w.bounds.area), reverse=True)
        return matches[0]

    def _describe(self, window_id: int, *, active_id: int | None) -> WindowInfo | None:
        display = self._conn()
        from Xlib import Xatom  # noqa: PLC0415

        try:
            window = display.create_resource_object("window", window_id)
            geom = window.get_geometry()
        except Exception:
            return None

        title = ""
        raw_title = self._prop(window, "_NET_WM_NAME")
        if raw_title:
            title = _decode(raw_title)
        if not title:
            try:
                title = window.get_wm_name() or ""
            except Exception:
                title = ""

        wm_class, instance = "", ""
        try:
            klass = window.get_wm_class()
            if klass:
                instance, wm_class = klass[0] or "", klass[1] or ""
        except Exception:
            pass

        pid_val = self._prop(window, "_NET_WM_PID", Xatom.CARDINAL)
        pid = int(pid_val[0]) if pid_val else None

        desktop_val = self._prop(window, "_NET_WM_DESKTOP", Xatom.CARDINAL)
        desktop = int(desktop_val[0]) if desktop_val else None

        state_atoms = self._prop(window, "_NET_WM_STATE", Xatom.ATOM) or []
        wanted = {self._atom(v): k for k, v in _STATE_ATOMS.items()}
        states = tuple(sorted(wanted[a] for a in state_atoms if a in wanted))

        try:
            coords = window.translate_coords(self._root, 0, 0)
            abs_x, abs_y = -coords.x, -coords.y
        except Exception:
            abs_x, abs_y = geom.x, geom.y

        bounds = Rect(abs_x, abs_y, geom.width, geom.height)

        # GTK client-side-decorated windows include an invisible shadow border
        # in their X geometry, so the raw rectangle is larger than the window
        # the user sees — often extending off-screen. Trim it, otherwise the
        # agent aims clicks at pixels that belong to nothing.
        extents = self._prop(window, "_GTK_FRAME_EXTENTS", Xatom.CARDINAL)
        if extents and len(extents) >= 4:
            left, right, top, bottom = (int(v) for v in extents[:4])
            trimmed = Rect(
                bounds.x + left,
                bounds.y + top,
                bounds.width - left - right,
                bounds.height - top - bottom,
            )
            if trimmed.width > 0 and trimmed.height > 0:
                bounds = trimmed

        return WindowInfo(
            id=window_id,
            title=title,
            wm_class=wm_class,
            instance=instance,
            pid=pid,
            bounds=bounds,
            desktop=desktop,
            states=states,
            active=(active_id == window_id),
        )

    # -- writing ---------------------------------------------------------

    def _client_message(self, window, msg: str, data: Iterable[int]) -> None:
        display = self._conn()
        from Xlib import X  # noqa: PLC0415
        from Xlib.protocol import event  # noqa: PLC0415

        payload = list(data)[:5]
        payload += [0] * (5 - len(payload))
        message = event.ClientMessage(
            window=window, client_type=self._atom(msg), data=(32, payload)
        )
        self._root.send_event(
            message, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask
        )
        display.flush()

    def _window(self, window_id: int):
        return self._conn().create_resource_object("window", int(window_id))

    def focus(self, window_id: int) -> WindowInfo:
        window = self._window(window_id)
        desktop = self._prop(window, "_NET_WM_DESKTOP")
        if desktop:
            self._client_message(self._root, "_NET_CURRENT_DESKTOP", [int(desktop[0]), 0])
        self._client_message(window, "_NET_ACTIVE_WINDOW", [_SOURCE_APP, int(time.time()), 0])
        try:
            window.configure(stack_mode=0)  # X.Above
            self._conn().flush()
        except Exception:
            pass
        time.sleep(0.12)
        return self.get(window_id)

    def close_window(self, window_id: int) -> None:
        """Ask the window manager to close a window (distinct from :meth:`close`,
        which releases our X11 connection)."""
        self._client_message(
            self._window(window_id), "_NET_CLOSE_WINDOW", [int(time.time()), _SOURCE_APP, 0]
        )

    def move_resize(self, window_id: int, bounds: Rect) -> WindowInfo:
        # gravity(0) | flags for x,y,w,h (bits 8..11) | source (bit 12)
        flags = (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (_SOURCE_APP << 12)
        self._client_message(
            self._window(window_id),
            "_NET_MOVERESIZE_WINDOW",
            [flags, bounds.x, bounds.y, bounds.width, bounds.height],
        )
        self._conn().flush()
        return self.settle(window_id)

    def set_state(self, window_id: int, state: str, enabled: bool | None = None) -> WindowInfo:
        if state == "maximized":
            atoms = [self._atom(_STATE_ATOMS["maximized_vert"]), self._atom(_STATE_ATOMS["maximized_horz"])]
        else:
            key = _STATE_ATOMS.get(state)
            if key is None:
                raise WindowNotFound(
                    f"unknown window state {state!r}; known: {[*sorted(_STATE_ATOMS), 'maximized']}"
                )
            atoms = [self._atom(key)]
        action = _ACTION_TOGGLE if enabled is None else (_ACTION_ADD if enabled else _ACTION_REMOVE)
        self._client_message(
            self._window(window_id), "_NET_WM_STATE", [action, *atoms[:2], _SOURCE_APP]
        )
        return self.settle(window_id)

    def settle(self, window_id: int, *, timeout: float = 1.2, poll: float = 0.06) -> WindowInfo:
        """Wait until a window stops moving, then report where it ended up.

        A window manager animates: it acknowledges a resize immediately and
        finishes it a few frames later. Returning the geometry from the middle
        of that is how an agent reads a half-changed window — the accessibility
        tree is briefly inconsistent too, so the next `ui_click` misses.
        Two identical readings in a row means the motion is over.
        """
        deadline = time.monotonic() + timeout
        previous = None
        while True:
            time.sleep(poll)
            current = self.get(window_id)
            if previous is not None and current.bounds == previous.bounds and current.states == previous.states:
                return current
            if time.monotonic() >= deadline:
                return current
            previous = current

    def minimize(self, window_id: int) -> None:
        # WM_CHANGE_STATE with IconicState(3) is the portable way to iconify.
        self._client_message(self._window(window_id), "WM_CHANGE_STATE", [3, 0, 0, 0, 0])

    # -- waiting ---------------------------------------------------------

    def wait_for(
        self,
        predicate: Callable[[WindowInfo], bool],
        *,
        timeout: float = 20.0,
        poll: float = 0.25,
        ignore_ids: set[int] | None = None,
    ) -> WindowInfo | None:
        """Poll until a window satisfies ``predicate``. Returns None on timeout."""
        deadline = time.monotonic() + timeout
        skip = ignore_ids or set()
        while time.monotonic() < deadline:
            for window in self.list_windows():
                if window.id in skip:
                    continue
                try:
                    if predicate(window):
                        return window
                except Exception:
                    continue
            time.sleep(poll)
        return None

    def window_at(self, point: Point) -> WindowInfo | None:
        """Topmost window containing ``point`` (client-list order is bottom-to-top)."""
        hit = [w for w in self.list_windows() if w.bounds.contains(point)]
        if not hit:
            return None
        return hit[-1]

    # -- workspaces --------------------------------------------------------

    def workspace_count(self) -> int:
        self._conn()
        from Xlib import Xatom  # noqa: PLC0415

        value = self._prop(self._root, "_NET_NUMBER_OF_DESKTOPS", Xatom.CARDINAL)
        return int(value[0]) if value else 0

    def current_workspace(self) -> int:
        self._conn()
        from Xlib import Xatom  # noqa: PLC0415

        value = self._prop(self._root, "_NET_CURRENT_DESKTOP", Xatom.CARDINAL)
        return int(value[0]) if value else 0

    def workspace_names(self) -> list[str]:
        """Names published via ``_NET_DESKTOP_NAMES``; empty if the WM doesn't set it."""
        self._conn()
        raw = self._prop(self._root, "_NET_DESKTOP_NAMES")
        if not raw:
            return []
        # A UTF8_STRING property packing multiple names as NUL-separated
        # segments (some WMs omit the final terminator, hence the filter).
        return [name for name in _decode(raw).split("\x00") if name]

    def _validate_workspace_index(self, index: int) -> None:
        count = self.workspace_count()
        if count and not 0 <= index < count:
            raise InvalidToolInput(f"workspace {index} out of range (have {count})")

    def switch_workspace(self, index: int) -> int:
        self._validate_workspace_index(index)
        self._client_message(self._root, "_NET_CURRENT_DESKTOP", [int(index), int(time.time())])
        self._conn().flush()
        time.sleep(0.12)
        return self.current_workspace()

    def move_window_to_workspace(self, window_id: int, index: int) -> WindowInfo:
        self._validate_workspace_index(index)
        self._client_message(self._window(window_id), "_NET_WM_DESKTOP", [int(index), _SOURCE_APP])
        self._conn().flush()
        return self.settle(window_id)


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
