"""AT-SPI accessibility layer — the desktop's DOM.

This is what separates LAI from a pixel-guessing agent. Where a browser agent
reads the DOM, LAI reads the AT-SPI2 tree: every button, entry and menu item with
its role, name, state and exact screen bounds. The agent can then say
*"click the Save button"* instead of *"click at 743,912 and hope"*.

Apps without a11y (Chrome without ``--force-renderer-accessibility``, Electron,
games, VMs) degrade gracefully: the snapshot comes back empty and the agent falls
back to screenshot + coordinates.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..errors import BackendUnavailable, ElementNotFound
from .geometry import Point, Rect

# Roles a user can actually act on.
INTERACTIVE_ROLES: frozenset[str] = frozenset(
    {
        "push button", "toggle button", "check box", "radio button", "menu item",
        "check menu item", "radio menu item", "menu", "combo box", "entry", "text",
        "password text", "link", "list item", "tab", "page tab", "slider",
        "spin button", "tree item", "table cell", "icon", "list box", "calendar",
        "color chooser", "date editor", "file chooser", "editbar", "embedded",
        "scroll bar", "split pane", "toggle switch",
    }
)

# Roles worth showing for context even though you don't click them.
READABLE_ROLES: frozenset[str] = frozenset(
    {"label", "heading", "static", "paragraph", "terminal", "document web", "document frame",
     "status bar", "tool tip", "alert", "dialog", "frame", "window", "notification"}
)

# Never worth listing: pure structure.
CONTAINER_ROLES: frozenset[str] = frozenset(
    {"filler", "panel", "section", "redundant object", "unknown", "invalid", "layered pane",
     "scroll pane", "viewport", "tool bar", "menu bar", "table", "tree", "list", "application"}
)

DEFAULT_MAX_ELEMENTS = 220
DEFAULT_MAX_DEPTH = 22
DEFAULT_NODE_BUDGET = 4000
DEFAULT_TIMEOUT_MS = 800


@dataclass(frozen=True, slots=True)
class Element:
    """One accessible object, flattened for the model."""

    ref: int
    role: str
    name: str
    bounds: Rect
    states: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    description: str = ""
    value: str | None = None
    path: tuple[int, ...] = ()
    app: str = ""
    pid: int | None = None
    depth: int = 0

    @property
    def interactive(self) -> bool:
        return self.role in INTERACTIVE_ROLES or bool(self.actions)

    @property
    def enabled(self) -> bool:
        return "enabled" in self.states or "sensitive" in self.states

    @property
    def center(self) -> Point:
        return self.bounds.center

    def to_dict(self) -> dict:
        out: dict = {
            "ref": self.ref,
            "role": self.role,
            "name": self.name,
            "bounds": self.bounds.to_dict(),
        }
        if self.value is not None:
            out["value"] = self.value
        if self.description:
            out["description"] = self.description
        if self.actions:
            out["actions"] = list(self.actions)
        interesting = [s for s in self.states if s in ("focused", "selected", "checked", "expanded")]
        if interesting:
            out["states"] = interesting
        if not self.enabled:
            out["disabled"] = True
        return out

    def to_line(self) -> str:
        """Compact one-line rendering — this is what the model actually reads."""
        parts = [f"[{self.ref}]", self.role]
        if self.name:
            parts.append(f'"{self.name}"')
        if self.value:
            parts.append(f"value={self.value!r}")
        flags = [s for s in ("focused", "selected", "checked", "expanded") if s in self.states]
        if not self.enabled:
            flags.append("disabled")
        if flags:
            parts.append(f"({','.join(flags)})")
        c = self.center
        parts.append(f"@{c.x},{c.y}")
        return " ".join(parts)

    def matches(self, query: str, *, role: str | None = None) -> bool:
        if role and self.role != role.lower():
            return False
        if not query:
            return True
        needle = query.lower()
        return (
            needle in self.name.lower()
            or needle in self.description.lower()
            or (self.value is not None and needle in str(self.value).lower())
        )


@dataclass(slots=True)
class Snapshot:
    """A flattened a11y tree plus the live handles needed to act on it."""

    elements: list[Element] = field(default_factory=list)
    captured_at: float = field(default_factory=time.time)
    app: str = ""
    truncated: bool = False
    _handles: dict[int, object] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self) -> Iterator[Element]:
        return iter(self.elements)

    def get(self, ref: int) -> Element:
        for element in self.elements:
            if element.ref == ref:
                return element
        raise ElementNotFound(f"no element with ref={ref} in this snapshot")

    def handle(self, ref: int):
        handle = self._handles.get(ref)
        if handle is None:
            raise ElementNotFound(f"no live handle for ref={ref}; take a fresh ui_snapshot")
        return handle

    def find(self, query: str = "", *, role: str | None = None, interactive: bool | None = None) -> list[Element]:
        found = [e for e in self.elements if e.matches(query, role=role)]
        if interactive is not None:
            found = [e for e in found if e.interactive == interactive]
        return found

    def find_one(self, query: str, *, role: str | None = None) -> Element:
        matches = self.find(query, role=role)
        if not matches:
            if role:
                # The name may well be there under a different role, and that
                # is exactly the next thing worth knowing: "no element matching
                # 'Calculator' with role 'text'" is a dead end when there is a
                # Calculator sitting right there as a frame.
                same_name = self.find(query)
                if same_name:
                    roles = sorted({element.role for element in same_name})
                    raise ElementNotFound(
                        f"no element matching {query!r} with role {role!r}",
                        detail=f"{query!r} does exist, with role(s): {', '.join(roles)} — "
                               f"ask for one of those, or drop the role filter",
                    )
            raise ElementNotFound(
                f"no element matching {query!r}"
                + (f" with role {role!r}" if role else "")
            )
        # Exact name match wins, then interactive, then smallest (most specific).
        exact = [e for e in matches if e.name.lower() == query.lower()]
        pool = exact or matches
        pool.sort(key=lambda e: (not e.interactive, e.bounds.area))
        return pool[0]

    def render(self, *, limit: int = DEFAULT_MAX_ELEMENTS, interactive_only: bool = False) -> str:
        rows = [e for e in self.elements if e.interactive] if interactive_only else self.elements
        lines = [e.to_line() for e in rows[:limit]]
        if len(rows) > limit or self.truncated:
            lines.append(f"... ({len(rows)} elements total, truncated)")
        if not lines:
            return "(no accessible elements — this app exposes no a11y tree; use screenshot + coordinates)"
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "app": self.app,
            "count": len(self.elements),
            "truncated": self.truncated,
            "elements": [e.to_dict() for e in self.elements],
        }


class A11yTree:
    """Reads and acts on the AT-SPI2 desktop tree."""

    def __init__(self, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        self._atspi = None
        self._ref_counter = 0
        self._timeout_ms = timeout_ms

    def _api(self):
        if self._atspi is None:
            try:
                import gi  # noqa: PLC0415

                gi.require_version("Atspi", "2.0")
                from gi.repository import Atspi  # noqa: PLC0415
            except Exception as exc:  # pragma: no cover - env dependent
                raise BackendUnavailable(
                    "AT-SPI (gi/Atspi) is unavailable",
                    detail="install with: sudo apt install python3-gi gir1.2-atspi-2.0",
                ) from exc
            # Never let an unresponsive app block the agent loop.
            if hasattr(Atspi, "set_timeout"):
                try:
                    Atspi.set_timeout(self._timeout_ms, self._timeout_ms * 3)
                except Exception:
                    pass
            self._atspi = Atspi
        return self._atspi

    @property
    def available(self) -> bool:
        try:
            self._api()
            return True
        except BackendUnavailable:
            return False

    def applications(self) -> list[tuple[str, int | None, object]]:
        """(name, pid, accessible) for every registered application."""
        atspi = self._api()
        desktop = atspi.get_desktop(0)
        out: list[tuple[str, int | None, object]] = []
        for i in range(_safe_count(desktop)):
            app = _safe_child(desktop, i)
            if app is None:
                continue
            try:
                pid = app.get_process_id()
            except Exception:
                pid = None
            out.append((_safe_name(app), pid, app))
        return out

    def snapshot(
        self,
        *,
        pid: int | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
        within: Rect | None = None,
        max_elements: int = DEFAULT_MAX_ELEMENTS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        node_budget: int = DEFAULT_NODE_BUDGET,
        interactive_only: bool = False,
        visible_only: bool = True,
    ) -> Snapshot:
        """Flatten the a11y tree for one app (or the whole desktop).

        Filters are ANDed. ``within`` clips to a screen rectangle — pass the
        focused window's bounds to get exactly what the user is looking at.
        """
        atspi = self._api()
        roots: list[tuple[str, int | None, object]] = []
        for name, app_pid, accessible in self.applications():
            if pid is not None and app_pid != pid:
                continue
            if app_name and app_name.lower() not in name.lower():
                continue
            roots.append((name, app_pid, accessible))

        snapshot = Snapshot(app=app_name or (roots[0][0] if len(roots) == 1 else "desktop"))
        budget = [node_budget]

        for name, app_pid, accessible in roots:
            targets: list[object] = []
            if window_title:
                for i in range(_safe_count(accessible)):
                    frame = _safe_child(accessible, i)
                    if frame is None:
                        continue
                    if window_title.lower() in _safe_name(frame).lower():
                        targets.append(frame)
                if not targets:
                    continue
            else:
                targets = [accessible]

            for target in targets:
                self._walk(
                    atspi,
                    target,
                    snapshot,
                    app_name=name,
                    pid=app_pid,
                    depth=0,
                    path=(),
                    max_depth=max_depth,
                    max_elements=max_elements,
                    budget=budget,
                    within=within,
                    interactive_only=interactive_only,
                    visible_only=visible_only,
                )
                if len(snapshot.elements) >= max_elements or budget[0] <= 0:
                    snapshot.truncated = True
                    break
            if len(snapshot.elements) >= max_elements or budget[0] <= 0:
                snapshot.truncated = True
                break

        return snapshot

    def _walk(
        self,
        atspi,
        node,
        snapshot: Snapshot,
        *,
        app_name: str,
        pid: int | None,
        depth: int,
        path: tuple[int, ...],
        max_depth: int,
        max_elements: int,
        budget: list[int],
        within: Rect | None,
        interactive_only: bool,
        visible_only: bool,
    ) -> None:
        if depth > max_depth or budget[0] <= 0 or len(snapshot.elements) >= max_elements:
            return
        budget[0] -= 1

        try:
            role = node.get_role_name() or "unknown"
        except Exception:
            return

        states = _states(atspi, node)
        showing = "showing" in states and "visible" in states

        bounds = _extents(atspi, node)
        in_region = True
        if within is not None and bounds is not None:
            in_region = bounds.intersects(within)

        keep = role not in CONTAINER_ROLES and (role in INTERACTIVE_ROLES or role in READABLE_ROLES)
        if interactive_only:
            keep = keep and role in INTERACTIVE_ROLES
        if visible_only and not showing:
            keep = False
        if bounds is None or bounds.area <= 0:
            keep = False
        if not in_region:
            keep = False

        if keep:
            name = _safe_name(node)
            value = _value(atspi, node, role)
            if value is not None and value == name:
                value = None  # labels expose their text as the name; don't say it twice
            if name or value or role in ("entry", "text", "password text", "terminal"):
                self._ref_counter += 1
                ref = self._ref_counter
                element = Element(
                    ref=ref,
                    role=role,
                    name=name,
                    bounds=bounds,
                    states=states,
                    actions=_actions(node),
                    description=_safe_description(node),
                    value=value,
                    path=path,
                    app=app_name,
                    pid=pid,
                    depth=depth,
                )
                snapshot.elements.append(element)
                snapshot._handles[ref] = node

        # Don't descend into invisible subtrees — that's where the time goes.
        if visible_only and bounds is not None and bounds.area == 0 and depth > 1:
            return
        if within is not None and bounds is not None and not in_region and depth > 1:
            return

        count = _safe_count(node)
        for index in range(min(count, 200)):
            child = _safe_child(node, index)
            if child is None:
                continue
            self._walk(
                atspi,
                child,
                snapshot,
                app_name=app_name,
                pid=pid,
                depth=depth + 1,
                path=(*path, index),
                max_depth=max_depth,
                max_elements=max_elements,
                budget=budget,
                within=within,
                interactive_only=interactive_only,
                visible_only=visible_only,
            )
            if len(snapshot.elements) >= max_elements or budget[0] <= 0:
                snapshot.truncated = True
                return

    # -- acting ----------------------------------------------------------

    def do_action(self, handle, preferred: tuple[str, ...] = ("click", "press", "activate", "jump")) -> str:
        """Invoke the element's own action. Returns the action name used."""
        names = _actions(handle)
        if not names:
            raise ElementNotFound("element exposes no accessible actions")
        for want in preferred:
            for index, name in enumerate(names):
                if name.lower() == want:
                    handle.do_action(index)
                    return name
        handle.do_action(0)
        return names[0]

    def focus(self, handle) -> bool:
        try:
            return bool(handle.grab_focus())
        except Exception:
            return False

    def set_text(self, handle, text: str) -> bool:
        """Replace an editable element's contents via the EditableText interface."""
        atspi = self._api()
        last_error: Exception | None = None
        try:
            iface = handle.get_editable_text_iface()
        except Exception as exc:  # pragma: no cover - toolkit dependent
            iface, last_error = None, exc
        if iface is not None:
            try:
                return bool(atspi.EditableText.set_text_contents(iface, text))
            except Exception as exc:
                last_error = exc
        # Older bindings expose the method straight on the accessible.
        try:
            return bool(handle.set_text_contents(text))
        except Exception as exc:
            last_error = exc
        raise ElementNotFound(
            "element is not editable via AT-SPI", detail=str(last_error) if last_error else None
        )

    def read_text(self, handle, *, limit: int = 4000) -> str | None:
        """Read an element's text via the Text interface."""
        return _text_of(self._api(), handle, limit=limit)

    def element_at(self, point: Point, *, snapshot: Snapshot) -> Element | None:
        """Smallest element in ``snapshot`` containing ``point``."""
        hits = [e for e in snapshot.elements if e.bounds.contains(point)]
        if not hits:
            return None
        hits.sort(key=lambda e: e.bounds.area)
        return hits[0]


# -- defensive AT-SPI helpers -------------------------------------------
# Every call can raise or return None if the target app dies mid-walk.


def _safe_count(node) -> int:
    try:
        return int(node.get_child_count() or 0)
    except Exception:
        return 0


def _safe_child(node, index: int):
    try:
        return node.get_child_at_index(index)
    except Exception:
        return None


def _safe_name(node) -> str:
    try:
        return (node.get_name() or "").strip()
    except Exception:
        return ""


def _safe_description(node) -> str:
    try:
        return (node.get_description() or "").strip()[:200]
    except Exception:
        return ""


def _states(atspi, node) -> tuple[str, ...]:
    try:
        state_set = node.get_state_set()
    except Exception:
        return ()
    out: list[str] = []
    for name in (
        "SHOWING", "VISIBLE", "ENABLED", "SENSITIVE", "FOCUSED", "FOCUSABLE",
        "SELECTED", "CHECKED", "EXPANDED", "EDITABLE", "PRESSED",
    ):
        try:
            if state_set.contains(getattr(atspi.StateType, name)):
                out.append(name.lower())
        except Exception:
            continue
    return tuple(out)


def _extents(atspi, node) -> Rect | None:
    try:
        ext = node.get_extents(atspi.CoordType.SCREEN)
    except Exception:
        return None
    if ext is None:
        return None
    try:
        return Rect(int(ext.x), int(ext.y), int(ext.width), int(ext.height))
    except Exception:
        return None


def _actions(node) -> tuple[str, ...]:
    try:
        count = int(node.get_n_actions() or 0)
    except Exception:
        return ()
    names: list[str] = []
    for i in range(min(count, 8)):
        try:
            names.append((node.get_action_name(i) or "").strip())
        except Exception:
            continue
    return tuple(n for n in names if n)


TEXT_ROLES: frozenset[str] = frozenset(
    {"entry", "text", "password text", "terminal", "document web", "document frame",
     "paragraph", "label", "static", "heading", "list item", "table cell"}
)

VALUE_ROLES: frozenset[str] = frozenset(
    {"slider", "spin button", "progress bar", "scroll bar", "level bar"}
)


def _text_of(atspi, node, *, limit: int = 400) -> str | None:
    """Read text through the AT-SPI Text interface.

    ``Accessible.get_text`` is deprecated and raises TypeError on modern
    bindings; the interface object is the supported path.
    """
    try:
        iface = node.get_text_iface()
    except Exception:
        iface = None
    if iface is None:
        return None
    try:
        count = int(atspi.Text.get_character_count(iface) or 0)
    except Exception:
        return None
    if count <= 0:
        return None
    try:
        text = atspi.Text.get_text(iface, 0, min(count, limit))
    except Exception:
        return None
    if not text:
        return None
    cleaned = text.strip()
    return cleaned or None


def _value(atspi, node, role: str) -> str | None:
    if role in TEXT_ROLES:
        text = _text_of(atspi, node)
        if text:
            return text
    if role in VALUE_ROLES:
        try:
            return str(node.get_current_value())
        except Exception:
            pass
    return None
