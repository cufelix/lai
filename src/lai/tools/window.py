"""Window management tools."""

from __future__ import annotations

from ..osl.geometry import Rect
from ..safety.policy import Risk
from .base import ToolContext, ToolRegistry, ToolResult

_WINDOW_TARGET = {
    "id": {"type": "integer", "description": "Window id from window_list"},
    "match": {"type": "string", "description": "Substring of the window title or app class"},
}


def _pick(ctx: ToolContext, args: dict):
    if args.get("id") is not None:
        return ctx.desktop.windows.get(int(args["id"]))
    if args.get("match"):
        return ctx.desktop.windows.find_one(args["match"])
    active = ctx.desktop.windows.active_window()
    if active is None:
        raise ValueError("no window specified and nothing is focused")
    return active


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "window_list",
        "List open windows: id, title, application class, pid, geometry and state. "
        "This is your map of what is currently running on screen.",
        {
            "properties": {
                "match": {"type": "string", "description": "Filter by title/class substring"},
                "include_invisible": {"type": "boolean"},
            }
        },
        risk=Risk.READ,
        group="window",
    )
    def window_list(ctx: ToolContext, args: dict) -> ToolResult:
        windows = ctx.desktop.windows.list_windows(
            include_invisible=bool(args.get("include_invisible", False))
        )
        if args.get("match"):
            windows = [w for w in windows if w.matches(args["match"])]
        if not windows:
            return ToolResult.text("No matching windows.")
        lines = [
            f"{'*' if w.active else ' '} id={w.id} [{w.wm_class or '?'}] {w.title[:70]!r} "
            f"at {w.bounds.as_tuple()}" + (f" {','.join(w.states)}" if w.states else "")
            for w in windows
        ]
        return ToolResult(
            ok=True,
            content=f"{len(windows)} window(s) (* = focused):\n" + "\n".join(lines),
            data={"windows": [w.to_dict() for w in windows]},
        )

    @registry.tool(
        "window_focus",
        "Raise a window and give it keyboard focus. Do this before typing into an app.",
        {"properties": _WINDOW_TARGET},
        risk=Risk.INPUT,
        group="window",
    )
    def window_focus(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            window = _pick(ctx, args)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        focused = ctx.desktop.windows.focus(window.id)
        return ToolResult.text(f"Focused {focused.title!r} [{focused.wm_class}]", **focused.to_dict())

    @registry.tool(
        "window_close",
        "Ask a window to close (the same as clicking its X). The app may show a "
        "'save changes?' dialog — check the result and handle it.",
        {"properties": _WINDOW_TARGET},
        risk=Risk.WRITE,
        group="window",
    )
    def window_close(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            window = _pick(ctx, args)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        out = ctx.desktop.close_window(window.id)
        note = "" if out["gone"] else " — the window is still open; it probably raised a confirmation dialog."
        return ToolResult.text(f"Sent close to {window.title!r}.{note}", **out)

    @registry.tool(
        "window_arrange",
        "Move, resize or change a window's state (maximize, fullscreen, minimize).",
        {
            "properties": {
                **_WINDOW_TARGET,
                "state": {
                    "type": "string",
                    "enum": ["maximized", "fullscreen", "minimize", "restore", "above"],
                },
                "bounds": {
                    "type": "object",
                    "description": "Explicit geometry {x,y,width,height}",
                    "properties": {
                        "x": {"type": "integer"}, "y": {"type": "integer"},
                        "width": {"type": "integer"}, "height": {"type": "integer"},
                    },
                },
            }
        },
        risk=Risk.INPUT,
        group="window",
    )
    def window_arrange(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            window = _pick(ctx, args)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        manager = ctx.desktop.windows
        if args.get("bounds"):
            try:
                rect = Rect.from_dict(args["bounds"])
            except (KeyError, TypeError, ValueError):
                return ToolResult.failure("bounds must have x, y, width and height")
            updated = manager.move_resize(window.id, rect)
            return ToolResult.text(f"Moved {window.title!r} to {updated.bounds.as_tuple()}", **updated.to_dict())

        state = args.get("state")
        if not state:
            return ToolResult.failure("provide either 'state' or 'bounds'")
        if state == "minimize":
            manager.minimize(window.id)
            return ToolResult.text(f"Minimized {window.title!r}")
        if state == "restore":
            updated = manager.set_state(window.id, "maximized", False)
            return ToolResult.text(f"Restored {window.title!r}", **updated.to_dict())
        updated = manager.set_state(window.id, state, True)
        return ToolResult.text(f"Set {state} on {window.title!r}", **updated.to_dict())
