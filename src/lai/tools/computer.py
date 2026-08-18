"""Raw input and screen tools — the pixel-level fallback path.

Prefer the semantic ``ui_*`` tools when the app exposes accessibility. These
exist for everything else: Electron apps, games, VMs, canvases.
"""

from __future__ import annotations

from ..osl.geometry import Rect
from ..safety.policy import Risk
from .base import ToolContext, ToolRegistry, ToolResult

_COORD = {
    "x": {"type": "integer", "description": "Absolute screen X in pixels"},
    "y": {"type": "integer", "description": "Absolute screen Y in pixels"},
}


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "computer_screenshot",
        "Capture the screen. Returns an image plus its scale factor. Coordinates you "
        "read off the image must be divided by `scale` to get real screen coordinates — "
        "or just use the `screen_x`/`screen_y` values reported for elements. "
        "Scope 'focused' captures only the focused window (cheaper, usually enough).",
        {
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["focused", "desktop", "monitor", "region"],
                    "description": "What to capture. Default 'desktop'.",
                },
                "monitor": {"type": "integer", "description": "Monitor index for scope='monitor'", "minimum": 0},
                "region": {
                    "type": "object",
                    "description": "Rectangle {x,y,width,height} for scope='region'",
                    "properties": {
                        "x": {"type": "integer"}, "y": {"type": "integer"},
                        "width": {"type": "integer"}, "height": {"type": "integer"},
                    },
                },
                "annotate": {
                    "type": "boolean",
                    "description": "Draw numbered boxes over accessible elements (refs usable with ui_click)",
                },
                "max_edge": {"type": "integer", "minimum": 320, "maximum": 3000},
            }
        },
        risk=Risk.READ,
        group="computer",
        returns_image=True,
    )
    def screenshot(ctx: ToolContext, args: dict) -> ToolResult:
        desktop = ctx.desktop
        scope = args.get("scope", "desktop")
        max_edge = args.get("max_edge")
        region = None

        if scope == "focused":
            active = desktop.windows.active_window()
            if active and active.bounds.area > 0:
                region = active.bounds.intersection(desktop.screen.virtual_bounds())
        elif scope == "monitor":
            monitors = desktop.screen.monitors()
            index = int(args.get("monitor", 0))
            if not 0 <= index < len(monitors):
                return ToolResult.failure(
                    f"monitor {index} out of range; {len(monitors)} monitor(s) available"
                )
            region = monitors[index].bounds
        elif scope == "region":
            raw = args.get("region") or {}
            try:
                region = Rect.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                return ToolResult.failure("region must be an object with x, y, width, height")

        shot = desktop.screen.grab(region, max_edge=max_edge) if region or max_edge else desktop.screen.grab(region)

        if args.get("annotate"):
            snapshot = desktop.last_snapshot or desktop.snapshot()
            boxes = [(e.bounds, str(e.ref)) for e in snapshot.elements if e.interactive]
            if boxes:
                from ..osl.screen import annotate as draw  # noqa: PLC0415

                shot = draw(shot, boxes)

        info = shot.to_dict()
        active = desktop.windows.active_window()
        lines = [
            f"Screenshot of {shot.region.as_tuple()} rendered at {shot.size[0]}x{shot.size[1]} "
            f"(scale {shot.scale:.3f}).",
            f"Image pixel (ix,iy) maps to screen ({shot.region.x} + ix/{shot.scale:.3f}, "
            f"{shot.region.y} + iy/{shot.scale:.3f}).",
        ]
        if active:
            lines.append(f"Focused window: {active.title!r} [{active.wm_class}] at {active.bounds.as_tuple()}")
        return ToolResult(ok=True, content="\n".join(lines), data=info, images=[shot.png])

    @registry.tool(
        "computer_click",
        "Click at absolute screen coordinates. Use ui_click instead whenever the target "
        "has an accessible name — it is far more reliable than guessing pixels.",
        {
            "properties": {
                **_COORD,
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
                "count": {"type": "integer", "minimum": 1, "maximum": 3, "description": "1=single, 2=double"},
            },
            "required": ["x", "y"],
        },
        risk=Risk.INPUT,
        group="computer",
    )
    def click(ctx: ToolContext, args: dict) -> ToolResult:
        result = ctx.desktop.click(
            args["x"], args["y"], button=args.get("button", "left"), count=int(args.get("count", 1))
        )
        return ToolResult.text(f"Clicked {result.to_dict()}", **result.to_dict())

    @registry.tool(
        "computer_move",
        "Move the mouse pointer without clicking (useful to trigger hover states).",
        {"properties": _COORD, "required": ["x", "y"]},
        risk=Risk.INPUT,
        group="computer",
    )
    def move(ctx: ToolContext, args: dict) -> ToolResult:
        result = ctx.desktop.move(args["x"], args["y"])
        return ToolResult.text(f"Moved pointer to {args['x']},{args['y']}", **result.to_dict())

    @registry.tool(
        "computer_drag",
        "Press at one point, glide to another, release. Use for sliders, selections, "
        "drag-and-drop and resizing.",
        {
            "properties": {
                "from_x": {"type": "integer"}, "from_y": {"type": "integer"},
                "to_x": {"type": "integer"}, "to_y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
            },
            "required": ["from_x", "from_y", "to_x", "to_y"],
        },
        risk=Risk.INPUT,
        group="computer",
    )
    def drag(ctx: ToolContext, args: dict) -> ToolResult:
        result = ctx.desktop.drag(
            args["from_x"], args["from_y"], args["to_x"], args["to_y"],
            button=args.get("button", "left"),
        )
        return ToolResult.text(f"Dragged {result.to_dict()}", **result.to_dict())

    @registry.tool(
        "computer_scroll",
        "Scroll the wheel. Give x/y to scroll a specific pane rather than whatever has focus.",
        {
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "amount": {"type": "integer", "minimum": 1, "maximum": 30, "description": "Wheel clicks (default 3)"},
                **_COORD,
            },
            "required": ["direction"],
        },
        risk=Risk.INPUT,
        group="computer",
    )
    def scroll(ctx: ToolContext, args: dict) -> ToolResult:
        result = ctx.desktop.scroll(
            args["direction"], amount=int(args.get("amount", 3)), x=args.get("x"), y=args.get("y")
        )
        return ToolResult.text(f"Scrolled {args['direction']}", **result.to_dict())

    @registry.tool(
        "computer_type",
        "Type text into whatever currently has keyboard focus. Unicode-safe. "
        "Click or focus the target field first.",
        {
            "properties": {
                "text": {"type": "string"},
                "delay_ms": {"type": "integer", "minimum": 0, "maximum": 500},
            },
            "required": ["text"],
        },
        risk=Risk.INPUT,
        group="computer",
    )
    def type_text(ctx: ToolContext, args: dict) -> ToolResult:
        delay = args.get("delay_ms")
        result = ctx.desktop.input.type_text(args["text"], delay_ms=delay)
        return ToolResult.text(f"Typed {len(args['text'])} characters", **result.to_dict())

    @registry.tool(
        "computer_key",
        "Press a key or chord, e.g. 'Return', 'Escape', 'ctrl+s', 'alt+F4', 'super', "
        "'ctrl+shift+t'. Names follow X keysyms; common aliases (enter, esc, tab) work.",
        {
            "properties": {
                "key": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": ["key"],
        },
        risk=Risk.INPUT,
        group="computer",
    )
    def key(ctx: ToolContext, args: dict) -> ToolResult:
        result = ctx.desktop.key(args["key"], count=int(args.get("count", 1)))
        return ToolResult.text(f"Pressed {result.to_dict()['key']}", **result.to_dict())

    @registry.tool(
        "computer_cursor",
        "Report the current mouse pointer position and which window is under it.",
        {"properties": {}},
        risk=Risk.READ,
        group="computer",
    )
    def cursor(ctx: ToolContext, args: dict) -> ToolResult:
        point = ctx.desktop.input.cursor_position()
        window = ctx.desktop.windows.window_at(point)
        return ToolResult.json(
            {
                "x": point.x,
                "y": point.y,
                "window": window.to_dict() if window else None,
            }
        )
