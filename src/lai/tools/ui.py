"""Semantic UI tools — read and drive apps through the accessibility tree.

This is the preferred path. ``ui_snapshot`` returns a numbered element list
(role, name, value, exact bounds); ``ui_click``/``ui_type`` act on those refs or
on a plain name. No pixel guessing, no coordinate drift when a window moves.
"""

from __future__ import annotations

from ..safety.policy import Risk
from .base import ToolContext, ToolRegistry, ToolResult

_TARGET = {
    "ref": {"type": "integer", "description": "Element ref from the most recent ui_snapshot"},
    "name": {
        "type": "string",
        "description": "Accessible name to match instead of a ref (substring, case-insensitive)",
    },
    "role": {"type": "string", "description": "Narrow a name match by role, e.g. 'push button', 'entry', 'menu item'"},
}


def _resolve_target(ctx: ToolContext, args: dict):
    ref, name = args.get("ref"), args.get("name")
    if ref is None and not name:
        raise ValueError("provide either 'ref' (from ui_snapshot) or 'name'")
    return ref if ref is not None else name


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "ui_snapshot",
        "Read the accessibility tree of the focused app: every button, field, menu item "
        "and label with its role, name, value and exact screen position. This is the "
        "desktop's equivalent of a DOM — always try this before taking a screenshot. "
        "Returns numbered refs to use with ui_click / ui_type. "
        "An empty result means the app exposes no accessibility data; fall back to "
        "computer_screenshot plus computer_click.",
        {
            "properties": {
                "app": {"type": "string", "description": "Limit to an application by name (default: the focused app)"},
                "window": {"type": "string", "description": "Limit to a window whose title contains this"},
                "scope": {
                    "type": "string",
                    "enum": ["focused", "desktop"],
                    "description": "'focused' (default) reads only the focused window; 'desktop' reads everything",
                },
                "interactive_only": {"type": "boolean", "description": "Only clickable/editable elements"},
                "max_elements": {"type": "integer", "minimum": 10, "maximum": 800},
            }
        },
        risk=Risk.READ,
        group="ui",
    )
    def snapshot(ctx: ToolContext, args: dict) -> ToolResult:
        desktop = ctx.desktop
        kwargs: dict = {
            "max_elements": int(args.get("max_elements", 220)),
            "interactive_only": bool(args.get("interactive_only", False)),
        }
        if args.get("app"):
            kwargs["app_name"] = args["app"]
        if args.get("window"):
            kwargs["window_title"] = args["window"]

        scope = args.get("scope", "focused")
        if scope == "focused" and not args.get("app") and not args.get("window"):
            active = desktop.windows.active_window()
            if active:
                kwargs["pid"] = active.pid
                kwargs["within"] = active.bounds

        snap = desktop.snapshot(**kwargs)
        active = desktop.windows.active_window()
        header = (
            f"Accessible elements in {snap.app or 'desktop'}"
            + (f" (focused window: {active.title!r})" if active else "")
            + f" — {len(snap)} element(s):"
        )
        body = snap.render(limit=kwargs["max_elements"])
        hint = (
            ""
            if len(snap)
            else "\n\nHint: this app publishes no accessibility tree. Chromium-based apps need "
            "--force-renderer-accessibility. Use computer_screenshot + computer_click instead."
        )
        return ToolResult(
            ok=True,
            content=f"{header}\n{body}{hint}",
            data={"count": len(snap), "truncated": snap.truncated, "app": snap.app},
        )

    @registry.tool(
        "ui_find",
        "Search the accessibility tree for elements matching a name and/or role. "
        "Cheaper than a full snapshot when you know what you are looking for.",
        {
            "properties": {
                "query": {"type": "string", "description": "Substring of the element's name"},
                "role": {"type": "string", "description": "Exact role, e.g. 'push button'"},
                "app": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
        },
        risk=Risk.READ,
        group="ui",
    )
    def find(ctx: ToolContext, args: dict) -> ToolResult:
        kwargs = {"app_name": args["app"]} if args.get("app") else {}
        snap = ctx.desktop.snapshot(max_elements=600, **kwargs)
        matches = snap.find(args["query"], role=args.get("role"))
        limit = int(args.get("limit", 25))
        if not matches:
            roles = sorted({e.role for e in snap.elements})
            return ToolResult.text(
                f"No element matching {args['query']!r}"
                + (f" with role {args['role']!r}" if args.get("role") else "")
                + f". The app exposes {len(snap)} element(s) with roles: {', '.join(roles[:15]) or '(none)'}"
            )
        lines = [e.to_line() for e in matches[:limit]]
        return ToolResult(
            ok=True,
            content=f"{len(matches)} match(es):\n" + "\n".join(lines),
            data={"matches": [e.to_dict() for e in matches[:limit]]},
        )

    @registry.tool(
        "ui_click",
        "Click a UI element by ref or by name. Uses the element's own accessible action "
        "when available (deterministic), otherwise clicks its centre point.",
        {
            "properties": {
                **_TARGET,
                "prefer_action": {
                    "type": "boolean",
                    "description": "Use the accessible action rather than a pointer click (default true)",
                },
            }
        },
        risk=Risk.INPUT,
        group="ui",
    )
    def click(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            target = _resolve_target(ctx, args)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        out = ctx.desktop.click_element(
            target, role=args.get("role"), prefer_action=args.get("prefer_action", True)
        )
        element = out.get("element", {})
        return ToolResult.text(
            f"Clicked {element.get('role')} {element.get('name')!r} via {out['method']}", **out
        )

    @registry.tool(
        "ui_type",
        "Put text into a field by ref or name. Sets the value through the accessibility "
        "API when supported, otherwise focuses the field and types.",
        {
            "properties": {
                **_TARGET,
                "text": {"type": "string"},
                "clear": {"type": "boolean", "description": "Replace existing contents (default true)"},
            },
            "required": ["text"],
        },
        risk=Risk.INPUT,
        group="ui",
    )
    def type_into(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            target = _resolve_target(ctx, args)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        out = ctx.desktop.set_element_text(target, args["text"], clear=args.get("clear", True))
        element = out.get("element", {})
        return ToolResult.text(
            f"Set text of {element.get('role')} {element.get('name')!r} "
            f"({out['length']} chars) via {out['method']}",
            **out,
        )

    @registry.tool(
        "ui_focus",
        "Give keyboard focus to an element by ref or name.",
        {"properties": _TARGET},
        risk=Risk.INPUT,
        group="ui",
    )
    def focus(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            target = _resolve_target(ctx, args)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        out = ctx.desktop.focus_element(target)
        return ToolResult.text(f"Focused {out['focused'].get('name')!r} via {out['method']}", **out)

    @registry.tool(
        "ui_read",
        "Read the text content of an element (a text area, terminal, label or document). "
        "Use this to verify what an app is actually showing.",
        {"properties": _TARGET},
        risk=Risk.READ,
        group="ui",
    )
    def read(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            target = _resolve_target(ctx, args)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        if isinstance(target, int):
            element, handle = ctx.desktop.resolve(target)
        else:
            element = ctx.desktop.find_element(target, role=args.get("role"))
            handle = ctx.desktop.last_snapshot.handle(element.ref)
        text = ctx.desktop.a11y.read_text(handle, limit=8000)
        if text is None:
            text = element.value
        if text is None:
            return ToolResult.text(
                f"{element.role} {element.name!r} exposes no readable text "
                f"(name={element.name!r})"
            )
        return ToolResult(
            ok=True,
            content=f"Text of {element.role} {element.name!r}:\n{text}",
            data={"element": element.to_dict(), "text": text, "length": len(text)},
        )

    @registry.tool(
        "ui_wait_for",
        "Block until an element with this name appears, or the timeout expires. "
        "Use after actions that load something (opening a dialog, switching a tab).",
        {
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "timeout": {"type": "number", "minimum": 0.5, "maximum": 120},
            },
            "required": ["name"],
        },
        risk=Risk.READ,
        group="ui",
    )
    def wait_for(ctx: ToolContext, args: dict) -> ToolResult:
        element = ctx.desktop.wait_for_element(
            args["name"], role=args.get("role"), timeout=float(args.get("timeout", 15))
        )
        return ToolResult.text(f"Appeared: {element.to_line()}", element=element.to_dict())
