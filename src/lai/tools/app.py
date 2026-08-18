"""Application lifecycle tools — open a program and know it is ready."""

from __future__ import annotations

from ..safety.policy import Risk
from .base import ToolContext, ToolRegistry, ToolResult


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "app_list",
        "List installed applications, optionally filtered. Use this to find out what is "
        "available before trying to open something.",
        {
            "properties": {
                "query": {"type": "string", "description": "Fuzzy filter on name/description"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            }
        },
        risk=Risk.READ,
        group="app",
    )
    def app_list(ctx: ToolContext, args: dict) -> ToolResult:
        limit = int(args.get("limit", 40))
        apps = ctx.desktop.list_apps(args.get("query", ""), limit=limit)
        if not apps:
            return ToolResult.text(f"No application matches {args.get('query', '')!r}.")
        lines = [f"{a.name} (id={a.id}){' — ' + a.comment[:60] if a.comment else ''}" for a in apps]
        return ToolResult(
            ok=True,
            content=f"{len(apps)} application(s):\n" + "\n".join(lines),
            data={"apps": [a.to_dict() for a in apps]},
        )

    @registry.tool(
        "app_open",
        "Launch an application by name and wait until its window actually appears, then "
        "focus it. This is the way to start working with a program — do not shell out to "
        "run GUI binaries. After it returns, call ui_snapshot to see what is on screen.",
        {
            "properties": {
                "name": {"type": "string", "description": "Application name, e.g. 'Text Editor', 'GIMP', 'Files'"},
                "args": {
                    "type": "array",
                    "description": "Extra command-line arguments, e.g. a file path to open",
                    "items": {"type": "string"},
                },
                "wait": {"type": "boolean", "description": "Wait for the window (default true)"},
                "timeout": {"type": "number", "minimum": 1, "maximum": 120},
            },
            "required": ["name"],
        },
        risk=Risk.WRITE,
        group="app",
    )
    def app_open(ctx: ToolContext, args: dict) -> ToolResult:
        result = ctx.desktop.open_app(
            args["name"],
            args=[str(a) for a in args.get("args", [])],
            wait_for_window=bool(args.get("wait", True)),
            timeout=float(args.get("timeout", 25)),
        )
        payload = result.to_dict()
        if result.window:
            ctx.desktop.wait_settle(timeout=2.0)
            summary = (
                f"Opened {result.app.name if result.app else args['name']!r}. "
                f"Window {result.window.title!r} [{result.window.wm_class}] "
                f"at {result.window.bounds.as_tuple()} after {result.waited:.1f}s."
            )
        else:
            summary = (
                f"Started {' '.join(result.command)} (pid {result.pid}) but no window appeared. "
                f"{result.note}"
            )
        return ToolResult(ok=True, content=summary, data=payload)

    @registry.tool(
        "app_close",
        "Terminate an application process. Prefer window_close first so the app can save; "
        "use this when it will not close cleanly.",
        {
            "properties": {
                "pid": {"type": "integer", "description": "Process id (from window_list)"},
                "match": {"type": "string", "description": "Window title/class substring instead of a pid"},
            }
        },
        risk=Risk.DESTRUCTIVE,
        group="app",
    )
    def app_close(ctx: ToolContext, args: dict) -> ToolResult:
        pid = args.get("pid")
        if pid is None:
            if not args.get("match"):
                return ToolResult.failure("provide 'pid' or 'match'")
            window = ctx.desktop.windows.find_one(args["match"])
            pid = window.pid
            if pid is None:
                return ToolResult.failure(f"window {window.title!r} does not report a pid")
        outcome = ctx.desktop.apps.terminate(int(pid))
        return ToolResult.text(f"Process {pid}: {outcome}", pid=int(pid), outcome=outcome)
