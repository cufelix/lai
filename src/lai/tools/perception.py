"""Perception and environment-awareness tools that fall outside the
``ui_*``/``computer_*`` families: OCR (pixels -> text, for apps with no
accessibility tree), desktop notifications (both reading and sending),
user-idle detection (so the agent doesn't fight the human for the mouse),
virtual-desktop/workspace switching, and session recording.

Non-obvious design decision: the backend engines (:class:`OCREngine`,
:class:`NotificationMonitor`, :class:`IdleMonitor`, :class:`ScreenRecorder`)
are cached on ``ctx.extra`` — a plain dict already carried by every
:class:`ToolContext` — rather than as module-level singletons. A module
global would leak state between unrelated sessions (two agent runs in the
same process would share one notification listener, one in-flight
recording, ...) and would make tests order-dependent. Keying off the
per-context dict gives each session its own engines while still reusing
them across repeated calls within that session (important for
``NotificationMonitor``, whose DBus listener thread should start once, and
for ``ScreenRecorder``, whose ``start``/``stop`` pair must operate on the
same ffmpeg process).
"""

from __future__ import annotations

from ..osl.geometry import Rect
from ..osl.idle import DEFAULT_ACTIVE_THRESHOLD, IdleMonitor
from ..osl.notifications import NotificationMonitor, send_notification
from ..osl.ocr import OCREngine
from ..osl.recorder import ScreenRecorder
from ..safety.policy import Risk
from .base import ToolContext, ToolRegistry, ToolResult

_SCOPE = {
    "scope": {
        "type": "string",
        "enum": ["focused", "desktop", "region"],
        "description": "What to read. 'focused' (default) reads only the focused window's area.",
    },
    "region": {
        "type": "object",
        "description": "Rectangle {x,y,width,height} for scope='region'",
        "properties": {
            "x": {"type": "integer"}, "y": {"type": "integer"},
            "width": {"type": "integer"}, "height": {"type": "integer"},
        },
    },
}

_WINDOW_TARGET = {
    "id": {"type": "integer", "description": "Window id from window_list"},
    "match": {"type": "string", "description": "Substring of the window title or app class"},
}


# -- shared engines, one per session (see module docstring) ----------------


def ocr_engine(ctx: ToolContext) -> OCREngine:
    return ctx.extra.setdefault("ocr_engine", OCREngine())


def _notification_monitor(ctx: ToolContext) -> NotificationMonitor:
    return ctx.extra.setdefault("notification_monitor", NotificationMonitor())


def _idle_monitor(ctx: ToolContext) -> IdleMonitor:
    return ctx.extra.setdefault("idle_monitor", IdleMonitor())


def _recorder(ctx: ToolContext) -> ScreenRecorder:
    return ctx.extra.setdefault("recorder", ScreenRecorder())


def _region_from_args(desktop, args: dict) -> Rect:
    scope = args.get("scope", "focused")
    if scope == "region":
        return Rect.from_dict(args.get("region") or {})
    if scope == "desktop":
        return desktop.screen.virtual_bounds()
    active = desktop.windows.active_window()
    if active and active.bounds.area > 0:
        return active.bounds.intersection(desktop.screen.virtual_bounds())
    return desktop.screen.virtual_bounds()


def _pick_window(ctx: ToolContext, args: dict):
    if args.get("id") is not None:
        return ctx.desktop.windows.get(int(args["id"]))
    if args.get("match"):
        return ctx.desktop.windows.find_one(args["match"])
    active = ctx.desktop.windows.active_window()
    if active is None:
        raise ValueError("no window specified and nothing is focused")
    return active


def register(registry: ToolRegistry) -> None:
    # -- OCR ---------------------------------------------------------------

    @registry.tool(
        "ocr_read",
        "Read text from pixels via OCR (tesseract). Use this ONLY when `ui_snapshot` "
        "returns no elements (or clearly incomplete ones) — the accessibility tree is "
        "always more reliable and gives you exact widget names, not just glyphs. OCR is "
        "the fallback for apps with no a11y tree: Chromium/Electron without "
        "--force-renderer-accessibility, games, canvases, VMs. Returns the recognised "
        "text plus each word's absolute screen coordinates, so text you read can also be "
        "clicked with computer_click.",
        {
            "properties": {
                **_SCOPE,
                "min_confidence": {
                    "type": "number", "minimum": 0, "maximum": 100,
                    "description": "Discard words below this OCR confidence (default 40)",
                },
            }
        },
        risk=Risk.READ,
        group="perception",
    )
    def ocr_read(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            region = _region_from_args(ctx.desktop, args)
        except (KeyError, TypeError, ValueError):
            return ToolResult.failure("region must be an object with x, y, width, height")
        engine = ocr_engine(ctx)
        result = engine.read(region, min_confidence=args.get("min_confidence"))
        if not result.words:
            return ToolResult.text("No text recognised in that region.", **result.to_dict())
        lines = [
            f'"{w.text}" @ {w.bounds.center.x},{w.bounds.center.y} (conf {w.confidence:.0f})'
            for w in result.words[:80]
        ]
        return ToolResult(
            ok=True,
            content=(
                f"OCR text ({len(result.words)} word(s), {result.duration:.2f}s):\n{result.text}\n\n"
                + "\n".join(lines)
            ),
            data=result.to_dict(),
        )

    @registry.tool(
        "ocr_find",
        "Locate a specific piece of text on screen via OCR and get clickable coordinates "
        "for it. Prefer `ui_find` whenever the app exposes an accessibility tree — this is "
        "the pixel-only fallback for apps that don't (see ocr_read).",
        {
            "properties": {
                **_SCOPE,
                "query": {"type": "string", "description": "Substring to search for, case-insensitive"},
                "min_confidence": {"type": "number", "minimum": 0, "maximum": 100},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
        risk=Risk.READ,
        group="perception",
    )
    def ocr_find(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            region = _region_from_args(ctx.desktop, args)
        except (KeyError, TypeError, ValueError):
            return ToolResult.failure("region must be an object with x, y, width, height")
        engine = ocr_engine(ctx)
        limit = int(args.get("limit", 10))
        matches = engine.find_text(
            args["query"], region, min_confidence=args.get("min_confidence"), limit=limit
        )
        if not matches:
            return ToolResult.text(f"No text matching {args['query']!r} found.", matches=[])
        lines = [
            f'"{m.text}" @ {m.bounds.center.x},{m.bounds.center.y} (conf {m.confidence:.0f})'
            for m in matches
        ]
        return ToolResult(
            ok=True,
            content=f"{len(matches)} match(es) for {args['query']!r}:\n" + "\n".join(lines),
            data={"matches": [m.to_dict() for m in matches]},
        )

    # -- notifications -------------------------------------------------------

    @registry.tool(
        "notify_user",
        "Show a desktop notification to the human. Use this to surface a status update, "
        "ask a non-blocking question, or flag something that needs their attention — unlike "
        "a dialog box, it does not steal keyboard/mouse focus, so prefer it over interrupting "
        "whatever the user is currently doing.",
        {
            "properties": {
                "summary": {"type": "string", "description": "Short title line"},
                "body": {"type": "string", "description": "Optional longer message"},
                "urgency": {"type": "string", "enum": ["low", "normal", "critical"]},
                "icon": {"type": "string", "description": "Icon name or path (optional)"},
                "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 60000},
            },
            "required": ["summary"],
        },
        risk=Risk.WRITE,
        group="perception",
    )
    def notify_user(ctx: ToolContext, args: dict) -> ToolResult:
        ok = send_notification(
            args["summary"],
            args.get("body", ""),
            urgency=args.get("urgency", "normal"),
            icon=args.get("icon", ""),
            timeout_ms=int(args.get("timeout_ms", 5000)),
        )
        if not ok:
            return ToolResult.failure(
                "could not show a desktop notification (no notification daemon reachable)"
            )
        return ToolResult.text(
            f"Notified: {args['summary']!r}", summary=args["summary"], body=args.get("body", "")
        )

    @registry.tool(
        "notifications_recent",
        "List desktop notifications LAI has observed recently, from any application, via "
        "the DBus notification bus. Use this to check whether something notified the user "
        "(a build finished, a message arrived) without needing a screenshot.",
        {"properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}}},
        risk=Risk.READ,
        group="perception",
    )
    def notifications_recent(ctx: ToolContext, args: dict) -> ToolResult:
        monitor = _notification_monitor(ctx)
        if not monitor.start():
            return ToolResult.failure(
                "notification monitoring is unavailable on this system "
                "(DBus eavesdropping is blocked, or no session bus is reachable)"
            )
        items = monitor.recent(int(args.get("limit", 20)))
        if not items:
            return ToolResult.text("No notifications observed yet.", notifications=[])
        lines = [f"[{n.urgency}] {n.app_name}: {n.summary!r} — {n.body!r}" for n in items]
        return ToolResult(
            ok=True,
            content=f"{len(items)} recent notification(s):\n" + "\n".join(lines),
            data={"notifications": [n.to_dict() for n in items]},
        )

    # -- idle ------------------------------------------------------------

    @registry.tool(
        "user_idle",
        "Report how long the human has been away from the keyboard/mouse (X11 idle time). "
        "Call this before taking over the pointer or keyboard for a multi-step sequence, so "
        "you don't fight the user for control — if they are active, prefer to wait or ask.",
        {
            "properties": {
                "threshold": {
                    "type": "number", "minimum": 0, "maximum": 3600,
                    "description": f"Seconds of inactivity counted as 'idle' (default {DEFAULT_ACTIVE_THRESHOLD})",
                }
            }
        },
        risk=Risk.READ,
        group="perception",
    )
    def user_idle(ctx: ToolContext, args: dict) -> ToolResult:
        monitor = _idle_monitor(ctx)
        threshold = float(args.get("threshold", DEFAULT_ACTIVE_THRESHOLD))
        state = monitor.state(threshold)
        verb = "active" if state.active else "idle"
        return ToolResult.text(
            f"User has been {verb} for {state.idle_seconds:.1f}s (threshold {threshold}s).",
            **state.to_dict(),
        )

    # -- workspaces --------------------------------------------------------

    @registry.tool(
        "workspace_list",
        "List virtual desktops/workspaces: how many exist, their names (if the window "
        "manager publishes them) and which one is currently active.",
        {"properties": {}},
        risk=Risk.READ,
        group="perception",
    )
    def workspace_list(ctx: ToolContext, args: dict) -> ToolResult:
        wm = ctx.desktop.windows
        count = wm.workspace_count()
        current = wm.current_workspace()
        names = wm.workspace_names()
        lines = [
            f"{'*' if i == current else ' '} {i}: {names[i] if i < len(names) else f'workspace {i}'}"
            for i in range(count)
        ]
        body = "\n".join(lines) if lines else "(workspace count not published by the window manager)"
        return ToolResult(
            ok=True,
            content=f"{count} workspace(s), current={current} (* = active):\n{body}",
            data={"count": count, "current": current, "names": names},
        )

    @registry.tool(
        "workspace_switch",
        "Switch to a different virtual desktop/workspace by index (0-based, from workspace_list).",
        {"properties": {"index": {"type": "integer", "minimum": 0}}, "required": ["index"]},
        risk=Risk.INPUT,
        group="perception",
    )
    def workspace_switch(ctx: ToolContext, args: dict) -> ToolResult:
        current = ctx.desktop.windows.switch_workspace(int(args["index"]))
        return ToolResult.text(f"Switched to workspace {current}.", current=current)

    @registry.tool(
        "window_to_workspace",
        "Move a window to a different virtual desktop/workspace without switching to it "
        "yourself. Use this to file a window away rather than closing it.",
        {
            "properties": {**_WINDOW_TARGET, "index": {"type": "integer", "minimum": 0}},
            "required": ["index"],
        },
        risk=Risk.INPUT,
        group="perception",
    )
    def window_to_workspace(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            window = _pick_window(ctx, args)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        updated = ctx.desktop.windows.move_window_to_workspace(window.id, int(args["index"]))
        return ToolResult.text(
            f"Moved {updated.title!r} to workspace {args['index']}.", **updated.to_dict()
        )

    # -- recording ---------------------------------------------------------

    @registry.tool(
        "record_start",
        "Start recording the screen to a video file (ffmpeg x11grab). Use this to produce "
        "evidence of what you did — e.g. when the user asked for the session to be recorded, "
        "or before a risky multi-step UI flow you may need to explain afterwards. Call "
        "record_stop when done; only one recording can be active at a time.",
        {
            "properties": {
                "path": {"type": "string", "description": "Output file path (.mp4)"},
                "region": {
                    "type": "object",
                    "description": "Rectangle {x,y,width,height}; default the whole screen",
                    "properties": {
                        "x": {"type": "integer"}, "y": {"type": "integer"},
                        "width": {"type": "integer"}, "height": {"type": "integer"},
                    },
                },
                "fps": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["path"],
        },
        risk=Risk.WRITE,
        group="perception",
    )
    def record_start(ctx: ToolContext, args: dict) -> ToolResult:
        region = None
        if args.get("region"):
            try:
                region = Rect.from_dict(args["region"])
            except (KeyError, TypeError, ValueError):
                return ToolResult.failure("region must be an object with x, y, width, height")
        info = _recorder(ctx).start(args["path"], region=region, fps=int(args.get("fps", 10)))
        return ToolResult.text(f"Recording started: {info.path}", **info.to_dict())

    @registry.tool(
        "record_stop",
        "Stop the current screen recording and finalize the video file.",
        {"properties": {}},
        risk=Risk.WRITE,
        group="perception",
    )
    def record_stop(ctx: ToolContext, args: dict) -> ToolResult:
        recorder = _recorder(ctx)
        path = recorder.stop()
        duration = recorder.last_info.duration if recorder.last_info else None
        suffix = f" ({duration:.1f}s)" if duration is not None else ""
        return ToolResult.text(f"Recording stopped: {path}{suffix}", path=str(path), duration=duration)
