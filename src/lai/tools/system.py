"""Shell, filesystem, clipboard and synchronisation tools."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..safety.policy import Risk
from .base import ToolContext, ToolRegistry, ToolResult

MAX_READ_BYTES = 400_000
MAX_LIST_ENTRIES = 500


def register(registry: ToolRegistry) -> None:
    # -- shell -----------------------------------------------------------

    @registry.tool(
        "shell_exec",
        "Run a shell command and capture its output. Use for non-GUI work: file "
        "manipulation, git, package queries, scripts. Do NOT use it to launch GUI "
        "applications — use app_open, which waits for the window.",
        {
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string", "description": "Working directory (default: the session cwd)"},
                "timeout": {"type": "number", "minimum": 1, "maximum": 600},
            },
            "required": ["command"],
        },
        risk=Risk.DESTRUCTIVE,
        group="system",
    )
    def shell_exec(ctx: ToolContext, args: dict) -> ToolResult:
        cwd = Path(args.get("cwd") or ctx.cwd).expanduser()
        if not cwd.is_dir():
            return ToolResult.failure(f"working directory does not exist: {cwd}")
        timeout = float(args.get("timeout", 60))
        try:
            proc = subprocess.run(  # noqa: S602 - running shell commands is this tool's purpose
                args["command"],
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"command timed out after {timeout}s: {args['command']}")

        stdout, stderr = proc.stdout or "", proc.stderr or ""
        body = stdout
        if stderr:
            body += ("\n--- stderr ---\n" if stdout else "") + stderr
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        return ToolResult(
            ok=proc.returncode == 0,
            content=f"[{status}] {args['command']}\n{body.strip() or '(no output)'}",
            data={"exit_code": proc.returncode, "stdout": stdout[:20000], "stderr": stderr[:8000], "cwd": str(cwd)},
        )

    # -- files -----------------------------------------------------------

    @registry.tool(
        "file_read",
        "Read a text file from disk.",
        {
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 100, "maximum": MAX_READ_BYTES},
            },
            "required": ["path"],
        },
        risk=Risk.READ,
        group="system",
    )
    def file_read(ctx: ToolContext, args: dict) -> ToolResult:
        path = _resolve(ctx, args["path"])
        if not path.is_file():
            return ToolResult.failure(f"not a file: {path}")
        limit = int(args.get("max_bytes", 100_000))
        try:
            data = path.read_bytes()[: limit + 1]
        except OSError as exc:
            return ToolResult.failure(f"cannot read {path}: {exc}")
        text = data.decode("utf-8", errors="replace")
        truncated = len(data) > limit
        if truncated:
            text = text[:limit] + f"\n… [truncated at {limit} bytes]"
        return ToolResult(
            ok=True,
            content=text,
            data={"path": str(path), "bytes": path.stat().st_size, "truncated": truncated},
        )

    @registry.tool(
        "file_write",
        "Write text to a file, creating parent directories as needed.",
        {
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean"},
            },
            "required": ["path", "content"],
        },
        risk=Risk.WRITE,
        group="system",
    )
    def file_write(ctx: ToolContext, args: dict) -> ToolResult:
        path = _resolve(ctx, args["path"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if args.get("append") else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(args["content"])
        except OSError as exc:
            return ToolResult.failure(f"cannot write {path}: {exc}")
        verb = "Appended to" if args.get("append") else "Wrote"
        return ToolResult.text(
            f"{verb} {path} ({len(args['content'])} chars)", path=str(path), bytes=path.stat().st_size
        )

    @registry.tool(
        "file_list",
        "List a directory, or glob for files.",
        {
            "properties": {
                "path": {"type": "string", "description": "Directory (default: session cwd)"},
                "pattern": {"type": "string", "description": "Glob such as '**/*.py'"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIST_ENTRIES},
            }
        },
        risk=Risk.READ,
        group="system",
    )
    def file_list(ctx: ToolContext, args: dict) -> ToolResult:
        base = _resolve(ctx, args.get("path") or ".")
        if not base.is_dir():
            return ToolResult.failure(f"not a directory: {base}")
        limit = int(args.get("limit", 200))
        pattern = args.get("pattern")
        try:
            entries = sorted(base.glob(pattern))[:limit] if pattern else sorted(base.iterdir())[:limit]
        except (OSError, ValueError) as exc:
            return ToolResult.failure(f"cannot list {base}: {exc}")
        lines = [
            f"{'d' if p.is_dir() else '-'} {p.name}"
            + ("" if p.is_dir() else f"  {_size(p)}")
            for p in entries
        ]
        return ToolResult(
            ok=True,
            content=f"{base} ({len(entries)} entries):\n" + "\n".join(lines),
            data={"path": str(base), "entries": [str(p) for p in entries]},
        )

    # -- clipboard -------------------------------------------------------

    @registry.tool(
        "clipboard_read",
        "Read the system clipboard. Useful to pull data out of an app that has no "
        "accessibility tree: select the text in the app, press ctrl+c, then read it here.",
        {"properties": {"selection": {"type": "string", "enum": ["clipboard", "primary"]}}},
        risk=Risk.READ,
        group="system",
    )
    def clipboard_read(ctx: ToolContext, args: dict) -> ToolResult:
        text = ctx.desktop.clipboard.read_text(args.get("selection", "clipboard"))
        if not text:
            has_image = ctx.desktop.clipboard.has_image(args.get("selection", "clipboard"))
            return ToolResult.text(
                "Clipboard holds an image (no text)." if has_image else "Clipboard is empty."
            )
        return ToolResult(ok=True, content=text, data={"length": len(text)})

    @registry.tool(
        "clipboard_write",
        "Put text on the system clipboard, so it can be pasted with ctrl+v. Often the "
        "most reliable way to enter a long or unicode-heavy string into an app.",
        {
            "properties": {
                "text": {"type": "string"},
                "selection": {"type": "string", "enum": ["clipboard", "primary"]},
            },
            "required": ["text"],
        },
        risk=Risk.WRITE,
        group="system",
    )
    def clipboard_write(ctx: ToolContext, args: dict) -> ToolResult:
        length = ctx.desktop.clipboard.write_text(args["text"], args.get("selection", "clipboard"))
        return ToolResult.text(f"Copied {length} characters to the clipboard.", length=length)

    # -- synchronisation -------------------------------------------------

    @registry.tool(
        "desktop_observe",
        "Get the full picture at once: focused window, window list, accessibility "
        "elements and a screenshot. The cheapest way to re-orient after something "
        "unexpected happens.",
        {
            "properties": {
                "screenshot": {"type": "boolean", "description": "Include an image (default true)"},
                "scope": {"type": "string", "enum": ["focused", "desktop"]},
                "annotate": {"type": "boolean", "description": "Number the interactive elements on the image"},
            }
        },
        risk=Risk.READ,
        group="core",
        returns_image=True,
    )
    def desktop_observe(ctx: ToolContext, args: dict) -> ToolResult:
        want_image = bool(args.get("screenshot", True))
        observation = ctx.desktop.observe(
            screenshot=want_image,
            scope=args.get("scope", "focused"),
            annotate_elements=bool(args.get("annotate", False)),
        )
        images = [observation.screenshot.png] if observation.screenshot else []
        return ToolResult(
            ok=True,
            content=observation.summary(),
            data={"windows": len(observation.windows), "elements": len(observation.elements or [])},
            images=images,
        )

    @registry.tool(
        "desktop_wait",
        "Wait for the screen to stop changing (an app finished loading or animating), "
        "or just sleep for a fixed time.",
        {
            "properties": {
                "seconds": {
                    "type": "number", "minimum": 0, "maximum": 60,
                    "description": "Fixed sleep instead of settle detection",
                },
                "timeout": {"type": "number", "minimum": 0.5, "maximum": 60},
            }
        },
        risk=Risk.READ,
        group="core",
    )
    def desktop_wait(ctx: ToolContext, args: dict) -> ToolResult:
        if args.get("seconds") is not None:
            import time  # noqa: PLC0415

            duration = max(0.0, min(float(args["seconds"]), 60.0))
            time.sleep(duration)
            return ToolResult.text(f"Waited {duration:.1f}s.", waited=duration)
        out = ctx.desktop.wait_settle(timeout=float(args.get("timeout", 3.0)))
        verb = "Screen settled" if out.get("settled") else "Screen still changing after timeout"
        return ToolResult.text(f"{verb}.", **out)


def _resolve(ctx: ToolContext, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (Path(ctx.cwd) / path).resolve()


def _size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}" if unit != "B" else f"{size}B"
        size /= 1024
    return f"{size:.0f}G"


__all__ = ["MAX_LIST_ENTRIES", "MAX_READ_BYTES", "register"]
