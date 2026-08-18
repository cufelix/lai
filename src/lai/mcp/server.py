"""MCP server — expose the desktop to any MCP client.

``lai mcp`` publishes all of LAI's tools over stdio, so Claude Code (or any MCP
client) can open applications, read accessibility trees, click widgets and drive
windows on the real machine:

    claude mcp add lai -- /path/to/lai/.venv/bin/lai mcp

Two constraints shape this module:

1. **stdout is the protocol.** Anything printed there corrupts the stream, so
   every diagnostic goes to stderr.
2. **Nobody can answer a prompt.** An MCP client has no way to relay LAI's
   interactive approval question, so ``ask`` mode denies gated actions with an
   explanatory message rather than silently auto-approving them.
"""

from __future__ import annotations

import sys
from typing import Any

from ..config import Config, load_config
from ..errors import BackendUnavailable
from ..osl.desktop import Desktop
from ..safety.audit import AuditLog
from ..safety.policy import PolicyEngine
from ..skills.registry import SkillRegistry
from ..skills.tools import register as register_skill_tools
from ..tools import build_registry
from ..tools.base import ToolContext, ToolRegistry

SERVER_NAME = "lai"
SERVER_VERSION = "0.1.0"

INSTRUCTIONS = """\
LAI gives you native control of this Linux desktop.

Work in this order:
1. `ui_snapshot` — read the focused app's accessibility tree (roles, names,
   values, exact bounds). This is the desktop's DOM; prefer it to screenshots.
2. Act semantically — `ui_click(name="Save")`, `ui_type(ref=12, text=...)`.
   Fall back to `computer_click(x, y)` only for apps with no a11y tree.
3. Verify — re-snapshot or read the value back before considering a step done.

Use `app_open` to launch programs (it waits for the window to exist), and
`desktop_wait` after anything that loads.
"""


def _log(message: str) -> None:
    """Diagnostics must never touch stdout — that is the protocol channel."""
    print(f"lai-mcp: {message}", file=sys.stderr, flush=True)


class DesktopToolServer:
    """Holds the runtime objects an MCP session needs."""

    def __init__(self, config: Config | None = None, *, groups: set[str] | None = None) -> None:
        self.config = config or load_config()
        self.config.ensure_dirs()
        self.desktop = Desktop(
            max_edge=self.config.desktop.max_edge,
            a11y_timeout_ms=self.config.desktop.a11y_timeout_ms,
            display=self.config.desktop.display or None,
        )
        self.policy = PolicyEngine(
            self.config.safety, focus_provider=self._safe_active_window
        )
        self.audit = AuditLog.for_session(
            self.config.logs_dir, "mcp", redact=self.config.safety.redact_secrets
        )
        self.registry: ToolRegistry = build_registry(policy=self.policy, groups=groups)
        register_skill_tools(self.registry)
        self.skills = SkillRegistry(self.config.resolved_skill_paths())

    def _safe_active_window(self):
        try:
            return self.desktop.windows.active_window()
        except Exception:
            return None

    def context(self) -> ToolContext:
        return ToolContext(
            desktop=self.desktop,
            config=self.config,
            policy=self.policy,
            audit=self.audit,
            skills=self.skills,
            registry=self.registry,
            approver=self._approver,
        )

    def _approver(self, name: str, tool_input: dict, verdict) -> bool:
        """No human is reachable over MCP, so an ASK verdict becomes a refusal."""
        _log(f"denied {name}: {verdict.reason}")
        return False

    def tool_definitions(self) -> list[dict]:
        return self.registry.to_mcp()

    def call(self, name: str, arguments: dict | None):
        return self.registry.call(name, arguments or {}, self.context())

    def close(self) -> None:
        try:
            self.desktop.close()
        except Exception:
            pass


def build_mcp_server(config: Config | None = None, *, groups: set[str] | None = None):
    """Build the MCP server object and its backing desktop runtime."""
    try:
        import mcp.types as types  # noqa: PLC0415
        from mcp.server import Server  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BackendUnavailable(
            "the 'mcp' package is required for server mode", detail="pip install mcp"
        ) from exc

    backend = DesktopToolServer(config, groups=groups)

    async def on_list_tools(_context, _params) -> Any:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=spec["name"],
                    description=spec["description"],
                    inputSchema=spec["inputSchema"],
                )
                for spec in backend.tool_definitions()
            ]
        )

    async def on_call_tool(_context, params) -> Any:
        name = getattr(params, "name", "")
        arguments = getattr(params, "arguments", None) or {}
        try:
            result = backend.call(name, arguments)
        except Exception as exc:
            _log(f"tool {name} raised {type(exc).__name__}: {exc}")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"{type(exc).__name__}: {exc}")],
                isError=True,
            )

        content: list[Any] = [
            types.TextContent(type="text", text=result.content or "(no output)")
        ]
        for image in result.images:
            import base64  # noqa: PLC0415

            content.append(
                types.ImageContent(
                    type="image",
                    data=base64.b64encode(image).decode("ascii"),
                    mimeType="image/png",
                )
            )
        return types.CallToolResult(content=content, isError=not result.ok)

    server = Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    server.lai_backend = backend  # type: ignore[attr-defined]
    return server


async def serve_stdio(config: Config | None = None) -> None:
    """Run the server over stdio until the client disconnects."""
    from mcp.server.stdio import stdio_server  # noqa: PLC0415

    server = build_mcp_server(config)
    backend: DesktopToolServer = server.lai_backend  # type: ignore[attr-defined]
    mode = backend.config.safety.mode
    _log(
        f"ready — {len(backend.registry)} tools, mode={mode}"
        + ("  (ask mode denies gated actions; set LAI_MODE=auto)" if mode == "ask" else "")
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
    finally:
        backend.close()


def run_stdio(config: Config | None = None) -> None:
    """Synchronous entry point used by ``lai mcp``."""
    import anyio  # noqa: PLC0415

    try:
        anyio.run(serve_stdio, config)
    except KeyboardInterrupt:
        _log("interrupted")
