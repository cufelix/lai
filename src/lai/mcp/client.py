"""MCP client — pull tools from any MCP server into LAI.

This is how the desktop agent gains capabilities it was never written to have:
point it at an MCP server and its tools appear in the registry as
``mcp__<server>__<tool>``, subject to the same permission gate as everything
else.

Config is read from LAI's own ``mcp.json``, the project's ``.mcp.json`` and
Claude Code's MCP config — the standard ``{"mcpServers": {...}}`` shape — so
servers you already use work with no extra setup.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import BackendUnavailable, LaiError
from ..safety.policy import Risk
from ..tools.base import ToolContext, ToolRegistry, ToolResult, ToolSpec
from .bridge import run_sync, spawn

CONNECT_TIMEOUT = 25.0
CALL_TIMEOUT = 120.0
MAX_RESULT_CHARS = 20_000

# Heuristics for how dangerous an unknown third-party tool is. Wrong in the
# safe direction: anything not obviously read-only is treated as a write.
_READ_HINTS = ("get", "list", "read", "search", "fetch", "query", "find", "show", "describe", "inspect", "view")
_DESTRUCTIVE_HINTS = (
    "delete", "remove", "drop", "kill", "destroy", "purge", "execute", "run_",
    "exec", "shell", "command", "write_file", "deploy", "publish",
)


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    name: str
    command: list[str]
    env: dict = field(default_factory=dict)
    cwd: str = ""
    enabled: bool = True
    transport: str = "stdio"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "command": self.command,
            "cwd": self.cwd,
            "enabled": self.enabled,
            "transport": self.transport,
            "env_keys": sorted(self.env),
        }


def load_mcp_configs(config, cwd: Path | None = None) -> list[MCPServerConfig]:
    """Read every configured MCP server. Never raises for a bad file."""
    servers: dict[str, MCPServerConfig] = {}
    for path in config.resolved_mcp_paths(cwd):
        for server in _read_config_file(Path(path)):
            # First file wins: earlier paths are higher priority.
            servers.setdefault(server.name, server)
    return list(servers.values())


def _read_config_file(path: Path) -> list[MCPServerConfig]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace") or "{}")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"lai: skipping unreadable MCP config {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(data, dict):
        return []

    raw_servers = data.get("mcpServers") or data.get("servers") or {}
    if not isinstance(raw_servers, dict):
        return []

    out: list[MCPServerConfig] = []
    for name, entry in raw_servers.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("transport") not in (None, "stdio") or entry.get("type") not in (None, "stdio"):
            continue  # only stdio is supported today
        command = entry.get("command")
        if not command:
            continue
        argv = [str(command), *[str(a) for a in (entry.get("args") or [])]]
        out.append(
            MCPServerConfig(
                name=str(name),
                command=argv,
                env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                cwd=str(entry.get("cwd") or ""),
                enabled=entry.get("enabled", entry.get("disabled") is not True),
            )
        )
    return out


class MCPClient:
    """One stdio MCP server, owned by a single background task."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.tools: list[dict] = []
        self.error: str = ""
        self._queue: asyncio.Queue | None = None
        self._ready: asyncio.Event | None = None
        self._stopped: asyncio.Event | None = None
        self._task = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> list[dict]:
        """Start the server and return its tool list."""
        if self._connected:
            return self.tools
        if not self.config.command:
            raise LaiError(f"MCP server {self.config.name!r} has no command")
        binary = self.config.command[0]
        if not shutil.which(binary) and not Path(binary).exists():
            raise BackendUnavailable(
                f"MCP server {self.config.name!r}: command not found: {binary}"
            )

        # These asyncio primitives must be created on the loop that uses them.
        def _make() -> None:
            self._queue = asyncio.Queue()
            self._ready = asyncio.Event()
            self._stopped = asyncio.Event()

        run_sync(_coro(_make), timeout=5.0)
        self._task = spawn(self._serve())

        ready = run_sync(self._await_ready(), timeout=CONNECT_TIMEOUT)
        if not ready:
            raise BackendUnavailable(
                f"MCP server {self.config.name!r} did not become ready",
                detail=self.error or f"timed out after {CONNECT_TIMEOUT}s",
            )
        self._connected = True
        return self.tools

    async def _await_ready(self) -> bool:
        assert self._ready is not None
        done, _ = await asyncio.wait(
            [asyncio.create_task(self._ready.wait()), asyncio.create_task(self._stopped.wait())],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=CONNECT_TIMEOUT,
        )
        for task in done:
            task.result()
        return self._ready.is_set()

    async def _serve(self) -> None:
        """Own the server's whole lifecycle in one task (see bridge docstring)."""
        try:
            from mcp.client.session import ClientSession  # noqa: PLC0415
            from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: PLC0415
        except ImportError as exc:
            self.error = f"mcp package unavailable: {exc}"
            if self._stopped:
                self._stopped.set()
            return

        params = StdioServerParameters(
            command=self.config.command[0],
            args=list(self.config.command[1:]),
            env=_child_env(self.config.env),
            cwd=self.config.cwd or None,
        )
        errlog = _open_server_log(self.config.name)
        try:
            async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), timeout=CONNECT_TIMEOUT)
                    listed = await asyncio.wait_for(session.list_tools(), timeout=CONNECT_TIMEOUT)
                    self.tools = [_tool_to_dict(tool) for tool in listed.tools]
                    assert self._ready is not None and self._queue is not None
                    self._ready.set()

                    while True:
                        command = await self._queue.get()
                        if command is None:
                            break
                        name, arguments, future = command
                        try:
                            raw = await asyncio.wait_for(
                                session.call_tool(name, arguments), timeout=CALL_TIMEOUT
                            )
                            future.get_loop().call_soon(future.set_result, _decode_result(raw))
                        except Exception as exc:
                            future.get_loop().call_soon(future.set_exception, exc)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._connected = False
            if self._stopped is not None:
                self._stopped.set()
            if self._ready is not None:
                self._ready.set()
            if errlog not in (sys.stderr, None):
                try:
                    errlog.close()
                except Exception:
                    pass

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """Invoke a tool on this server, synchronously."""
        if not self._connected:
            raise BackendUnavailable(
                f"MCP server {self.config.name!r} is not connected", detail=self.error
            )

        async def submit() -> dict:
            assert self._queue is not None
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            await self._queue.put((name, dict(arguments or {}), future))
            return await future

        return run_sync(submit(), timeout=CALL_TIMEOUT + 10)

    def close(self) -> None:
        if self._queue is None:
            return
        try:
            run_sync(_put_none(self._queue), timeout=5.0)
        except Exception:
            pass
        self._connected = False

    def __enter__(self) -> MCPClient:
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class MCPPool:
    """Several MCP servers; a broken one is recorded, not raised."""

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self.servers = list(servers)
        self.clients: dict[str, MCPClient] = {}
        self.errors: dict[str, str] = {}

    def connect_all(self) -> dict[str, list[dict]]:
        discovered: dict[str, list[dict]] = {}
        for server in self.servers:
            if not server.enabled:
                continue
            client = MCPClient(server)
            try:
                discovered[server.name] = client.connect()
                self.clients[server.name] = client
            except Exception as exc:
                self.errors[server.name] = str(exc)
                try:
                    client.close()
                except Exception:
                    pass
        return discovered

    def call(self, server: str, tool: str, arguments: dict) -> dict:
        client = self.clients.get(server)
        if client is None:
            raise BackendUnavailable(f"MCP server {server!r} is not connected")
        return client.call(tool, arguments)

    def close_all(self) -> None:
        for client in list(self.clients.values()):
            try:
                client.close()
            except Exception:
                continue
        self.clients.clear()


def classify_risk(name: str, description: str = "") -> Risk:
    """Guess how dangerous a third-party tool is, erring towards caution."""
    haystack = f"{name} {description}".lower()
    if any(hint in haystack for hint in _DESTRUCTIVE_HINTS):
        return Risk.DESTRUCTIVE
    bare = name.lower().split("__")[-1]
    if any(bare.startswith(hint) or f"_{hint}" in bare for hint in _READ_HINTS):
        return Risk.READ
    return Risk.WRITE


def register_mcp_tools(registry: ToolRegistry, pool: MCPPool, *, prefix: bool = True) -> list[str]:
    """Register every tool from every connected server."""
    registered: list[str] = []
    for server_name, client in pool.clients.items():
        for tool in client.tools:
            raw_name = tool.get("name", "")
            if not raw_name:
                continue
            name = f"mcp__{_slug(server_name)}__{raw_name}" if prefix else raw_name
            if name in registry:
                continue
            description = tool.get("description") or f"{raw_name} (via MCP server {server_name})"
            schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
            registry.register(
                ToolSpec(
                    name=name,
                    description=description[:1200],
                    parameters=schema,
                    handler=_make_handler(pool, server_name, raw_name),
                    risk=classify_risk(raw_name, description),
                    group=f"mcp:{server_name}",
                )
            )
            registered.append(name)
    return registered


def _make_handler(pool: MCPPool, server: str, tool: str):
    def handler(ctx: ToolContext, args: dict) -> ToolResult:
        payload = pool.call(server, tool, args)
        content = payload.get("content", "")
        if len(content) > MAX_RESULT_CHARS:
            content = content[:MAX_RESULT_CHARS] + "\n… [truncated]"
        return ToolResult(
            ok=not payload.get("is_error", False),
            content=content or "(no output)",
            data={"server": server, "tool": tool},
            images=list(payload.get("images", [])),
        )

    return handler


# -- SDK translation -----------------------------------------------------


def _tool_to_dict(tool: Any) -> dict:
    """Normalise an SDK Tool object to the plain dict the registry wants."""
    schema = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
        or {"type": "object", "properties": {}}
    )
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(by_alias=True, exclude_none=True)
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", "") or "",
        "inputSchema": schema if isinstance(schema, dict) else {"type": "object", "properties": {}},
    }


def _decode_result(raw: Any) -> dict:
    """Flatten an SDK CallToolResult into text plus PNG bytes."""
    import base64  # noqa: PLC0415

    texts: list[str] = []
    images: list[bytes] = []
    for block in getattr(raw, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(str(getattr(block, "text", "")))
        elif kind == "image":
            data = getattr(block, "data", "")
            try:
                images.append(base64.b64decode(data) if isinstance(data, str) else bytes(data))
            except (ValueError, TypeError):
                continue
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            text = getattr(resource, "text", None)
            if text:
                texts.append(str(text))

    structured = getattr(raw, "structured_content", None)
    if structured and not texts:
        texts.append(json.dumps(structured, ensure_ascii=False, default=str))

    return {
        "content": "\n".join(t for t in texts if t),
        "is_error": bool(getattr(raw, "is_error", False)),
        "images": images,
    }


# The SDK inherits only a safe subset of the environment, which drops the
# graphical session. A desktop-driving MCP server is useless without it.
_PASSTHROUGH_ENV = (
    "DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR",
    "XDG_SESSION_TYPE", "XDG_CURRENT_DESKTOP", "DBUS_SESSION_BUS_ADDRESS",
    "HOME", "USER", "LANG", "PATH", "SHELL", "TERM",
)


def _child_env(extra: dict | None) -> dict | None:
    """Environment for a spawned MCP server: session vars plus explicit config."""
    env = {name: os.environ[name] for name in _PASSTHROUGH_ENV if name in os.environ}
    env.update(extra or {})
    return env or None


def _open_server_log(server_name: str):
    """Per-server stderr sink.

    Third-party MCP servers chatter on stderr at startup (missing optional env
    vars, banners, warnings). Letting that reach the terminal buries the agent's
    own output, so it goes to a log file the user can read when a server
    misbehaves. Falls back to stderr if the log cannot be opened.
    """
    try:
        from ..config import DEFAULT_HOME  # noqa: PLC0415

        directory = Path(os.environ.get("LAI_HOME", str(DEFAULT_HOME))).expanduser() / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        return (directory / f"mcp-{_slug(server_name)}.log").open("a", encoding="utf-8")
    except OSError:
        return sys.stderr


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)[:40]


async def _coro(fn) -> Any:
    return fn()


async def _put_none(queue: asyncio.Queue) -> None:
    await queue.put(None)
