"""Lending the desktop to another agent.

LAI is an MCP server as well as an agent, which means any MCP client can pick
up its hands: `ui_snapshot`, `ui_click`, `app_open`, the screenshot tools, the
lot — with the safety gate still in force. opencode and Claude Code both speak
MCP and both already have a good loop, a good TUI and their own model
credentials, so there is no reason to make anybody choose.

Wiring that up by hand means knowing where each client keeps its config and
what shape the entry takes. That is exactly the sort of thing a command should
do for you.

What does *not* travel over MCP is the part of LAI's loop that is about
desktops rather than tools: standing aside while you use the mouse, refusing a
click it has already made five times, working on a screen of its own, reading
the pixels when the model turns out to have no vision. A borrowed pair of hands
is still worth having; it is just not the same thing as the whole agent.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

SERVER_NAME = "lai"


@dataclass(frozen=True, slots=True)
class Client:
    """An MCP client we know how to configure."""

    name: str
    label: str
    config: Path
    style: str
    """opencode | claude — the two entry shapes in the wild."""

    @property
    def installed(self) -> bool:
        return shutil.which(self.name.split(":")[0]) is not None


def _home(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, str(Path.home() / default))).expanduser()


def clients() -> list[Client]:
    config_home = _home("XDG_CONFIG_HOME", ".config")
    return [
        Client("opencode", "opencode", config_home / "opencode" / "opencode.jsonc", "opencode"),
        Client("claude", "Claude Code", Path.home() / ".claude.json", "claude"),
    ]


def get(name: str) -> Client | None:
    wanted = (name or "").strip().lower()
    return next((client for client in clients() if client.name == wanted), None)


def command() -> list[str]:
    """How another program should start LAI's MCP server.

    An absolute path: an MCP client's environment is not a login shell, and
    `~/.local/bin` is frequently missing from it.
    """
    found = shutil.which("lai") or "lai"
    # `--no-mcp` stops a loop: LAI is the MCP server here, and connecting to
    # every other configured server on the way in would be slow and circular.
    return [found, "mcp", "--no-mcp"]


def entry(style: str) -> dict:
    """The config fragment this client expects."""
    if style == "opencode":
        return {"type": "local", "command": command(), "enabled": True}
    return {"command": command()[0], "args": command()[1:]}


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # `.jsonc` allows comments, and opencode ships one by default.
        stripped = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("//")
        )
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {}


def status(client: Client) -> str:
    """connected | absent | missing — what this client's config says today."""
    if not client.installed:
        return "missing"
    data = _read(client.config)
    servers = data.get("mcp") if client.style == "opencode" else data.get("mcpServers")
    return "connected" if isinstance(servers, dict) and SERVER_NAME in servers else "absent"


def connect(client: Client) -> Path:
    """Add LAI to this client's MCP servers, keeping everything else."""
    data = _read(client.config)
    key = "mcp" if client.style == "opencode" else "mcpServers"
    servers = data.get(key)
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_NAME] = entry(client.style)
    data[key] = servers
    if client.style == "opencode":
        data.setdefault("$schema", "https://opencode.ai/config.json")

    client.config.parent.mkdir(parents=True, exist_ok=True)
    _write(client.config, data)
    return client.config


def disconnect(client: Client) -> bool:
    """Take it out again. True if there was something to remove."""
    data = _read(client.config)
    key = "mcp" if client.style == "opencode" else "mcpServers"
    servers = data.get(key)
    if not isinstance(servers, dict) or SERVER_NAME not in servers:
        return False
    servers.pop(SERVER_NAME)
    if not servers:
        data.pop(key, None)
    _write(client.config, data)
    return True


def _write(path: Path, data: dict) -> None:
    """Write it back, and never leave a half-written config behind."""
    temporary = path.with_suffix(path.suffix + ".lai-tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
