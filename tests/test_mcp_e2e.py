"""End-to-end MCP: LAI's client driving LAI's own server over real stdio.

This is the integration that matters — it is exactly the path Claude Code takes
when you run ``claude mcp add lai -- lai mcp``. It spawns a real subprocess,
performs the protocol handshake, lists tools, calls them, and checks that images
and the safety gate survive the round trip.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from lai.mcp.client import MCPClient, MCPServerConfig

pytestmark = [pytest.mark.slow, pytest.mark.x11]

REPO_ROOT = Path(__file__).resolve().parent.parent


def lai_command(*args: str) -> list[str]:
    """Prefer the installed console script; fall back to `python -m`."""
    console = REPO_ROOT / ".venv" / "bin" / "lai"
    if console.is_file():
        return [str(console), *args]
    found = shutil.which("lai")
    if found:
        return [found, *args]
    return [sys.executable, "-m", "lai.cli", *args]


@pytest.fixture(scope="module")
def client():
    config = MCPServerConfig(
        name="lai-self",
        command=lai_command("mcp", "--mode", "readonly"),
        cwd=str(REPO_ROOT),
    )
    connection = MCPClient(config)
    try:
        connection.connect()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not start the LAI MCP server: {exc}")
    try:
        yield connection
    finally:
        connection.close()


def test_handshake_exposes_the_desktop_toolset(client):
    assert client.connected
    names = {tool["name"] for tool in client.tools}
    for expected in (
        "ui_snapshot", "ui_click", "ui_type", "app_open", "app_list",
        "computer_screenshot", "computer_click", "window_list", "desktop_observe",
    ):
        assert expected in names, f"{expected} missing from the MCP toolset"
    assert len(names) > 25


def test_every_advertised_tool_has_a_usable_schema(client):
    for tool in client.tools:
        assert tool["name"]
        assert tool["description"], f"{tool['name']} has no description"
        schema = tool["inputSchema"]
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_a_read_tool_returns_real_data(client):
    result = client.call("window_list", {})
    assert result["is_error"] is False
    assert "window(s)" in result["content"]


def test_a_screenshot_survives_the_protocol_as_an_image(client):
    result = client.call("computer_screenshot", {"scope": "desktop", "max_edge": 640})
    assert result["is_error"] is False
    assert len(result["images"]) == 1
    png = result["images"][0]
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # Assert the decoded dimensions, not the byte count: a desktop showing a
    # mostly-uniform image compresses to well under a kilobyte, which made a
    # size threshold fail for reasons that had nothing to do with the protocol.
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    assert max(width, height) == 640, f"max_edge should cap the long side, got {width}x{height}"
    assert min(width, height) > 0
    assert "maps to screen" in result["content"]


def test_the_safety_gate_still_applies_over_mcp(client):
    """The server was started in readonly mode; input must be refused."""
    result = client.call("computer_click", {"x": 10, "y": 10})
    assert result["is_error"] is True
    assert "BLOCKED" in result["content"]
    assert "readonly" in result["content"]


def test_an_invalid_argument_is_reported_not_crashed(client):
    result = client.call("computer_screenshot", {"scope": "not-a-scope"})
    assert result["is_error"] is True
    assert client.connected, "the server must survive a bad request"


def test_an_unknown_tool_is_reported(client):
    result = client.call("no_such_tool", {})
    assert result["is_error"] is True


def test_the_session_survives_repeated_calls(client):
    for _ in range(5):
        assert client.call("window_list", {})["is_error"] is False
    assert client.connected


def test_client_is_reusable_as_a_context_manager():
    config = MCPServerConfig(
        name="lai-ctx", command=lai_command("mcp", "--mode", "readonly"), cwd=str(REPO_ROOT)
    )
    with MCPClient(config) as connection:
        assert connection.connected
        assert connection.call("app_list", {"limit": 3})["is_error"] is False
    assert not connection.connected


def test_calling_before_connecting_fails_clearly():
    from lai.errors import BackendUnavailable

    connection = MCPClient(MCPServerConfig(name="x", command=lai_command("mcp")))
    with pytest.raises(BackendUnavailable, match="not connected"):
        connection.call("window_list", {})


def test_a_missing_binary_is_reported_without_hanging():
    from lai.errors import BackendUnavailable

    connection = MCPClient(
        MCPServerConfig(name="ghost", command=["definitely-not-a-real-binary-xyz"])
    )
    with pytest.raises(BackendUnavailable, match="command not found"):
        connection.connect()


def test_pool_records_a_failing_server_instead_of_raising():
    from lai.mcp.client import MCPPool

    pool = MCPPool([
        MCPServerConfig(name="broken", command=["definitely-not-a-real-binary-xyz"]),
        MCPServerConfig(name="good", command=lai_command("mcp", "--mode", "readonly")),
    ])
    try:
        discovered = pool.connect_all()
        assert "broken" in pool.errors
        assert discovered.get("good")
    finally:
        pool.close_all()


def test_pool_registers_prefixed_tools_into_a_registry():
    from lai.mcp.client import MCPPool, register_mcp_tools
    from lai.tools.base import ToolRegistry

    pool = MCPPool([
        MCPServerConfig(
            name="desk", command=lai_command("mcp", "--mode", "readonly"), cwd=str(REPO_ROOT)
        )
    ])
    try:
        pool.connect_all()
        if "desk" not in pool.clients:
            pytest.skip(f"server did not start: {pool.errors}")
        registry = ToolRegistry()
        names = register_mcp_tools(registry, pool)
        assert "mcp__desk__window_list" in names
        assert registry.get("mcp__desk__window_list").group == "mcp:desk"

        # And the proxied tool actually works through the registry.
        from lai.tools.base import ToolContext

        result = registry.call("mcp__desk__window_list", {}, ToolContext())
        assert result.ok and "window(s)" in result.content
    finally:
        pool.close_all()
