"""Lending the desktop to another agent.

LAI is an MCP server as well as an agent, so any MCP client can pick up its
hands. Wiring that up by hand means knowing where each client keeps its config
and what shape the entry takes — which is what a command is for.
"""

from __future__ import annotations

import json

import pytest

from lai import connect


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(connect.shutil, "which", lambda name: f"/usr/bin/{name}")
    return tmp_path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# -- the two shapes in the wild ------------------------------------------


def test_opencode_gets_the_shape_opencode_expects(config_home):
    client = connect.get("opencode")
    path = connect.connect(client)
    entry = read(path)["mcp"]["lai"]
    assert entry["type"] == "local"
    assert entry["enabled"] is True
    assert entry["command"][1:] == ["mcp", "--no-mcp"]


def test_claude_gets_the_shape_claude_expects(config_home):
    client = connect.get("claude")
    entry = read(connect.connect(client))["mcpServers"]["lai"]
    assert entry["args"] == ["mcp", "--no-mcp"]
    assert entry["command"].endswith("lai")


def test_the_server_never_loads_other_servers(config_home):
    """LAI is the MCP server here; connecting out to every other configured
    one on the way in would be slow and circular."""
    assert "--no-mcp" in connect.command()


def test_an_absolute_path_is_used(config_home, monkeypatch):
    monkeypatch.setattr(connect.shutil, "which", lambda name: "/home/x/.local/bin/lai")
    assert connect.command()[0] == "/home/x/.local/bin/lai"


# -- not destroying what is already there --------------------------------


def test_existing_settings_survive(config_home):
    client = connect.get("opencode")
    client.config.parent.mkdir(parents=True, exist_ok=True)
    client.config.write_text(
        json.dumps({"theme": "dark", "mcp": {"other": {"type": "local"}}}), encoding="utf-8"
    )
    data = read(connect.connect(client))
    assert data["theme"] == "dark"
    assert set(data["mcp"]) == {"other", "lai"}


def test_a_config_with_comments_is_read(config_home):
    """opencode ships a `.jsonc`, and JSON does not have comments."""
    client = connect.get("opencode")
    client.config.parent.mkdir(parents=True, exist_ok=True)
    client.config.write_text('// a comment\n{"theme": "dark"}\n', encoding="utf-8")
    assert read(connect.connect(client))["theme"] == "dark"


def test_an_unreadable_config_is_replaced_rather_than_losing_the_connection(config_home):
    client = connect.get("opencode")
    client.config.parent.mkdir(parents=True, exist_ok=True)
    client.config.write_text("{ this is not json at all", encoding="utf-8")
    assert "lai" in read(connect.connect(client))["mcp"]


def test_connecting_twice_changes_nothing(config_home):
    client = connect.get("opencode")
    first = connect.connect(client).read_text(encoding="utf-8")
    assert connect.connect(client).read_text(encoding="utf-8") == first


# -- taking it out again --------------------------------------------------


def test_disconnecting_leaves_the_other_servers(config_home):
    client = connect.get("opencode")
    connect.connect(client)
    data = read(client.config)
    data["mcp"]["other"] = {"type": "local"}
    client.config.write_text(json.dumps(data), encoding="utf-8")

    assert connect.disconnect(client) is True
    assert set(read(client.config)["mcp"]) == {"other"}


def test_disconnecting_the_last_one_removes_the_empty_section(config_home):
    client = connect.get("opencode")
    connect.connect(client)
    connect.disconnect(client)
    assert "mcp" not in read(client.config)


def test_disconnecting_what_was_never_connected(config_home):
    assert connect.disconnect(connect.get("opencode")) is False


# -- saying where things stand -------------------------------------------


def test_status_tells_the_three_cases_apart(config_home, monkeypatch):
    client = connect.get("opencode")
    assert connect.status(client) == "absent"
    connect.connect(client)
    assert connect.status(client) == "connected"

    monkeypatch.setattr(connect.shutil, "which", lambda name: None)
    assert connect.status(client) == "missing", "installed is not the same as configured"


def test_an_unknown_client_is_not_invented():
    assert connect.get("emacs") is None
    assert connect.get("") is None


def test_the_config_is_never_left_half_written(config_home):
    """A crash mid-write must not take somebody's editor config with it."""
    client = connect.get("opencode")
    connect.connect(client)
    assert not list(client.config.parent.glob("*.lai-tmp"))
