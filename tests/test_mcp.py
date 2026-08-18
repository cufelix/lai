"""MCP: config loading, risk classification, tool registration, server exposure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lai.config import Config
from lai.mcp.client import (
    MCPPool,
    MCPServerConfig,
    _child_env,
    _decode_result,
    _read_config_file,
    _tool_to_dict,
    classify_risk,
    load_mcp_configs,
    register_mcp_tools,
)
from lai.safety.policy import Risk
from lai.tools.base import ToolRegistry


def write_mcp(path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# -- config loading ------------------------------------------------------


def test_reads_the_standard_shape(tmp_path):
    path = write_mcp(tmp_path / "mcp.json", {
        "mcpServers": {
            "files": {"command": "npx", "args": ["-y", "@mcp/files"], "env": {"ROOT": "/tmp"}},
            "db": {"command": "/usr/bin/dbmcp", "cwd": "/srv"},
        }
    })
    servers = {s.name: s for s in _read_config_file(path)}
    assert servers["files"].command == ["npx", "-y", "@mcp/files"]
    assert servers["files"].env == {"ROOT": "/tmp"}
    assert servers["db"].command == ["/usr/bin/dbmcp"]
    assert servers["db"].cwd == "/srv"
    assert all(s.enabled for s in servers.values())


def test_servers_key_is_also_accepted(tmp_path):
    path = write_mcp(tmp_path / "m.json", {"servers": {"x": {"command": "echo"}}})
    assert [s.name for s in _read_config_file(path)] == ["x"]


def test_disabled_servers_are_marked(tmp_path):
    path = write_mcp(tmp_path / "m.json", {
        "mcpServers": {
            "off": {"command": "echo", "disabled": True},
            "explicit": {"command": "echo", "enabled": False},
            "on": {"command": "echo"},
        }
    })
    servers = {s.name: s.enabled for s in _read_config_file(path)}
    assert servers == {"off": False, "explicit": False, "on": True}


def test_entries_without_a_command_are_skipped(tmp_path):
    path = write_mcp(tmp_path / "m.json", {
        "mcpServers": {"broken": {"args": ["x"]}, "fine": {"command": "echo"}}
    })
    assert [s.name for s in _read_config_file(path)] == ["fine"]


def test_missing_file_is_empty(tmp_path):
    assert _read_config_file(tmp_path / "nothing.json") == []


def test_malformed_json_is_skipped_not_raised(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert _read_config_file(path) == []


def test_wrong_shape_is_skipped(tmp_path):
    assert _read_config_file(write_mcp(tmp_path / "a.json", {"mcpServers": []})) == []
    path = tmp_path / "b.json"
    path.write_text('["a list"]', encoding="utf-8")
    assert _read_config_file(path) == []


def test_first_config_file_wins(tmp_path):
    high = write_mcp(tmp_path / "high" / "mcp.json", {"mcpServers": {"dup": {"command": "winner"}}})
    write_mcp(tmp_path / "low" / ".mcp.json", {"mcpServers": {"dup": {"command": "loser"}}})
    config = Config(home=tmp_path / "high", mcp_config_paths=("{home}/mcp.json", "{cwd}/.mcp.json"))
    servers = load_mcp_configs(config, tmp_path / "low")
    assert len(servers) == 1
    assert servers[0].command == ["winner"]
    assert high.exists()


def test_load_with_no_config_files_at_all(tmp_path):
    config = Config(home=tmp_path, mcp_config_paths=("{home}/absent.json",))
    assert load_mcp_configs(config, tmp_path) == []


def test_server_config_to_dict_hides_env_values():
    server = MCPServerConfig(name="s", command=["x"], env={"SECRET_TOKEN": "hunter2"})
    dumped = server.to_dict()
    assert dumped["env_keys"] == ["SECRET_TOKEN"]
    assert "hunter2" not in json.dumps(dumped)


# -- child environment ---------------------------------------------------


def test_child_env_passes_the_graphical_session(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XAUTHORITY", "/home/u/.Xauthority")
    monkeypatch.setenv("SOME_PRIVATE_TOKEN", "should-not-leak")
    env = _child_env(None)
    assert env["DISPLAY"] == ":0"
    assert env["XAUTHORITY"] == "/home/u/.Xauthority"
    assert "SOME_PRIVATE_TOKEN" not in env


def test_child_env_merges_explicit_config(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    env = _child_env({"API_TOKEN": "given-deliberately"})
    assert env["API_TOKEN"] == "given-deliberately"
    assert env["DISPLAY"] == ":0"


# -- risk classification -------------------------------------------------


@pytest.mark.parametrize(
    "name", ["get_weather", "list_files", "read_document", "search_issues", "fetch_url", "query_db"]
)
def test_read_shaped_names(name):
    assert classify_risk(name) is Risk.READ


@pytest.mark.parametrize(
    "name", ["delete_branch", "remove_file", "drop_table", "kill_process", "execute_sql", "run_command"]
)
def test_destructive_shaped_names(name):
    assert classify_risk(name) is Risk.DESTRUCTIVE


@pytest.mark.parametrize("name", ["create_issue", "update_record", "send_message", "wibble"])
def test_anything_else_defaults_to_write(name):
    assert classify_risk(name) is Risk.WRITE


def test_a_destructive_description_outweighs_a_read_shaped_name():
    assert classify_risk("get_thing", "deletes the thing permanently") is Risk.DESTRUCTIVE


def test_prefixed_names_are_classified_on_the_bare_tool_name():
    assert classify_risk("mcp__server__list_things") is Risk.READ


# -- tool registration ---------------------------------------------------


class FakeClient:
    def __init__(self, tools):
        self.tools = tools
        self.calls = []


def fake_pool(**servers) -> MCPPool:
    pool = MCPPool([])
    for name, tools in servers.items():
        pool.clients[name] = FakeClient(tools)  # type: ignore[assignment]
    return pool


def test_tools_are_registered_with_a_namespaced_name():
    pool = fake_pool(github=[
        {"name": "list_repos", "description": "List repositories", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "delete_repo", "description": "Delete a repository", "inputSchema": {"type": "object", "properties": {}}},
    ])
    registry = ToolRegistry()
    names = register_mcp_tools(registry, pool)
    assert names == ["mcp__github__list_repos", "mcp__github__delete_repo"]
    assert registry.get("mcp__github__list_repos").risk is Risk.READ
    assert registry.get("mcp__github__delete_repo").risk is Risk.DESTRUCTIVE
    assert registry.get("mcp__github__list_repos").group == "mcp:github"


def test_registration_without_a_prefix():
    pool = fake_pool(srv=[{"name": "plain", "description": "d", "inputSchema": {"type": "object"}}])
    registry = ToolRegistry()
    assert register_mcp_tools(registry, pool, prefix=False) == ["plain"]


def test_existing_names_are_not_clobbered():
    pool = fake_pool(srv=[{"name": "thing", "description": "d", "inputSchema": {"type": "object"}}])
    registry = ToolRegistry()
    register_mcp_tools(registry, pool)
    assert register_mcp_tools(registry, pool) == []
    assert len(registry) == 1


def test_nameless_tools_are_skipped():
    pool = fake_pool(srv=[{"name": "", "description": "d"}, {"description": "no name"}])
    registry = ToolRegistry()
    assert register_mcp_tools(registry, pool) == []


def test_a_missing_schema_becomes_an_empty_object_schema():
    pool = fake_pool(srv=[{"name": "bare", "description": "d"}])
    registry = ToolRegistry()
    register_mcp_tools(registry, pool)
    assert registry.get("mcp__srv__bare").to_anthropic()["input_schema"]["type"] == "object"


def test_calling_a_tool_from_a_disconnected_server_fails_cleanly():
    from lai.errors import BackendUnavailable

    with pytest.raises(BackendUnavailable):
        MCPPool([]).call("ghost", "tool", {})


def test_close_all_is_idempotent():
    pool = MCPPool([])
    pool.close_all()
    pool.close_all()


# -- SDK translation -----------------------------------------------------


class FakeTool:
    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.input_schema = schema


def test_tool_to_dict_normalises_the_schema_key():
    out = _tool_to_dict(FakeTool("t", "d", {"type": "object", "properties": {"a": {}}}))
    assert out == {"name": "t", "description": "d", "inputSchema": {"type": "object", "properties": {"a": {}}}}


def test_tool_to_dict_handles_a_missing_description():
    out = _tool_to_dict(FakeTool("t", None, None))
    assert out["description"] == "" and out["inputSchema"]["type"] == "object"


class Block:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeResult:
    def __init__(self, content, is_error=False, structured_content=None):
        self.content = content
        self.is_error = is_error
        self.structured_content = structured_content


def test_decode_text_blocks():
    out = _decode_result(FakeResult([Block(type="text", text="one"), Block(type="text", text="two")]))
    assert out["content"] == "one\ntwo" and out["is_error"] is False and out["images"] == []


def test_decode_image_block():
    import base64

    payload = base64.b64encode(b"PNGDATA").decode()
    out = _decode_result(FakeResult([Block(type="image", data=payload, mime_type="image/png")]))
    assert out["images"] == [b"PNGDATA"]


def test_decode_ignores_undecodable_image_data():
    out = _decode_result(FakeResult([Block(type="image", data="!!!not base64!!!")]))
    assert out["images"] == [] or isinstance(out["images"][0], bytes)


def test_decode_error_flag():
    assert _decode_result(FakeResult([Block(type="text", text="boom")], is_error=True))["is_error"]


def test_decode_falls_back_to_structured_content():
    out = _decode_result(FakeResult([], structured_content={"rows": 3}))
    assert "rows" in out["content"]


def test_decode_embedded_resource_text():
    out = _decode_result(FakeResult([Block(type="resource", resource=Block(text="file body"))]))
    assert out["content"] == "file body"


def test_decode_empty_result():
    out = _decode_result(FakeResult([]))
    assert out["content"] == "" and out["images"] == []


# -- server --------------------------------------------------------------


@pytest.mark.slow
def test_server_exposes_valid_tool_definitions():
    from lai.mcp.server import build_mcp_server

    server = build_mcp_server()
    backend = server.lai_backend
    try:
        definitions = backend.tool_definitions()
        names = {entry["name"] for entry in definitions}
        for expected in ("ui_snapshot", "ui_click", "app_open", "computer_screenshot", "window_list"):
            assert expected in names
        for entry in definitions:
            assert entry["description"]
            assert entry["inputSchema"]["type"] == "object"
            assert "properties" in entry["inputSchema"]
    finally:
        backend.close()


@pytest.mark.slow
def test_server_denies_gated_actions_because_nobody_can_be_asked():
    from dataclasses import replace

    from lai.config import load_config
    from lai.mcp.server import build_mcp_server

    config = load_config()
    config = config.with_overrides(safety=replace(config.safety, mode="ask"))
    server = build_mcp_server(config)
    backend = server.lai_backend
    try:
        result = backend.call("computer_click", {"x": 5, "y": 5})
        assert result.ok is False
        assert "NOT APPROVED" in result.content
    finally:
        backend.close()


def test_server_stderr_is_routed_to_a_log_file(monkeypatch, tmp_path):
    """Third-party servers chatter on stderr; that must not bury agent output."""
    from lai.mcp.client import _open_server_log

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    handle = _open_server_log("some server/name")
    try:
        assert handle is not __import__("sys").stderr
        assert str(tmp_path) in handle.name
        assert "/" not in Path(handle.name).name.replace("mcp-", "")
    finally:
        handle.close()


def test_server_log_falls_back_to_stderr_when_unwritable(monkeypatch):
    import sys as _sys

    from lai.mcp import client as client_module

    monkeypatch.setattr(
        client_module.Path, "mkdir", lambda *a, **kw: (_ for _ in ()).throw(OSError("read-only"))
    )
    assert client_module._open_server_log("x") is _sys.stderr
