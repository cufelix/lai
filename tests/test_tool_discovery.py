"""`tool_find` — getting back a tool the gate did not put in the prompt."""

from __future__ import annotations

import pytest

from lai.agent.toolgate import ToolGate
from lai.safety.policy import Risk
from lai.tools.base import ToolContext, ToolRegistry, ToolResult, ToolSpec
from lai.tools.discovery import register


def spec(name: str, description: str = "", group: str = "core") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters={"properties": {}},
        handler=lambda ctx, args: ToolResult.text("ok"),
        risk=Risk.READ,
        group=group,
    )


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(spec("window_list", "List open windows"))
    register(reg)
    return reg


def find(registry, query: str, *, gate=None) -> ToolResult:
    context = ToolContext(registry=registry, extra={"tool_gate": gate} if gate else {})
    return registry.call("tool_find", {"query": query}, context)


def test_it_says_so_when_nothing_is_connected(registry):
    result = find(registry, "run a sql query")
    assert result.ok
    assert "No external services are connected" in result.content


def test_it_finds_a_withheld_tool_and_unlocks_it(registry):
    registry.register(spec("mcp__db__run_sql", "Run a SQL query", group="mcp:db"))
    gate = ToolGate(registry)
    assert not gate.choose("open the calculator")[0][-1].name.startswith("mcp__")

    result = find(registry, "run a sql query", gate=gate)
    assert result.ok
    assert "mcp__db__run_sql" in result.content
    assert result.data["unlocked"] == ["mcp__db__run_sql"]
    assert "mcp__db__run_sql" in {s.name for s in gate.choose("open the calculator")[0]}


def test_a_query_nobody_matches_names_the_services_instead(registry):
    registry.register(spec("mcp__db__run_sql", "Run a SQL query", group="mcp:db"))
    registry.register(spec("mcp__mail__send", "Send an email", group="mcp:mail"))
    result = find(registry, "photosynthesis")
    assert result.ok
    assert "Nothing matches" in result.content
    assert "db, mail" in result.content


def test_it_works_without_a_gate(registry):
    """The daemon and the MCP server call tools outside a run."""
    registry.register(spec("mcp__db__run_sql", "Run a SQL query", group="mcp:db"))
    result = find(registry, "run a sql query")
    assert result.ok and result.data["unlocked"] == []
    assert "already callable" in result.content


def test_core_tools_are_not_offered_as_discoveries(registry):
    result = find(registry, "list the open windows")
    assert "window_list" not in result.content


def test_naming_a_service_finds_that_service(registry):
    """`github` is in all twenty-six github tool names, which is exactly why a
    plain word-frequency ranking discards it as noise and answers with web
    crawlers instead."""
    for i in range(26):
        registry.register(spec(f"mcp__github__op_{i}", "Do a github thing", group="mcp:github"))
    registry.register(spec("mcp__github__get_pull_request", "Read a pull request", group="mcp:github"))
    registry.register(spec("mcp__crawler__scrape", "Read a web page", group="mcp:crawler"))

    result = find(registry, "read a github pull request")
    assert "mcp__github__get_pull_request" in result.content
    assert "mcp__crawler__scrape" not in result.content


def test_an_explicit_search_accepts_a_weaker_match_than_the_gate_does(registry):
    """The gate runs unprompted and must be strict. A model that went looking
    asked for this, so a hit in the description is worth showing it."""
    registry.register(spec("mcp__db__execute", "Run a SQL query", group="mcp:db"))
    from lai.agent.toolgate import rank_extensions

    candidates = [s for s in registry.specs() if s.name.startswith("mcp__")]
    assert rank_extensions(candidates, "run a sql query", limit=5) == []
    assert rank_extensions(candidates, "run a sql query", limit=5, floor=1)
    assert "mcp__db__execute" in find(registry, "run a sql query").content
