"""Finding a tool that was not put in front of the model.

Connected MCP servers can add hundreds of tools. Sending all their schemas
every turn is the single most expensive thing a well-equipped machine does, so
the ones that do not match the task are withheld — named in the prompt, but not
described. This is how the model gets one back: search by what it needs to do,
and the matches become callable for the rest of the run.
"""

from __future__ import annotations

from ..agent.toolgate import is_extension, rank_extensions, server_of
from ..safety.policy import Risk
from ..tools.base import ToolContext, ToolRegistry, ToolResult

RESULT_LIMIT = 8
"""Enough to choose from; few enough that the schemas stay cheap."""


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "tool_find",
        "Search every connected tool, including ones not listed in your prompt, and make "
        "the matches callable. Use this when a task needs a service — a database, a deploy "
        "platform, a browser — whose tools you cannot see. Describe what you want to do, "
        "not a tool name: `tool_find(\"run a SQL query\")`.",
        {
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you are trying to do, or part of a tool name",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
        },
        risk=Risk.READ,
        group="skill",
    )
    def tool_find(ctx: ToolContext, args: dict) -> ToolResult:
        if ctx.registry is None:
            return ToolResult.failure("No tool registry is available.")
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", RESULT_LIMIT))
        candidates = [spec for spec in ctx.registry.specs() if is_extension(spec)]
        if not candidates:
            return ToolResult.text(
                "No external services are connected — every tool you have is already listed."
            )

        # The same ranking the gate uses, so naming a service here works the
        # way it does there: "read a github pull request" was answering with
        # web crawlers, because `github` is in all twenty-six github tool names
        # and a word that common is normally noise.
        found = rank_extensions(candidates, query, limit=limit, floor=1)
        if not found:
            servers = sorted({server_of(spec) for spec in candidates})
            return ToolResult.text(
                f"Nothing matches {query!r}. Connected services: {', '.join(servers)}. "
                "Try naming the service, or the operation you want to perform."
            )

        gate = (ctx.extra or {}).get("tool_gate")
        unlocked = gate.unlock(spec.name for spec in found) if gate is not None else []
        lines = [f"- {spec.name}: {(spec.description or '').splitlines()[0][:160]}" for spec in found]
        note = (
            "\nThese are callable from your next turn."
            if unlocked else "\nThese are already callable."
        )
        return ToolResult(
            ok=True,
            content=f"{len(found)} tool(s) matching {query!r}:\n" + "\n".join(lines) + note,
            data={"tools": [spec.name for spec in found], "unlocked": unlocked},
        )
