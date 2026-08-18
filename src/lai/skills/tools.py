"""Skill tools: discover, load and install procedures at runtime."""

from __future__ import annotations

from pathlib import Path

from ..safety.policy import Risk
from ..tools.base import ToolContext, ToolRegistry, ToolResult
from .install import install as install_skill


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "skill_list",
        "List available skills — reusable procedures for specific kinds of task. "
        "Search with a query to find one relevant to what you are doing.",
        {
            "properties": {
                "query": {"type": "string", "description": "Filter by name or description"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            }
        },
        risk=Risk.READ,
        group="skill",
    )
    def skill_list(ctx: ToolContext, args: dict) -> ToolResult:
        if ctx.skills is None:
            return ToolResult.text("No skill registry is configured.")
        limit = int(args.get("limit", 40))
        found = ctx.skills.search(args["query"], limit=limit) if args.get("query") else ctx.skills.list()[:limit]
        if not found:
            return ToolResult.text(f"No skill matches {args.get('query', '')!r}.")
        lines = [f"- {s.name}: {s.description}" for s in found]
        return ToolResult(
            ok=True,
            content=f"{len(found)} skill(s):\n" + "\n".join(lines),
            data={"skills": [s.to_dict() for s in found]},
        )

    @registry.tool(
        "skill_load",
        "Load a skill's full instructions. Do this when a skill's description matches "
        "the task at hand — then follow those instructions for the rest of the work.",
        {
            "properties": {"name": {"type": "string", "description": "The skill's name"}},
            "required": ["name"],
        },
        risk=Risk.READ,
        group="skill",
    )
    def skill_load(ctx: ToolContext, args: dict) -> ToolResult:
        if ctx.skills is None:
            return ToolResult.failure("No skill registry is configured.")
        skill = ctx.skills.get(args["name"])
        if ctx.session is not None:
            loaded = ctx.session.metadata.setdefault("loaded_skills", [])
            if skill.name not in loaded:
                loaded.append(skill.name)
        return ToolResult(
            ok=True,
            content=skill.render(),
            data={"name": skill.name, "path": str(skill.path), "chars": len(skill.body)},
        )

    @registry.tool(
        "skill_install",
        "Install a skill from the internet: a git URL, a GitHub 'owner/repo', a zip or "
        "tarball URL, or a local directory. Use this when the user asks for a capability "
        "you do not have and a published skill provides it. The skill is available "
        "immediately afterwards via skill_load.",
        {
            "properties": {
                "source": {
                    "type": "string",
                    "description": "git URL, 'owner/repo', archive URL, or local path",
                },
                "name": {"type": "string", "description": "Override the installed folder name"},
                "overwrite": {"type": "boolean", "description": "Replace an existing skill of the same name"},
            },
            "required": ["source"],
        },
        risk=Risk.WRITE,
        group="skill",
    )
    def skill_install_tool(ctx: ToolContext, args: dict) -> ToolResult:
        target = _skills_dir(ctx)
        result = install_skill(
            args["source"],
            target,
            name=args.get("name"),
            overwrite=bool(args.get("overwrite", False)),
        )
        if ctx.skills is not None:
            ctx.skills.refresh()
        detail = f"Installed: {', '.join(result.installed)}" if result.installed else "Nothing installed."
        if result.skipped:
            detail += f"\nSkipped: {', '.join(result.skipped[:8])}"
        return ToolResult(
            ok=bool(result.installed),
            content=f"{detail}\nInto: {result.destination}",
            data=result.to_dict(),
        )


def _skills_dir(ctx: ToolContext) -> Path:
    config = ctx.config
    if config is not None and hasattr(config, "skills_dir"):
        directory = Path(config.skills_dir)
    else:
        directory = Path.home() / ".lai" / "skills"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
