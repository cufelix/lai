"""Control-flow tools: how the agent declares it is finished or stuck.

Making completion an explicit tool call rather than "the model stopped talking"
gives the loop a real termination signal, forces a verification statement, and
makes autonomous runs auditable.
"""

from __future__ import annotations

from ..safety.policy import Risk
from .base import ToolContext, ToolRegistry, ToolResult

DONE_MARKER = "__lai_task_complete__"
BLOCKED_MARKER = "__lai_task_blocked__"


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "task_complete",
        "Declare the task finished. Call this only after you have verified the result — "
        "re-read the file, re-snapshot the window, check the value actually changed. "
        "The summary is what the user will read.",
        {
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What you did, factually. Mention anything you could not do.",
                },
                "verification": {
                    "type": "string",
                    "description": "How you confirmed it worked (what you observed after acting).",
                },
                "artifacts": {
                    "type": "array",
                    "description": "Files created or modified, if any.",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary"],
        },
        risk=Risk.READ,
        group="control",
    )
    def task_complete(ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            content=DONE_MARKER,
            data={
                "done": True,
                "summary": args["summary"],
                "verification": args.get("verification", ""),
                "artifacts": list(args.get("artifacts", [])),
            },
        )

    @registry.tool(
        "task_blocked",
        "Declare that you cannot continue. Use this instead of looping on an approach "
        "that has already failed, or guessing at an ambiguous requirement.",
        {
            "properties": {
                "reason": {"type": "string", "description": "Precisely what stopped you."},
                "attempted": {
                    "type": "array",
                    "description": "What you already tried.",
                    "items": {"type": "string"},
                },
                "needs": {
                    "type": "string",
                    "description": "What would unblock you — a decision, a permission, a missing app.",
                },
            },
            "required": ["reason"],
        },
        risk=Risk.READ,
        group="control",
    )
    def task_blocked(ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            content=BLOCKED_MARKER,
            data={
                "blocked": True,
                "reason": args["reason"],
                "attempted": list(args.get("attempted", [])),
                "needs": args.get("needs", ""),
            },
        )

    @registry.tool(
        "plan_update",
        "Record or revise your plan for a multi-step task. Use it at the start of "
        "anything non-trivial and whenever the plan changes — it keeps you honest about "
        "what is left and gives the user something to follow.",
        {
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "Ordered steps, each a short imperative phrase.",
                    "items": {"type": "string"},
                },
                "current": {"type": "integer", "description": "0-based index of the step you are on", "minimum": 0},
                "note": {"type": "string"},
            },
            "required": ["steps"],
        },
        risk=Risk.READ,
        group="control",
    )
    def plan_update(ctx: ToolContext, args: dict) -> ToolResult:
        steps = [str(s) for s in args["steps"]]
        current = int(args.get("current", 0))
        if ctx.session is not None:
            ctx.session.metadata["plan"] = {"steps": steps, "current": current}
        rendered = "\n".join(
            f"{'→' if i == current else ' '} {i + 1}. {step}" for i, step in enumerate(steps)
        )
        note = f"\n{args['note']}" if args.get("note") else ""
        return ToolResult(
            ok=True,
            content=f"Plan recorded ({len(steps)} steps, on step {current + 1}):\n{rendered}{note}",
            data={"steps": steps, "current": current},
        )
