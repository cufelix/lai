"""Agent-runtime tools: memory, delegation, and scheduling.

These tools give the model three capabilities beyond the base desktop toolset,
all of which are about the agent managing *itself* rather than the desktop:
remembering things across sessions, offloading a subtask into an isolated
context, and arranging to run again later without a human asking each time.

Wiring note: ``memory_save``/``memory_search``/``memory_forget`` and
``schedule_*`` look for a ready-made store on ``ctx.extra`` first
(``ctx.extra["memory_store"]``, ``ctx.extra["task_store"]`` or
``ctx.extra["scheduler"]``) and only fall back to opening one at the default
path under ``ctx.config.home`` if nothing was injected — this is what lets
tests supply an in-memory or temp-dir store without touching global state.
``delegate`` is different: it needs the *running* :class:`~lai.agent.loop.Agent`
that issued the call (to share its provider, registry, desktop, policy and
audit log), which :class:`~lai.tools.base.ToolContext` has no field for, so it
must be supplied as ``ctx.extra["agent"]`` by whoever drives the loop.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

from ..agent.memory import KINDS as MEMORY_KINDS
from ..agent.memory import MemoryStore
from ..agent.subagent import SubagentDepthExceeded, run_subagent
from ..safety.policy import Risk
from ..scheduler import Scheduler, TaskStore, make_task
from .base import ToolContext, ToolRegistry, ToolResult

DEFAULT_MEMORY_LIMIT = 10
DEFAULT_DELEGATE_STEPS = 15
MAX_DELEGATE_STEPS = 30


def register(registry: ToolRegistry) -> None:
    # -- memory ------------------------------------------------------

    @registry.tool(
        "memory_save",
        "Remember something that will still be true and useful in a *future, unrelated* "
        "session — not what you are doing right now. The best candidates are: how a "
        "specific application behaves (an exact button or field label, a dialog quirk, "
        "a menu path that isn't obvious), a preference the user stated ('always save to "
        "~/Documents', 'use dark mode'), or a durable fact about their setup. Do NOT use "
        "this to restate the current task, log a one-off result, or save anything that is "
        "already obvious from context — that just adds noise future sessions have to wade "
        "through. Pass `key` when this is a fact you might need to correct later (e.g. "
        "'xed.save_field_label'); a second save with the same kind+key replaces it instead "
        "of creating a duplicate.",
        {
            "properties": {
                "content": {"type": "string", "description": "The fact itself, stated plainly."},
                "kind": {
                    "type": "string",
                    "enum": list(MEMORY_KINDS),
                    "description": "fact | app (how an application behaves) | preference | task",
                },
                "key": {
                    "type": "string",
                    "description": "Stable name for this fact, to upsert on later. Omit for a one-off note.",
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"],
        },
        risk=Risk.WRITE,
        group="agentic",
    )
    def memory_save(ctx: ToolContext, args: dict) -> ToolResult:
        kind = args.get("kind") or "fact"
        try:
            with _memory_store(ctx) as store:
                entry = store.remember(
                    args["content"],
                    kind=kind,
                    key=args.get("key") or None,
                    tags=tuple(args.get("tags") or ()),
                    source_session=getattr(ctx.session, "id", "") or "",
                )
        except ValueError as exc:
            return ToolResult.failure(f"could not save memory: {exc}")
        # Not ToolResult.text(**entry.to_dict()): MemoryEntry.to_dict() has its
        # own "content" key, which collides with .text()'s "content" parameter.
        return ToolResult(
            ok=True,
            content=f"Remembered ({entry.kind}{f', key={entry.key}' if entry.key else ''}): {entry.content}",
            data=entry.to_dict(),
        )

    @registry.tool(
        "memory_search",
        "Recall things learned in earlier sessions before assuming you don't know how "
        "something behaves — check this at the start of work on an app or topic you may "
        "have touched before, not only when you are already stuck.",
        {
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for. Empty returns the most recent memories.",
                },
                "kind": {"type": "string", "enum": list(MEMORY_KINDS)},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
        risk=Risk.READ,
        group="agentic",
    )
    def memory_search(ctx: ToolContext, args: dict) -> ToolResult:
        limit = int(args.get("limit") or DEFAULT_MEMORY_LIMIT)
        with _memory_store(ctx) as store:
            entries = store.recall(args.get("query", ""), kind=args.get("kind") or None, limit=limit)
        if not entries:
            return ToolResult.text("No relevant memories found.", results=[])
        lines = [f"- [{e.kind}] {e.content}" + (f" (key={e.key})" if e.key else "") for e in entries]
        return ToolResult(
            ok=True,
            content=f"{len(entries)} memor{'y' if len(entries) == 1 else 'ies'}:\n" + "\n".join(lines),
            data={"results": [e.to_dict() for e in entries]},
        )

    @registry.tool(
        "memory_forget",
        "Delete a remembered fact that turned out to be wrong, outdated, or was saved by "
        "mistake. Provide either the numeric `id` from a memory_search result, or the exact "
        "`key` it was saved under.",
        {
            "properties": {
                "id_or_key": {"type": "string", "description": "A memory id (e.g. '7') or its key."},
            },
            "required": ["id_or_key"],
        },
        risk=Risk.WRITE,
        group="agentic",
    )
    def memory_forget(ctx: ToolContext, args: dict) -> ToolResult:
        with _memory_store(ctx) as store:
            removed = store.forget(args["id_or_key"])
        if not removed:
            return ToolResult.failure(f"no memory found matching {args['id_or_key']!r}")
        return ToolResult.text(f"Forgot {removed} matching memor{'y' if removed == 1 else 'ies'}.", removed=removed)

    # -- delegation ----------------------------------------------------

    @registry.tool(
        "delegate",
        "Run a focused, self-contained subtask in an isolated context and get back only "
        "its conclusion — not the step-by-step transcript. Worth it when a subtask is long "
        "or exploratory enough that doing it inline would flood your own context with "
        "screenshots and intermediate steps you don't need to remember afterward (e.g. "
        "'search these ten log files and summarize the root cause', 'work through this "
        "multi-step install wizard and report whether it succeeded'). NOT worth it for a "
        "quick single action you could just do yourself, or for a subtask that depends on "
        "fine-grained context you already have loaded — you would have to re-explain all of "
        "it in `task`, which defeats the purpose. Delegation nests at most two levels deep.",
        {
            "properties": {
                "task": {"type": "string", "description": "A complete, self-contained description of the subtask."},
                "max_steps": {
                    "type": "integer", "minimum": 1, "maximum": MAX_DELEGATE_STEPS,
                    "description": f"Step budget for the subagent (default {DEFAULT_DELEGATE_STEPS}).",
                },
                "system_extra": {
                    "type": "string",
                    "description": "Optional extra instructions/context to prepend for the subagent only.",
                },
            },
            "required": ["task"],
        },
        risk=Risk.WRITE,
        group="agentic",
    )
    def delegate(ctx: ToolContext, args: dict) -> ToolResult:
        parent = (ctx.extra or {}).get("agent")
        if parent is None:
            return ToolResult.failure(
                "delegate is unavailable: no running agent was wired into ctx.extra['agent']"
            )
        try:
            result = run_subagent(
                task=args["task"],
                parent=parent,
                max_steps=int(args.get("max_steps") or DEFAULT_DELEGATE_STEPS),
                system_extra=args.get("system_extra") or "",
            )
        except SubagentDepthExceeded as exc:
            return ToolResult.failure(str(exc), **exc.to_dict())
        except ValueError as exc:
            return ToolResult.failure(f"delegate: {exc}")
        return ToolResult(
            ok=result.status == "completed",
            content=f"Subagent {result.status}: {result.summary}",
            data=result.to_dict(),
        )

    # -- scheduling ------------------------------------------------------

    @registry.tool(
        "schedule_task",
        "Schedule a task to run automatically in the future, on a recurring cron pattern "
        "or a fixed interval — for work the user wants repeated or deferred without asking "
        "again each time. Not for anything they want done right now; do that directly. "
        "`schedule` is either a standard 5-field cron expression (e.g. '0 9 * * 1-5' for "
        "weekday mornings), one of @hourly/@daily/@weekly/@monthly, or 'every:<seconds>'.",
        {
            "properties": {
                "name": {"type": "string", "description": "Short human-readable label."},
                "task": {"type": "string", "description": "The prompt to run when this fires."},
                "schedule": {"type": "string"},
                "mode": {"type": "string", "description": "Permission mode to run under (default 'auto')."},
            },
            "required": ["name", "task", "schedule"],
        },
        risk=Risk.WRITE,
        group="agentic",
    )
    def schedule_task(ctx: ToolContext, args: dict) -> ToolResult:
        try:
            entry = make_task(
                name=args["name"], task=args["task"], schedule=args["schedule"],
                mode=args.get("mode") or "auto",
            )
        except ValueError as exc:
            return ToolResult.failure(f"invalid schedule: {exc}")
        _task_store(ctx).add(entry)
        return ToolResult.text(
            f"Scheduled {entry.name!r} ({entry.schedule}), id={entry.id}.", **entry.to_dict()
        )

    @registry.tool(
        "schedule_list",
        "List the currently scheduled tasks, including when each last ran and is next due.",
        {"properties": {"enabled_only": {"type": "boolean"}}},
        risk=Risk.READ,
        group="agentic",
    )
    def schedule_list(ctx: ToolContext, args: dict) -> ToolResult:
        tasks = _task_store(ctx).list(enabled_only=bool(args.get("enabled_only", False)))
        if not tasks:
            return ToolResult.text("No scheduled tasks.", tasks=[])
        lines = [f"- [{t.id}] {t.name!r} ({t.schedule}) — {'enabled' if t.enabled else 'disabled'}" for t in tasks]
        return ToolResult(
            ok=True,
            content=f"{len(tasks)} scheduled task(s):\n" + "\n".join(lines),
            data={"tasks": [t.to_dict() for t in tasks]},
        )

    @registry.tool(
        "schedule_remove",
        "Cancel a scheduled task so it will not run again.",
        {"properties": {"id": {"type": "string"}}, "required": ["id"]},
        risk=Risk.WRITE,
        group="agentic",
    )
    def schedule_remove(ctx: ToolContext, args: dict) -> ToolResult:
        removed = _task_store(ctx).remove(args["id"])
        if not removed:
            return ToolResult.failure(f"no scheduled task with id {args['id']!r}")
        return ToolResult.text(f"Removed scheduled task {args['id']!r}.", removed=True)


# -- store lookup -------------------------------------------------------


@contextlib.contextmanager
def _memory_store(ctx: ToolContext) -> Iterator[MemoryStore]:
    """Yield the injected store from ``ctx.extra``, or open+close a fresh one.

    Only a freshly-opened fallback store is closed on exit — an injected store
    is assumed to be owned (and kept open) by whoever put it there.
    """
    injected = (ctx.extra or {}).get("memory_store")
    if injected is not None:
        yield injected
        return
    home = Path(getattr(ctx.config, "home", Path.home() / ".lai"))
    store = MemoryStore.open(home)
    try:
        yield store
    finally:
        store.close()


def _task_store(ctx: ToolContext) -> TaskStore:
    extra = ctx.extra or {}
    scheduler = extra.get("scheduler")
    if isinstance(scheduler, Scheduler):
        return scheduler.store
    store = extra.get("task_store")
    if store is not None:
        return store
    home = Path(getattr(ctx.config, "home", Path.home() / ".lai"))
    return TaskStore(home / "schedule.json")


__all__ = ["register"]
