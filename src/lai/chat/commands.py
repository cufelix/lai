"""Slash commands for the chat interface.

Each one is a small function over (context, argument) returning text to print.
They are collected in ``COMMANDS`` so ``/help`` and tab-completion are derived
from the same table that dispatches them — a command can never exist without
being discoverable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from ..config import PERMISSION_MODES
from . import backends

QUIT = "\x00quit"
NEW = "\x00new"


@dataclass(slots=True)
class Context:
    """What a command is allowed to touch."""

    runtime: object
    ask: Callable[[str, list[str]], int] | None = None
    """Present a numbered choice; returns the index, or -1 for cancelled."""


def cmd_help(ctx: Context, arg: str) -> str:
    width = max(len(name) for name in COMMANDS) + 2
    lines = ["[bold]commands[/bold]"]
    lines += [
        f"  [cyan]/{name.ljust(width)}[/cyan][dim]{spec[1]}[/dim]"
        for name, spec in COMMANDS.items()
        if not spec[2]
    ]
    lines.append("")
    lines.append("[dim]Anything else is a task: “open firefox and find the weather”.[/dim]")
    lines.append("[dim]Ctrl+C interrupts a running task · Ctrl+D quits[/dim]")
    return "\n".join(lines)


def cmd_quit(ctx: Context, arg: str) -> str:
    return QUIT


def cmd_new(ctx: Context, arg: str) -> str:
    return NEW


def cmd_status(ctx: Context, arg: str) -> str:
    runtime = ctx.runtime
    info = backends.describe(runtime)
    lines = [
        f"[bold]{info['name']}/{info['model']}[/bold]"
        + (f"  [dim](configured: {info['configured']})[/dim]" if info["configured"] != info["name"] else ""),
        f"mode [bold]{runtime.config.safety.mode}[/bold] · "
        f"{len(runtime.registry)} tools · {len(runtime.skills)} skills",
    ]
    standbys = list(info["chain"][1:])
    if standbys:
        lines.append("[dim]standby: " + " → ".join(standbys) + "[/dim]")
    for name, why in info["failures"].items():
        lines.append(f"[yellow]! {name} stepped aside:[/yellow] [dim]{why}[/dim]")
    return "\n".join(lines)


def cmd_model(ctx: Context, arg: str) -> str:
    """Switch backend. With no argument, offer a menu of what works."""
    runtime = ctx.runtime
    if arg.strip():
        target, _, model = arg.strip().partition(" ")
        return "[green]now using[/green] " + backends.use(runtime, target, model=model.strip())

    found = [b for b in backends.catalogue() if b.status == "ready"]
    if not found:
        return "[yellow]No backend is ready. Run `lai setup`, or paste a key with /key.[/yellow]"
    if ctx.ask is None:
        return "\n".join(f"  {b.name}  [dim]{b.model}[/dim]" for b in found)

    labels = [f"{b.name}  ({b.model or b.kind})  — {b.detail}" for b in found]
    index = ctx.ask("Which backend should answer?", labels)
    if index < 0:
        return "[dim]unchanged[/dim]"
    return "[green]now using[/green] " + backends.use(runtime, found[index].name)


def cmd_fallback(ctx: Context, arg: str) -> str:
    """Show or set the standby order used when a backend refuses."""
    runtime = ctx.runtime
    text = arg.strip()
    if text:
        if text.lower() in ("off", "none", "no"):
            backends.set_fallback(runtime, [])
            return "[yellow]failover off[/yellow] — a quota error will now end the run"
        if text.lower() == "auto":
            backends.set_fallback(runtime, ["auto"])
            return "[green]failover: auto[/green] — every other working backend, best first"
        chain = backends.set_fallback(runtime, text.replace(",", " ").split())
        return "[green]failover:[/green] " + " → ".join(chain)

    configured = list(runtime.config.provider.fallback)
    live = backends.describe(runtime)["chain"]
    if not configured:
        return "[yellow]failover is off[/yellow] — `/fallback auto` turns it back on"
    lines = ["[bold]failover[/bold] " + " → ".join(configured)]
    if len(live) > 1:
        lines.append("[dim]resolves to: " + " → ".join(live) + "[/dim]")
    return "\n".join(lines)


def cmd_mode(ctx: Context, arg: str) -> str:
    runtime = ctx.runtime
    text = arg.strip().lower()
    if not text:
        if ctx.ask is None:
            return f"mode is [bold]{runtime.config.safety.mode}[/bold] ({', '.join(PERMISSION_MODES)})"
        labels = [
            "readonly — look, never touch",
            "ask — confirm anything risky",
            "auto — act, confirm shell and kills",
            "yolo — no prompts at all",
        ]
        index = ctx.ask("How much may it do on its own?", labels)
        if index < 0:
            return "[dim]unchanged[/dim]"
        text = PERMISSION_MODES[index]
    try:
        return f"[green]mode → {backends.set_mode(runtime, text)}[/green]"
    except ValueError as exc:
        return f"[red]{exc}[/red]"


def cmd_tools(ctx: Context, arg: str) -> str:
    specs = ctx.runtime.registry.specs()
    if arg.strip():
        specs = [s for s in specs if arg.strip().lower() in s.name.lower()]
    lines = [f"  [cyan]{s.name}[/cyan] [dim]({s.risk.value}) {s.description[:80]}[/dim]" for s in specs[:60]]
    lines.append(f"[dim]{len(specs)} tool(s)[/dim]")
    return "\n".join(lines)


def cmd_skills(ctx: Context, arg: str) -> str:
    registry = ctx.runtime.skills
    found = registry.search(arg.strip()) if arg.strip() else registry.list()
    lines = [f"  [green]{s.name}[/green] [dim]{s.description[:80]}[/dim]" for s in found[:40]]
    lines.append(f"[dim]{len(found)} skill(s)[/dim]")
    return "\n".join(lines)


def cmd_observe(ctx: Context, arg: str) -> str:
    return ctx.runtime.desktop.observe(screenshot=False).summary()


def cmd_doctor(ctx: Context, arg: str) -> str:
    from ..checks import run_checks  # noqa: PLC0415

    report = run_checks(ctx.runtime)
    mark = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "fail": "[red]✗[/red]"}
    lines = [f"  {mark[c.status]} {c.label.ljust(24)}[dim]{c.detail}[/dim]" for c in report]
    lines.append("[green]ready[/green]" if report.ready else "[red]not ready — `lai doctor --fix`[/red]")
    return "\n".join(lines)


def cmd_session(ctx: Context, arg: str) -> str:
    session = getattr(ctx.runtime, "extra", {}).get("chat_session")
    return json.dumps(session.summary(), indent=2) if session else "[dim]no session yet[/dim]"


def cmd_web(ctx: Context, arg: str) -> str:
    return (
        "[dim]run [bold]lai web[/bold] in another terminal — it opens the same agent "
        "in your browser[/dim]"
    )


# name → (handler, one-line help, hidden)
COMMANDS: dict[str, tuple] = {
    "help": (cmd_help, "this list", False),
    "status": (cmd_status, "who is answering, and what stands behind them", False),
    "model": (cmd_model, "switch backend — `/model` to pick from a menu", False),
    "fallback": (cmd_fallback, "standby order when a backend refuses", False),
    "mode": (cmd_mode, "permission mode: readonly · ask · auto · yolo", False),
    "new": (cmd_new, "start a fresh session", False),
    "tools": (cmd_tools, "list tools", False),
    "skills": (cmd_skills, "list skills", False),
    "observe": (cmd_observe, "what the agent sees right now", False),
    "doctor": (cmd_doctor, "check this machine", False),
    "web": (cmd_web, "open the same agent in a browser", False),
    "session": (cmd_session, "this session as JSON", False),
    "quit": (cmd_quit, "leave", False),
    "exit": (cmd_quit, "", True),
    "q": (cmd_quit, "", True),
    "h": (cmd_help, "", True),
    "reset": (cmd_new, "", True),
    "provider": (cmd_model, "", True),
}


def run(ctx: Context, line: str) -> str:
    """Dispatch one ``/command``. Unknown commands name the nearest match."""
    name, _, argument = line[1:].strip().partition(" ")
    spec = COMMANDS.get(name.lower())
    if spec is None:
        near = [c for c in COMMANDS if c.startswith(name.lower()[:2])][:3]
        hint = f" — did you mean {', '.join('/' + n for n in near)}?" if near else ""
        return f"[red]unknown command /{name}[/red]{hint}  [dim](/help)[/dim]"
    return spec[0](ctx, argument)
