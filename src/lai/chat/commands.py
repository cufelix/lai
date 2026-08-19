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


def _journal(ctx: Context):
    return getattr(ctx.runtime, "journal", None)


def cmd_notes(ctx: Context, arg: str) -> str:
    """What the agent has learned about this machine."""
    from .notes import render_list, render_note  # noqa: PLC0415

    journal = _journal(ctx)
    if journal is None:
        return "[yellow]the journal is unavailable[/yellow]"
    if arg.strip():
        note = journal.get(arg.strip())
        return render_note(note) if note else f"[red]no note called {arg.strip()!r}[/red]"
    return render_list(journal.list())


def cmd_note(ctx: Context, arg: str) -> str:
    journal = _journal(ctx)
    if journal is None or not arg.strip():
        return "[dim]usage: /note <name>[/dim]"
    from .notes import render_note  # noqa: PLC0415

    note = journal.get(arg.strip())
    return render_note(note) if note else f"[red]no note called {arg.strip()!r}[/red]"


def cmd_edit(ctx: Context, arg: str) -> str:
    """Open a note in $EDITOR — the agent's beliefs, corrected by hand."""
    from .notes import edit  # noqa: PLC0415

    journal = _journal(ctx)
    if journal is None:
        return "[yellow]the journal is unavailable[/yellow]"
    name = arg.strip()
    if not name:
        return "[dim]usage: /edit <name> — /notes lists them[/dim]"
    return edit(journal, name)


def cmd_learn(ctx: Context, arg: str) -> str:
    """Teach it something directly: /learn drawing: the canvas starts at y=140."""
    journal = _journal(ctx)
    if journal is None:
        return "[yellow]the journal is unavailable[/yellow]"
    topic, sep, lesson = arg.partition(":")
    if not sep or not lesson.strip():
        return "[dim]usage: /learn <topic>: <what you know>[/dim]"
    note = journal.append(topic.strip(), lesson.strip())
    return f"[green]noted[/green] [dim]{note.name} — {len(note.body.splitlines())} line(s)[/dim]"


def cmd_forget(ctx: Context, arg: str) -> str:
    journal = _journal(ctx)
    if journal is None or not arg.strip():
        return "[dim]usage: /forget <name>[/dim]"
    return (
        f"[yellow]forgot {arg.strip()}[/yellow]" if journal.delete(arg.strip())
        else f"[red]no note called {arg.strip()!r}[/red]"
    )


RESUME = "\x00resume:"


def cmd_resume(ctx: Context, arg: str) -> str:
    """Pick up an earlier conversation, by id or simply the last one."""
    return RESUME + arg.strip()


def cmd_sessions(ctx: Context, arg: str) -> str:
    from ..agent.session import Session  # noqa: PLC0415

    listing = Session.list_sessions(ctx.runtime.config.sessions_dir, limit=15)
    if not listing:
        return "[dim]no past sessions yet[/dim]"
    import time as _time  # noqa: PLC0415

    lines = []
    for entry in listing:
        when = _time.strftime("%d %b %H:%M", _time.localtime(entry["modified"]))
        lines.append(f"  [cyan]{entry['id']}[/cyan]  [dim]{when}  {(entry['task'] or '')[:60]}[/dim]")
    lines.append("[dim]/resume <id> to continue one, /resume for the most recent[/dim]")
    return "\n".join(lines)


def cmd_settings(ctx: Context, arg: str) -> str:
    """Everything you can change, and what it is set to."""
    runtime = ctx.runtime
    info = backends.describe(runtime)
    learning = getattr(runtime.config, "learning", None)
    journal = _journal(ctx)
    notes = len(journal.list()) if journal is not None else 0

    rows = [
        ("model", f"{info['name']}/{info['model']}", "/model"),
        ("failover", " → ".join(info["fallback"]) or "off", "/fallback"),
        ("permissions", runtime.config.safety.mode, "/mode"),
        ("learning", ("on" if learning and learning.enabled else "off") + f" · {notes} note(s)", "/learning"),
        ("tools", str(len(runtime.registry)), "/tools"),
        ("skills", str(len(runtime.skills)), "/skills"),
        ("config", str(runtime.config.home / "config.toml"), ""),
    ]
    width = max(len(label) for label, _, _ in rows) + 2
    lines = ["[bold]Settings[/bold]"]
    lines += [
        f"  {label.ljust(width)}[bold]{value}[/bold]" + (f"  [dim]{command}[/dim]" if command else "")
        for label, value, command in rows
    ]
    return "\n".join(lines)


def cmd_learning(ctx: Context, arg: str) -> str:
    """Turn the journal on or off."""
    from dataclasses import replace  # noqa: PLC0415

    runtime = ctx.runtime
    current = getattr(runtime.config, "learning", None)
    if current is None:
        return "[yellow]learning is unavailable in this build[/yellow]"
    text = arg.strip().lower()
    if text not in ("on", "off"):
        state = "on" if current.enabled else "off"
        return f"learning is [bold]{state}[/bold] [dim](/learning on|off)[/dim]"
    enabled = text == "on"
    runtime.config = runtime.config.with_overrides(
        learning=replace(current, enabled=enabled, reflect=enabled)
    )
    backends.save(runtime.config, {"learning": {"enabled": enabled, "reflect": enabled}})
    return f"[green]learning {text}[/green]"


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
    "resume": (cmd_resume, "continue an earlier conversation", False),
    "sessions": (cmd_sessions, "past conversations", False),
    "settings": (cmd_settings, "everything you can change, and what it is set to", False),
    "notes": (cmd_notes, "what it has learned about this machine", False),
    "learn": (cmd_learn, "teach it: /learn <topic>: <what you know>", False),
    "edit": (cmd_edit, "open a note in $EDITOR", False),
    "forget": (cmd_forget, "delete a note", False),
    "learning": (cmd_learning, "turn note-keeping on or off", False),
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
    "note": (cmd_note, "", True),
    "config": (cmd_settings, "", True),
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
