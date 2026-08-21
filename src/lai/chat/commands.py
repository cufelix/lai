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

MODEL_MENU_LIMIT = 20
"""How many models a menu offers before you are expected to narrow the search."""

QUIT = "\x00quit"
NEW = "\x00new"


@dataclass(slots=True)
class Context:
    """What a command is allowed to touch."""

    runtime: object
    ask: Callable[[str, list[str]], int] | None = None
    """Present a numbered choice; returns the index, or -1 for cancelled."""
    secret: Callable[[str], str] | None = None
    """Ask for something that must not be echoed — an API key."""
    confirm: Callable[[str], bool] | None = None
    """A yes/no question."""
    say: Callable[[str], None] | None = None
    """Print something mid-command, for a flow that takes several steps."""


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
    for name, entry in _resting(runtime).items():
        if name not in info["failures"]:
            lines.append(f"[yellow]⏳ {name}:[/yellow] [dim]{entry}[/dim]")
    return "\n".join(lines)


def _resting(runtime) -> dict:
    """Backends known to be refusing, from earlier runs as well as this one."""
    try:
        from ..agent.providers.health import cooling  # noqa: PLC0415

        return {name: entry.describe() for name, entry in cooling(runtime.config.home).items()}
    except Exception:
        return {}


# Backends worth offering to somebody who has not set one up. One key, many
# models, and a signup page that takes a minute — which is the whole bar for
# "could a person who just installed this get going in five minutes".
SUGGESTED = ("openrouter", "openai", "anthropic", "gemini", "groq", "deepseek")


def cmd_model(ctx: Context, arg: str) -> str:
    """Switch backend. With no argument, offer everything — including what is
    not set up yet, because a choice you cannot see is a choice you do not have.
    """
    runtime = ctx.runtime
    if arg.strip():
        target, _, model = arg.strip().partition(" ")
        return "[green]now using[/green] " + backends.use(runtime, target, model=model.strip())

    catalogue = backends.catalogue(runtime)
    ready = [b for b in catalogue if b.status == "ready"]
    addable = [b for b in catalogue if b.status != "ready" and b.name in SUGGESTED]
    addable.sort(key=lambda b: SUGGESTED.index(b.name))

    if ctx.ask is None:
        lines = [f"  {b.name}  [dim]{b.model}[/dim]" for b in ready]
        lines += [f"  [dim]+ {b.name} — needs a key[/dim]" for b in addable]
        return "\n".join(lines) or "[yellow]No backend is ready. Run `lai setup`.[/yellow]"

    labels = [
        f"{b.name}  ({b.model or b.kind})  — " + (f"⏳ {b.resting}" if b.resting else b.detail)
        for b in ready
    ]
    # A person choosing here has not set anything up, so the label has to read
    # as an invitation rather than as the name of an environment variable.
    labels += [f"+ {b.label or b.name} — add a key, takes a minute" for b in addable]
    if not labels:
        return "[yellow]No backend is ready. Run `lai setup`.[/yellow]"

    index = ctx.ask("Which backend should answer?", labels)
    if index < 0:
        return "[dim]unchanged[/dim]"
    if index < len(ready):
        return "[green]now using[/green] " + backends.use(runtime, ready[index].name)
    return _add_backend(ctx, addable[index - len(ready)])


def _add_backend(ctx: Context, backend) -> str:
    """Walk somebody through adding a backend they do not have yet.

    Three steps, each of which can be abandoned: where to get a key, the key
    itself, and which model to use. Nothing is written until the key has
    answered a real request — being told "saved" and then failing on the first
    task is the worst possible order for those two events.
    """
    say = ctx.say or (lambda text: None)
    if ctx.secret is None:
        return f"[dim]run `/key {backend.name} <your-key>` to add it[/dim]"

    if backend.signup:
        say(f"  Get a key here: [cyan]{backend.signup}[/cyan]")
        if ctx.confirm is not None and ctx.confirm("  open that page in your browser?"):
            _open_url(backend.signup)

    key = ctx.secret(f"  Paste your {backend.name} key (it is not echoed): ")
    if not key.strip():
        return "[dim]nothing pasted — unchanged[/dim]"

    say("  [dim]verifying…[/dim]")
    try:
        label = backends.set_key(ctx.runtime, backend.name, key.strip())
    except Exception as exc:
        return f"[red]that key did not work:[/red] [dim]{str(exc)[:200]}[/dim]"

    say(f"[green]✓ {label}[/green]")
    picked = _offer_models(ctx, backend.name)
    return picked or f"[green]now using[/green] {label}"


def _offer_models(ctx: Context, backend: str) -> str:
    """Right after a key works, the next question is always which model."""
    if ctx.ask is None or ctx.confirm is None:
        return ""
    try:
        from ..models import available_models  # noqa: PLC0415

        found = available_models(backend)
    except Exception:
        return ""
    if len(found) < 2:
        return ""
    if not ctx.confirm(f"  {backend} serves {len(found)} models — pick one now?"):
        return ""
    return cmd_models(ctx, backend)


def _open_url(url: str) -> None:
    import webbrowser  # noqa: PLC0415

    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass


def cmd_models(ctx: Context, arg: str) -> str:
    """Browse what a backend serves, and switch to one of its models.

    `/models` alone lists the backends; `/models openrouter claude` searches
    that vendor's live catalogue; a numbered menu then switches to the choice.
    """
    runtime = ctx.runtime
    words = arg.split()
    if not words:
        return cmd_model(ctx, "")

    backend, query = words[0], " ".join(words[1:])
    try:
        from ..models import available_models  # noqa: PLC0415

        found = available_models(backend)
    except LookupError:
        return f"[red]unknown backend {backend!r}[/red] [dim]— /model lists them[/dim]"
    except Exception as exc:
        return f"[red]{exc}[/red]"

    from ..agent.providers.listing import search  # noqa: PLC0415

    if query:
        found = search(found, query)
    if not found:
        return f"[yellow]{backend} serves nothing matching {query!r}[/yellow]"

    shown = found[:MODEL_MENU_LIMIT]
    if ctx.ask is None:
        lines = [f"  [cyan]{m.id}[/cyan] [dim]{m.describe()}[/dim]" for m in shown]
        lines.append(f"[dim]{len(found)} model(s) — /model {backend} <id> to switch[/dim]")
        return "\n".join(lines)

    index = ctx.ask(
        f"Which {backend} model?",
        [f"{m.id}  ({m.describe()})" if m.describe() else m.id for m in shown],
    )
    if index < 0:
        return "[dim]unchanged[/dim]"
    return "[green]now using[/green] " + backends.use(runtime, backend, model=shown[index].id)


def cmd_key(ctx: Context, arg: str) -> str:
    """Add an API key for a vendor without leaving the conversation.

    `/key openrouter sk-or-…` — verified with a real request before it is
    saved, because a key you find out is wrong during your next task is worse
    than one that never got saved.
    """
    words = arg.split()
    if len(words) < 2:
        return (
            "[dim]usage: /key <backend> <api-key> [model][/dim]\n"
            "[dim]e.g.  /key openrouter sk-or-v1-… z-ai/glm-5.2:free[/dim]"
        )
    backend, key = words[0], words[1]
    model = words[2] if len(words) > 2 else ""
    try:
        return "[green]saved and verified[/green] " + backends.set_key(
            ctx.runtime, backend, key, model=model
        )
    except Exception as exc:
        return f"[red]{type(exc).__name__}: {str(exc)[:200]}[/red]"


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
    "models": (cmd_models, "browse a backend's models — `/models openrouter claude`", False),
    "key": (cmd_key, "add an API key: `/key openrouter sk-or-…`", False),
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
