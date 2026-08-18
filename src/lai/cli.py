"""Command line interface.

``lai do "<task>"`` runs one autonomous task. ``lai repl`` is the interactive
session. ``lai mcp`` turns LAI into an MCP server so Claude Code — or any MCP
client — gains full native desktop control.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

from .config import PERMISSION_MODES, load_config
from .errors import LaiError
from .runtime import build_runtime

BANNER = "LAI — native desktop agent"

# The command list, once. `--help` renders from here and build_parser pulls its
# help strings from here, so a new command can never exist in one and not the
# other — a help screen that lies about the commands is worse than none.
COMMAND_GROUPS: dict[str, str] = {
    "start here": "set up in a minute",
    "run tasks": "give the agent work",
    "inspect": "see what it sees",
    "connect": "reach it from anywhere",
}
COMMANDS: dict[str, tuple[str, str]] = {
    "setup": ("start here", "Guided first-run setup: model, permissions, self-check"),
    "doctor": ("start here", "Check the environment — --fix repairs what it can"),
    "models": ("start here", "Model backends: list, test one, pick a default"),
    "do": ("run tasks", "Run one task autonomously"),
    "repl": ("run tasks", "Interactive session in the terminal"),
    "tui": ("run tasks", "Full-screen dashboard: live view, costs, approvals"),
    "observe": ("inspect", "Print what the agent currently sees"),
    "sessions": ("inspect", "List or inspect past sessions"),
    "tools": ("inspect", "List available tools"),
    "skills": ("inspect", "Manage skills"),
    "serve": ("connect", "Run the HTTP daemon and remote channels"),
    "channels": ("connect", "Manage remote connectors and who may use them"),
    "schedule": ("connect", "Recurring tasks, run by `lai serve`"),
    "mcp": ("connect", "Expose desktop tools over MCP to Claude Code"),
}
EXAMPLES = [
    ('lai do "open firefox and find the weather for prague"', "just do one thing"),
    ("lai models use cli:claude", "think with an installed coding CLI, no API key"),
    ("lai tui", "watch it work in real time"),
    ("lai doctor --fix", "repair what is broken"),
]


# -- output helpers ------------------------------------------------------


class Out:
    """Console output. Uses rich when available, plain text otherwise."""

    def __init__(self, *, quiet: bool = False, color: bool = True) -> None:
        self.quiet = quiet
        self._console = None
        if color and not quiet:
            try:
                from rich.console import Console  # noqa: PLC0415

                self._console = Console(stderr=False, highlight=False, soft_wrap=True)
            except ImportError:
                self._console = None

    def write(self, text: str = "", *, style: str = "") -> None:
        if self.quiet:
            return
        if self._console is not None and style:
            self._console.print(text, style=style)
        elif self._console is not None:
            self._console.print(text)
        else:
            print(_strip_markup(text))

    def raw(self, text: str) -> None:
        if self.quiet:
            return
        sys.stdout.write(text)
        sys.stdout.flush()

    def error(self, text: str) -> None:
        sys.stderr.write(f"error: {_strip_markup(text)}\n")

    def rule(self, title: str = "") -> None:
        if self.quiet:
            return
        if self._console is not None:
            self._console.rule(title)
        else:
            width = shutil.get_terminal_size((80, 24)).columns
            print(f"── {title} ".ljust(width, "─") if title else "─" * width)

    def spinner(self, text: str):
        """A running status line; stop() it. None whenever there is nowhere to show it.

        The gap between one tool finishing and the model saying anything next is
        where users assume the thing died. A spinner is the cheapest possible
        answer to "is it still working?".
        """
        if self.quiet or self._console is None:
            return None
        try:
            status = self._console.status(text)
            status.start()
            return status
        except Exception:
            return None


def _strip_markup(text: str) -> str:
    import re

    return re.sub(r"\[/?[a-z0-9 _#]+\]", "", str(text))


# -- shared arguments ----------------------------------------------------


def _apply_overrides(config, args):
    provider = config.provider
    if getattr(args, "model", None):
        provider = replace(provider, model=args.model)
    if getattr(args, "provider", None):
        provider = replace(provider, name=args.provider)
    if getattr(args, "thinking", None):
        provider = replace(provider, thinking_budget=int(args.thinking))

    safety = config.safety
    if getattr(args, "mode", None):
        safety = replace(safety, mode=args.mode)
    if getattr(args, "dry_run", False):
        safety = replace(safety, dry_run=True)

    limits = config.limits
    if getattr(args, "steps", None):
        limits = replace(limits, max_steps=int(args.steps))
    if getattr(args, "timeout", None):
        limits = replace(limits, max_seconds=float(args.timeout))

    return config.with_overrides(provider=provider, safety=safety, limits=limits)


def _interactive_approver(out: Out):
    """Ask the human before a gated action. Non-tty means refuse."""

    def approve(name: str, tool_input: dict, verdict) -> bool:
        if not sys.stdin.isatty():
            out.write(f"[yellow]refused (no tty): {name} — {verdict.reason}[/yellow]")
            return False
        preview = json.dumps(tool_input, ensure_ascii=False)[:300]
        out.write(f"\n[yellow]● approval needed[/yellow] [bold]{name}[/bold]  {preview}")
        out.write(f"  reason: {verdict.reason}")
        try:
            answer = input("  allow? [y/N/a=allow this tool for the rest of the run] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer == "a":
            approve.always.add(name)  # type: ignore[attr-defined]
            return True
        return answer in ("y", "yes")

    approve.always = set()  # type: ignore[attr-defined]

    def wrapper(name: str, tool_input: dict, verdict) -> bool:
        if name in approve.always:  # type: ignore[attr-defined]
            return True
        return approve(name, tool_input, verdict)

    return wrapper


def _make_reporter(out: Out, *, stream: bool = True, verbose: bool = False):
    """Render loop events to the console."""
    state = {"streaming": False, "status": None}

    def spin() -> None:
        if state["status"] is None:
            state["status"] = out.spinner("[dim]thinking…[/dim]")

    def halt() -> None:
        if state["status"] is not None:
            state["status"].stop()
            state["status"] = None

    def report(kind: str, payload: dict) -> None:
        if kind == "start":
            out.write(
                f"[dim]{payload['provider']}/{payload['model']}[/dim] → [bold]{payload['task']}[/bold]"
            )
            out.rule()
            spin()
        elif kind == "step" and verbose:
            out.write(f"[dim]— step {payload['step']}/{payload['of']}[/dim]")
        elif kind == "text" and stream:
            halt()
            out.raw(payload.get("delta", ""))
            state["streaming"] = True
        elif kind == "thinking" and verbose and stream:
            halt()
            out.raw(payload.get("delta", ""))
        elif kind == "assistant" and not stream:
            halt()
            out.write(payload.get("text", ""))
        elif kind == "tool_call":
            halt()
            if state["streaming"]:
                out.raw("\n")
                state["streaming"] = False
            args = json.dumps(payload.get("input", {}), ensure_ascii=False)
            out.write(f"[cyan]▸ {payload['name']}[/cyan] [dim]{args[:160]}[/dim]")
        elif kind == "tool_result":
            halt()
            mark = "[green]✓[/green]" if payload.get("ok") else "[red]✗[/red]"
            summary = (payload.get("summary") or "").strip().splitlines()
            first = summary[0][:170] if summary else ""
            extra = f" [dim](+{len(summary) - 1} lines)[/dim]" if len(summary) > 1 else ""
            image = " [magenta]+image[/magenta]" if payload.get("images") else ""
            out.write(f"  {mark} [dim]{first}[/dim]{extra}{image}")
            spin()
        elif kind == "compacting":
            halt()
            out.write(f"[dim]… compacting context ({payload['estimated_tokens']} tokens)[/dim]")
            spin()
        elif kind == "error":
            halt()
            out.write(f"[red]! {payload.get('error', '')}[/red]")
        elif kind == "done":
            halt()
            if state["streaming"]:
                out.raw("\n")
            out.rule()

    return report


def _no_provider(out: Out, runtime) -> int:
    """The most common wall a new user hits. Name the command that gets past it."""
    out.error(runtime.provider_error or "no model backend available")
    out.write("")
    out.write("[yellow]LAI needs a model to think with.[/yellow]")
    out.write("  [bold]lai setup[/bold]   guided — paste a key, or use a local ollama")
    out.write("  [dim]or export one of ANTHROPIC_API_KEY / ZAI_API_KEY / OPENAI_API_KEY[/dim]")
    return 2


# -- commands ------------------------------------------------------------


def cmd_do(args) -> int:
    out = Out(quiet=args.json)
    config = _apply_overrides(load_config(), args)
    runtime = build_runtime(config, with_mcp=not args.no_mcp)
    try:
        if runtime.provider is None:
            return _no_provider(out, runtime)
        agent = runtime.agent(
            approver=_interactive_approver(out),
            on_event=None if args.json else _make_reporter(out, stream=not args.no_stream, verbose=args.verbose),
        )
        result = agent.run(args.task)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            out.write(result.render())
            usage = result.usage
            out.write(
                f"[dim]{usage.input_tokens} in / {usage.output_tokens} out tokens · "
                f"session {result.session_id}[/dim]"
            )
        return 0 if result.ok else 1
    finally:
        runtime.close()


def cmd_repl(args) -> int:
    out = Out()
    config = _apply_overrides(load_config(), args)
    runtime = build_runtime(config, with_mcp=not args.no_mcp)
    if runtime.provider is None:
        code = _no_provider(out, runtime)
        runtime.close()
        return code

    from . import __version__  # noqa: PLC0415

    out.write(f"[bold]{BANNER}[/bold] [dim]v{__version__}[/dim]")
    out.write(
        f"[dim]{runtime.provider.name}/{runtime.provider.model} · mode={config.safety.mode} · "
        f"{len(runtime.registry)} tools · {len(runtime.skills)} skills[/dim]"
    )
    out.write(
        "[dim]Type a task. /help for commands, /quit to exit — "
        "or run [bold]lai tui[/bold] for the full-screen dashboard.[/dim]\n"
    )

    from .agent.session import Session  # noqa: PLC0415

    session = Session()
    approver = _interactive_approver(out)
    reporter = _make_reporter(out, verbose=args.verbose)

    try:
        while True:
            try:
                line = input("lai> ").strip()
            except (EOFError, KeyboardInterrupt):
                out.write("\nbye")
                return 0
            if not line:
                continue
            if line.startswith("/"):
                action = _repl_command(line, runtime, session, out)
                if action == "quit":
                    return 0
                if action == "reset":
                    session = Session()
                continue

            agent = runtime.agent(session=session, approver=approver, on_event=reporter)
            try:
                result = agent.run(line)
                out.write(result.render())
            except KeyboardInterrupt:
                agent.interrupt()
                out.write("\n[yellow]interrupted[/yellow]")
            except LaiError as exc:
                out.error(str(exc))
    finally:
        runtime.close()


def _repl_command(line: str, runtime, session, out: Out) -> str:
    command, _, rest = line[1:].partition(" ")
    command = command.lower()
    if command in ("quit", "exit", "q"):
        return "quit"
    if command in ("help", "h", "?"):
        out.write(
            "/help  /quit  /reset (new session)  /tools [filter]  /skills [query]  "
            "/mode <readonly|ask|auto|yolo>  /session  /observe"
        )
    elif command == "reset":
        out.write("[dim]new session[/dim]")
        return "reset"
    elif command == "tools":
        specs = runtime.registry.specs()
        if rest.strip():
            specs = [s for s in specs if rest.strip().lower() in s.name.lower()]
        for spec in specs:
            out.write(f"  [cyan]{spec.name}[/cyan] [dim]({spec.risk.value})[/dim] {spec.description[:90]}")
        out.write(f"[dim]{len(specs)} tool(s)[/dim]")
    elif command == "skills":
        found = runtime.skills.search(rest.strip()) if rest.strip() else runtime.skills.list()
        for skill in found[:40]:
            out.write(f"  [green]{skill.name}[/green] {skill.description[:90]}")
        out.write(f"[dim]{len(found)} skill(s)[/dim]")
    elif command == "mode":
        mode = rest.strip()
        if mode not in PERMISSION_MODES:
            out.write(f"[red]mode must be one of {', '.join(PERMISSION_MODES)}[/red]")
        else:
            runtime.policy.config = replace(runtime.policy.config, mode=mode)
            out.write(f"[dim]permission mode → {mode}[/dim]")
    elif command == "session":
        out.write(json.dumps(session.summary(), indent=2))
    elif command == "observe":
        observation = runtime.desktop.observe(screenshot=False)
        out.write(observation.summary())
    else:
        out.write(f"[red]unknown command /{command}[/red] — try /help")
    return "ok"


def cmd_doctor(args) -> int:
    """Diagnose the machine — and, with --fix, repair what can be repaired."""
    out = Out(quiet=getattr(args, "json", False))
    config = load_config()
    from .checks import FAIL, OK, WARN, run_checks  # noqa: PLC0415

    # MCP is off unless asked for: connecting every configured server takes
    # tens of seconds, and `lai doctor` is the command someone runs *because*
    # something is already wrong. The environment check must be instant.
    with_mcp = bool(getattr(args, "mcp", False))
    runtime = build_runtime(config, with_provider=True, with_mcp=with_mcp)
    try:
        report = run_checks(runtime, config)

        if getattr(args, "json", False):
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.ready else 1

        out.write(f"[bold]{BANNER}[/bold] — diagnostics\n")
        for check in report:
            icon = {OK: "[green]✓[/green]", WARN: "[yellow]![/yellow]", FAIL: "[red]✗[/red]"}[check.status]
            out.write(f" {icon} [bold]{check.label:24}[/bold] [dim]{check.detail}[/dim]")

        out.write("")
        out.write(f" [dim]tools {len(runtime.registry)} · skills {len(runtime.skills)} · "
                  f"mode {config.safety.mode} · home {config.home}[/dim]")
        if with_mcp:
            problems = f"; errors: {runtime.mcp_errors}" if runtime.mcp_errors else ""
            out.write(f" [dim]mcp   {len(runtime.mcp_tools)} tool(s){problems}[/dim]")
        else:
            out.write(" [dim]mcp   not checked — add --mcp to connect external servers[/dim]")

        repairable = [c for c in report if c.status != OK and c.fix is not None]
        if repairable and not getattr(args, "fix", False):
            out.write("")
            out.write("[bold]Fixes available:[/bold]")
            for check in repairable:
                fix = check.fix
                shell = fix.shell()
                out.write(f"  [bold]{check.label}[/bold] — {fix.description}")
                if shell:
                    out.write(f"    [cyan]{shell}[/cyan]")
                for line in (fix.manual or "").splitlines():
                    out.write(f"    [dim]{line}[/dim]")
            out.write("")
            out.write("  [dim]run `lai doctor --fix` to apply them, or `lai setup` for the guided path[/dim]")

        if repairable and getattr(args, "fix", False):
            out.write("")
            out.write("[bold]Applying fixes[/bold]")
            for check in repairable:
                fix = check.fix
                if not fix.automatic:
                    out.write(f"  [yellow]{check.label}[/yellow]: needs you — {fix.manual or fix.description}")
                    continue
                out.write(f"  [cyan]{fix.shell() or fix.description}[/cyan]")
                ok, output = fix.run()
                out.write("    [green]✓[/green]" if ok else f"    [red]✗ {output.strip()[:200]}[/red]")
            report = run_checks(runtime, config)

        out.write("")
        if report.ready:
            out.write("[green]Ready.[/green]")
        else:
            out.write("[yellow]Not ready — see the failures above, or run `lai setup`.[/yellow]")
        return 0 if report.ready else 1
    finally:
        runtime.close()


def cmd_setup(args) -> int:
    """Guided first-run setup."""
    from .setup_wizard import run_setup  # noqa: PLC0415

    out = Out()
    code, _ = run_setup(
        out,
        assume_yes=getattr(args, "yes", False),
        skip_demo=getattr(args, "no_demo", False),
    )
    return code


def cmd_observe(args) -> int:
    out = Out(quiet=args.json)
    runtime = build_runtime(load_config(), with_provider=False, with_mcp=False)
    try:
        observation = runtime.desktop.observe(
            screenshot=bool(args.screenshot), scope=args.scope, annotate_elements=args.annotate
        )
        if args.json:
            print(json.dumps(observation.to_dict(), ensure_ascii=False, indent=2, default=str))
        else:
            out.write(observation.summary(max_elements=args.limit))
        if args.screenshot and observation.screenshot and args.save:
            path = Path(args.save)
            path.write_bytes(observation.screenshot.png)
            out.write(f"[dim]screenshot → {path}[/dim]")
        return 0
    finally:
        runtime.close()


def cmd_tools(args) -> int:
    out = Out(quiet=args.json)
    runtime = build_runtime(load_config(), with_provider=False, with_mcp=not args.no_mcp)
    try:
        specs = runtime.registry.specs()
        if args.filter:
            specs = [s for s in specs if args.filter.lower() in s.name.lower()]
        if args.json:
            print(json.dumps(runtime.registry.to_anthropic(), ensure_ascii=False, indent=2))
            return 0
        group = ""
        for spec in sorted(specs, key=lambda s: (s.group, s.name)):
            if spec.group != group:
                group = spec.group
                out.write(f"\n[bold]{group}[/bold]")
            out.write(f"  [cyan]{spec.name:24}[/cyan] [dim]{spec.risk.value:12}[/dim] {spec.description[:80]}")
        out.write(f"\n[dim]{len(specs)} tool(s)[/dim]")
        return 0
    finally:
        runtime.close()


def cmd_skills(args) -> int:
    out = Out()
    config = load_config()
    runtime = build_runtime(config, with_provider=False, with_mcp=False)
    try:
        action = args.action or "list"
        if action == "list":
            found = runtime.skills.search(args.query) if args.query else runtime.skills.list()
            for skill in found:
                out.write(f"  [green]{skill.name:32}[/green] [dim]{skill.description[:88]}[/dim]")
            out.write(f"\n[dim]{len(found)} skill(s)[/dim]")
        elif action == "show":
            if not args.query:
                out.error("usage: lai skills show <name>")
                return 2
            out.write(runtime.skills.get(args.query).render())
        elif action == "install":
            if not args.query:
                out.error("usage: lai skills install <git-url|owner/repo|archive-url|path>")
                return 2
            from .skills.install import install  # noqa: PLC0415

            result = install(args.query, config.skills_dir, overwrite=args.overwrite)
            out.write(f"[green]installed:[/green] {', '.join(result.installed) or '(none)'}")
            if result.skipped:
                out.write(f"[yellow]skipped:[/yellow] {', '.join(result.skipped)}")
            out.write(f"[dim]→ {result.destination}[/dim]")
        elif action == "remove":
            from .skills.install import uninstall  # noqa: PLC0415

            removed = uninstall(args.query or "", config.skills_dir)
            out.write("[green]removed[/green]" if removed else "[yellow]not installed here[/yellow]")
        return 0
    except LaiError as exc:
        out.error(str(exc))
        return 1
    finally:
        runtime.close()


def cmd_sessions(args) -> int:
    out = Out(quiet=args.json)
    config = load_config()
    from .agent.session import Session  # noqa: PLC0415

    if args.id:
        session = Session.load(Path(config.sessions_dir) / f"{args.id}.jsonl")
        if args.json:
            print(json.dumps(session.summary(), indent=2))
        else:
            out.write(json.dumps(session.summary(), indent=2))
            for message in session.messages:
                text = message.text.strip()
                if text:
                    out.write(f"\n[bold]{message.role}[/bold]: {text[:600]}")
                for call in message.tool_calls:
                    out.write(f"  [cyan]▸ {call.name}[/cyan] [dim]{json.dumps(call.input)[:120]}[/dim]")
        return 0

    sessions = Session.list_sessions(config.sessions_dir, limit=args.limit)
    if args.json:
        print(json.dumps(sessions, indent=2))
        return 0
    for entry in sessions:
        import time

        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["modified"]))
        out.write(f"  [dim]{when}[/dim] [cyan]{entry['id']}[/cyan] {entry['task'][:70]}")
    out.write(f"\n[dim]{len(sessions)} session(s) in {config.sessions_dir}[/dim]")
    return 0


def cmd_serve(args) -> int:
    from .daemon.server import serve  # noqa: PLC0415

    config = _apply_overrides(load_config(), args)
    try:
        if args.channels:
            from dataclasses import replace as _replace  # noqa: PLC0415

            config = config.with_overrides(
                channels=_replace(
                    config.channels,
                    enabled=tuple(c.strip() for c in args.channels.split(",") if c.strip()),
                )
            )
        serve(
            config,
            host=args.host,
            port=args.port,
            allow_remote=args.allow_remote,
        )
    except LaiError as exc:
        Out().error(str(exc))
        return 1
    return 0


def cmd_tui(args) -> int:
    """Full-screen interactive interface."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        Out().error("the TUI needs a real terminal — in a script use `lai do` or `lai repl`")
        return 2
    config = _apply_overrides(load_config(), args)
    runtime = build_runtime(config, with_mcp=not args.no_mcp)
    try:
        from .tui import run_tui  # noqa: PLC0415
    except ImportError as exc:
        Out().error(f"the TUI needs textual: pip install textual ({exc})")
        runtime.close()
        return 2
    try:
        run_tui(runtime, task=args.task or "")
    finally:
        runtime.close()
    return 0


def cmd_channels(args) -> int:
    """Manage remote connectors (Telegram, webhooks) and who may use them."""
    out = Out(quiet=getattr(args, "json", False))
    config = load_config()
    from .channels import REGISTRY, AccessPolicy, build_channels  # noqa: PLC0415

    policy = AccessPolicy(config.channels_file, open_access=config.channels.open_access)
    action = args.action or "status"

    if action == "status":
        built, problems = build_channels(config)
        if args.json:
            print(json.dumps({
                "enabled": list(config.channels.enabled),
                "ready": [c.name for c in built],
                "problems": problems,
                "access": policy.summary(),
            }, indent=2, default=str))
            return 0
        out.write("[bold]Available connectors[/bold]")
        for name, description in sorted(REGISTRY.items()):
            mark = "[green]on[/green] " if name in config.channels.enabled else "[dim]off[/dim]"
            out.write(f"  {mark} [cyan]{name:9}[/cyan] [dim]{description}[/dim]")
        if problems:
            out.write("\n[yellow]Not usable:[/yellow]")
            for name, reason in problems.items():
                out.write(f"  [red]{name}[/red]: {reason}")
        out.write(f"\n[bold]Authorised senders[/bold] [dim]({config.channels_file})[/dim]")
        people = policy.principals()
        for principal in people:
            role = " [magenta](admin)[/magenta]" if principal.admin else ""
            out.write(f"  {principal.channel}:{principal.sender} {principal.name}{role}")
        if not people:
            out.write("  [dim]nobody yet — run `lai channels pair` while the daemon is up[/dim]")
        if policy.open_access:
            out.write("\n[red]open_access is ON — anyone who finds the bot can drive this desktop[/red]")
        return 0

    if action == "pair":
        code = policy.new_pairing_code()
        out.write("[bold]Pairing code:[/bold] [green]" + code + "[/green]")
        out.write("[dim]Send this to the bot within 15 minutes:[/dim]")
        out.write(f"  /pair {code}")
        out.write("[dim]The first person to pair becomes the admin.[/dim]")
        out.write("\n[yellow]Note:[/yellow] the running daemon holds its own policy in memory — "
                  "use its /pair flow, or restart it after pairing here.")
        return 0

    if action == "allow":
        if not args.target or ":" not in args.target:
            out.error("usage: lai channels allow <channel>:<sender-id>")
            return 2
        channel, _, sender = args.target.partition(":")
        principal = policy.allow(channel, sender, admin=args.admin)
        out.write(f"[green]allowed[/green] {principal.key}" + (" (admin)" if principal.admin else ""))
        return 0

    if action == "revoke":
        if not args.target or ":" not in args.target:
            out.error("usage: lai channels revoke <channel>:<sender-id>")
            return 2
        channel, _, sender = args.target.partition(":")
        out.write("[green]revoked[/green]" if policy.revoke(channel, sender) else "[yellow]not listed[/yellow]")
        return 0

    if action == "test":
        name = args.target or "telegram"
        if name == "telegram":
            from .channels import TelegramChannel  # noqa: PLC0415

            channel = TelegramChannel(config.channels.telegram_token)
            hint = "open https://t.me/{username} and send /pair <code>"
        elif name == "discord":
            from .channels import DiscordChannel  # noqa: PLC0415

            channel = DiscordChannel(config.channels.discord_token)
            hint = "invite the bot to a server, then DM it /pair <code>"
        else:
            out.error("only 'telegram' and 'discord' can be tested this way")
            return 2
        if not channel.available:
            out.error(f"no {name} token configured (LAI_{name.upper()}_TOKEN or channels.{name}.token)")
            return 1
        try:
            me = channel.verify()
        except Exception as exc:
            out.error(f"token rejected: {exc}")
            return 1
        finally:
            channel.stop()
        username = me.get("username", "?")
        out.write(f"[green]token OK[/green] — bot @{username}")
        out.write("[dim]" + hint.format(username=username) + "[/dim]")
        return 0

    out.error(f"unknown action {action}")
    return 2


def cmd_schedule(args) -> int:
    """Inspect and edit recurring tasks. The daemon is what actually runs them."""
    out = Out(quiet=getattr(args, "json", False))
    config = load_config()
    from .scheduler import TaskStore, describe_schedule, make_task  # noqa: PLC0415

    store = TaskStore(config.schedule_file)
    action = args.action or "list"

    if action == "list":
        tasks = store.list()
        if args.json:
            print(json.dumps([t.to_dict() for t in tasks], indent=2))
            return 0
        if not tasks:
            out.write("[dim]No scheduled tasks. Add one with:[/dim]")
            out.write('  lai schedule add nightly "@daily" "summarise today\'s notes"')
            return 0
        for task in tasks:
            state = "[green]on[/green] " if task.enabled else "[dim]off[/dim]"
            when = _format_time(task.next_run) if task.enabled else "—"
            out.write(
                f"  {state} [cyan]{task.id}[/cyan] [bold]{task.name}[/bold] "
                f"[dim]{describe_schedule(task.schedule)}[/dim]"
            )
            out.write(f"       {task.task[:70]}")
            out.write(
                f"       [dim]next {when} · {task.runs} run(s), {task.failures} failure(s)[/dim]"
            )
        out.write(f"\n[dim]{len(tasks)} task(s) in {config.schedule_file}[/dim]")
        out.write("[dim]They run only while `lai serve` is up.[/dim]")
        return 0

    if action == "add":
        if not args.name or not args.schedule or not args.task:
            out.error('usage: lai schedule add <name> <schedule> "<task>"')
            return 2
        try:
            task = store.add(
                make_task(name=args.name, task=args.task, schedule=args.schedule, mode=args.mode or "auto")
            )
        except ValueError as exc:
            out.error(str(exc))
            return 2
        out.write(f"[green]added[/green] [cyan]{task.id}[/cyan] {task.name}")
        out.write(f"  [dim]{describe_schedule(task.schedule)} · next {_format_time(task.next_run)}[/dim]")
        return 0

    if action in ("remove", "rm"):
        if not args.name:
            out.error("usage: lai schedule remove <id>")
            return 2
        removed = store.remove(args.name)
        out.write("[green]removed[/green]" if removed else "[yellow]no such task[/yellow]")
        return 0 if removed else 1

    if action in ("enable", "disable"):
        if not args.name:
            out.error(f"usage: lai schedule {action} <id>")
            return 2
        task = store.get(args.name)
        if task is None:
            out.error(f"no task with id {args.name}")
            return 1
        from dataclasses import replace as _replace  # noqa: PLC0415

        store.update(_replace(task, enabled=action == "enable"))
        out.write(f"[green]{action}d[/green] {task.name}")
        return 0

    if action == "run":
        if not args.name:
            out.error("usage: lai schedule run <id>")
            return 2
        task = store.get(args.name)
        if task is None:
            out.error(f"no task with id {args.name}")
            return 1
        out.write(f"[dim]running {task.name} now (ignoring its schedule)[/dim]")
        runtime = build_runtime(config.with_overrides(
            safety=replace(config.safety, mode=task.mode or config.safety.mode)
        ))
        try:
            if runtime.provider is None:
                return _no_provider(out, runtime)
            agent = runtime.agent(on_event=_make_reporter(out, verbose=args.verbose))
            result = agent.run(task.task)
            out.write(result.render())
            return 0 if result.ok else 1
        finally:
            runtime.close()

    out.error(f"unknown action {action}")
    return 2


def _format_time(stamp: float | None) -> str:
    if not stamp:
        return "—"
    import time  # noqa: PLC0415

    return time.strftime("%a %d %b %H:%M", time.localtime(stamp))


def cmd_models(args) -> int:
    """List, test and choose a model backend."""
    out = Out(quiet=getattr(args, "json", False))
    from . import models as backends  # noqa: PLC0415

    action = args.action or "list"

    if action == "test":
        if not args.name:
            out.error("usage: lai models test <name>")
            return 2
        out.write(f"[dim]asking {args.name} for one word…[/dim]")
        works, detail = backends.check(args.name)
        out.write(f"[green]✓ works[/green] [dim]{detail}[/dim]" if works else f"[red]✗ {detail}[/red]")
        return 0 if works else 1

    if action == "use":
        if not args.name:
            out.error("usage: lai models use <name>")
            return 2
        works, detail = backends.check(args.name)
        if not works and not args.force:
            out.error(f"{args.name} does not work yet: {detail}")
            out.write("[dim]pass --force to save it anyway[/dim]")
            return 1
        from . import config_file  # noqa: PLC0415

        config = load_config()
        settings = config_file.merge(
            config_file.read(config.home), {"provider": {"name": args.name}}
        )
        path = config_file.write(config.home, settings)
        out.write(f"[green]✓[/green] default backend is now [bold]{args.name}[/bold] [dim]({path})[/dim]")
        if works:
            out.write(f"  [dim]{detail}[/dim]")
        return 0

    found = backends.discover(probe_local=not args.no_probe)
    if args.json:
        print(json.dumps([b.to_dict() for b in found], indent=2))
        return 0

    groups = [
        (backends.READY, "Ready now", "green"),
        (backends.NEEDS_AUTH, "Installed, needs a sign-in", "yellow"),
        (backends.KNOWN, "Known — add a key to use", "dim"),
    ]
    kinds = {backends.KIND_API: "api", backends.KIND_LOCAL: "local", backends.KIND_CLI: "cli"}

    for status, title, style in groups:
        rows = [b for b in found if b.status == status]
        if not rows:
            continue
        if status == backends.KNOWN and not args.all:
            out.write(f"\n[dim]…and {len(rows)} more — `lai models --all` to see them[/dim]")
            continue
        out.write(f"\n[bold {style}]{title}[/bold {style}]")
        for backend in rows:
            vision = "" if backend.vision else " [dim](no vision)[/dim]"
            out.write(
                f"  [cyan]{backend.name:<14}[/cyan] [dim]{kinds.get(backend.kind, ''):<5}[/dim] "
                f"{backend.model:<28}{vision}"
            )
            if backend.detail:
                out.write(f"                 [dim]{backend.detail}[/dim]")
            if backend.hint and status != backends.READY:
                out.write(f"                 [dim]→ {backend.hint}[/dim]")

    ready = [b for b in found if b.usable]
    out.write("")
    out.write(f"[dim]{len(ready)} usable now, {len(found)} known in total[/dim]")
    out.write("[dim]lai models test <name>   prove one works[/dim]")
    out.write("[dim]lai models use <name>    make it the default[/dim]")
    return 0


def cmd_mcp(args) -> int:
    """Serve LAI's desktop tools over MCP stdio. Nothing may touch stdout here."""
    try:
        from .mcp.server import run_stdio  # noqa: PLC0415
    except ImportError as exc:
        sys.stderr.write(f"MCP support unavailable: {exc}\npip install mcp\n")
        return 2
    config = _apply_overrides(load_config(), args)
    run_stdio(config)
    return 0


# -- parser --------------------------------------------------------------


def _help(name: str) -> str:
    """The help line for a command, from the same table --help renders."""
    return COMMANDS.get(name, ("", name))[1]


def build_parser() -> argparse.ArgumentParser:
    from . import __version__  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="lai",
        description=BANNER + " — drives real applications on your Linux desktop.",
        epilog="New here? Run `lai setup`. Something broken? Run `lai doctor`.",
    )
    parser.add_argument("--version", action="version", version=f"lai {__version__}")
    sub = parser.add_subparsers(dest="command")

    def add_agent_flags(p):
        p.add_argument("--mode", choices=PERMISSION_MODES, help="Permission mode for this run")
        p.add_argument("--model", help="Override the model")
        p.add_argument("--provider", help="anthropic | zai | openai | openrouter | ollama | auto")
        p.add_argument("--steps", type=int, help="Maximum agent steps")
        p.add_argument("--timeout", type=float, help="Wall-clock budget in seconds")
        p.add_argument("--thinking", type=int, help="Extended thinking budget in tokens")
        p.add_argument("--dry-run", action="store_true", help="Block all side effects")
        p.add_argument("--no-mcp", action="store_true", help="Skip external MCP servers")
        p.add_argument("--verbose", "-v", action="store_true")

    p_do = sub.add_parser(
        "do",
        help=_help("do"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              lai do "open firefox and check today's weather"
              lai do "open libreoffice and write a haiku about monday" --mode ask
              lai do "count the open windows" --provider cli:claude --no-mcp
        """),
    )
    p_do.add_argument("task", help="What you want done")
    p_do.add_argument("--json", action="store_true", help="Machine-readable result only")
    p_do.add_argument("--no-stream", action="store_true")
    add_agent_flags(p_do)
    p_do.set_defaults(func=cmd_do)

    p_repl = sub.add_parser("repl", help=_help("repl"))
    add_agent_flags(p_repl)
    p_repl.set_defaults(func=cmd_repl)

    p_tui = sub.add_parser("tui", help=_help("tui"))
    p_tui.add_argument("task", nargs="?", default="", help="Optional task to start with")
    add_agent_flags(p_tui)
    p_tui.set_defaults(func=cmd_tui)

    p_doctor = sub.add_parser(
        "doctor", help=_help("doctor"), epilog="`lai doctor --fix` applies every fix that needs no human."
    )
    p_doctor.add_argument("--fix", action="store_true", help="Apply every automatic fix")
    p_doctor.add_argument("--json", action="store_true")
    p_doctor.add_argument("--mcp", action="store_true", help="Also connect MCP servers (slow)")
    p_doctor.add_argument("--no-mcp", action="store_true", help=argparse.SUPPRESS)
    p_doctor.set_defaults(func=cmd_doctor)

    p_setup = sub.add_parser(
        "setup",
        help=_help("setup"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Detects every backend that could work on this machine — installed coding
            CLIs (claude, codex, gemini…), API keys in the environment, local servers
            (ollama, lm-studio…) — and verifies the one you pick with a real request
            before saving anything.
        """),
    )
    p_setup.add_argument("--yes", "-y", action="store_true", help="Accept every default, no questions")
    p_setup.add_argument("--no-demo", action="store_true", help="Skip the first-run demo task")
    p_setup.set_defaults(func=cmd_setup)

    p_observe = sub.add_parser("observe", help=_help("observe"))
    p_observe.add_argument("--scope", choices=["focused", "desktop"], default="focused")
    p_observe.add_argument("--screenshot", action="store_true")
    p_observe.add_argument("--annotate", action="store_true")
    p_observe.add_argument("--save", help="Write the screenshot to this path")
    p_observe.add_argument("--limit", type=int, default=80)
    p_observe.add_argument("--json", action="store_true")
    p_observe.set_defaults(func=cmd_observe)

    p_tools = sub.add_parser("tools", help=_help("tools"))
    p_tools.add_argument("filter", nargs="?", default="")
    p_tools.add_argument("--json", action="store_true", help="Emit the model-facing schemas")
    p_tools.add_argument("--no-mcp", action="store_true")
    p_tools.set_defaults(func=cmd_tools)

    p_skills = sub.add_parser("skills", help=_help("skills"))
    p_skills.add_argument("action", nargs="?", choices=["list", "show", "install", "remove"], default="list")
    p_skills.add_argument("query", nargs="?", default="")
    p_skills.add_argument("--overwrite", action="store_true")
    p_skills.set_defaults(func=cmd_skills)

    p_sessions = sub.add_parser("sessions", help=_help("sessions"))
    p_sessions.add_argument("id", nargs="?", default="")
    p_sessions.add_argument("--limit", type=int, default=20)
    p_sessions.add_argument("--json", action="store_true")
    p_sessions.set_defaults(func=cmd_sessions)

    p_serve = sub.add_parser(
        "serve",
        help=_help("serve"),
        epilog="The daemon owns the desktop gate: one agent at a time, whichever way it was reached.",
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.add_argument("--allow-remote", action="store_true", help="Permit binding a non-loopback address")
    p_serve.add_argument("--channels", default="", help="Comma-separated connectors to start, e.g. telegram,webhook")
    add_agent_flags(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_channels = sub.add_parser("channels", help=_help("channels"))
    p_channels.add_argument(
        "action", nargs="?", choices=["status", "pair", "allow", "revoke", "test"], default="status"
    )
    p_channels.add_argument("target", nargs="?", default="")
    p_channels.add_argument("--admin", action="store_true", help="Grant admin rights when allowing")
    p_channels.add_argument("--json", action="store_true")
    p_channels.set_defaults(func=cmd_channels)

    p_schedule = sub.add_parser("schedule", help=_help("schedule"))
    p_schedule.add_argument(
        "action",
        nargs="?",
        choices=["list", "add", "remove", "rm", "enable", "disable", "run"],
        default="list",
    )
    p_schedule.add_argument("name", nargs="?", default="", help="Task name (add) or id (everything else)")
    p_schedule.add_argument("schedule", nargs="?", default="", help="Cron, @daily, or every:<seconds>")
    p_schedule.add_argument("task", nargs="?", default="", help="The instruction to run")
    p_schedule.add_argument("--mode", choices=PERMISSION_MODES, default="", help="Permission mode when it fires")
    p_schedule.add_argument("--verbose", action="store_true")
    p_schedule.add_argument("--json", action="store_true")
    p_schedule.set_defaults(func=cmd_schedule)

    p_models = sub.add_parser(
        "models",
        help=_help("models"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              lai models                 what works on this machine, right now
              lai models use zai         verify and save a backend as the default
              lai models use cli:claude  think with the installed claude CLI — no API key
        """),
    )
    p_models.add_argument("action", nargs="?", choices=["list", "test", "use"], default="list")
    p_models.add_argument("name", nargs="?", default="", help="Backend name, e.g. groq or cli:claude")
    p_models.add_argument("--all", action="store_true", help="Include every known vendor")
    p_models.add_argument("--no-probe", action="store_true", help="Skip probing local servers")
    p_models.add_argument("--force", action="store_true", help="`use` even if the check fails")
    p_models.add_argument("--json", action="store_true")
    p_models.set_defaults(func=cmd_models)

    p_mcp = sub.add_parser("mcp", help=_help("mcp"))
    add_agent_flags(p_mcp)
    p_mcp.set_defaults(func=cmd_mcp)

    return parser


def _print_overview() -> None:
    """The top-level help: what LAI is, what to run first, and where everything lives.

    Rendered by hand rather than by argparse because a wall of fourteen
    same-weight subcommands answers neither question a new user has ("how do I
    start?" and "where is X?"). Rich when available, plain text otherwise —
    identical wording either way.
    """
    from . import __version__  # noqa: PLC0415

    console = None
    try:
        from rich.console import Console  # noqa: PLC0415

        console = Console(highlight=False, soft_wrap=True)
    except ImportError:
        console = None

    width = shutil.get_terminal_size((80, 24)).columns
    pad = max(len(name) for name in COMMANDS) + 2

    def emit(text: str = "", *, style: str = "") -> None:
        if console is not None:
            console.print(text, style=style or None)
        else:
            print(_strip_markup(text))

    emit(f"LAI [dim]v{__version__}[/dim] — the agent for your Linux desktop", style="bold")
    emit()
    emit("It runs real applications: reads the screen, clicks, types, drags —")
    emit("like the Claude Chrome extension, but for the whole OS.")
    emit()
    for group in COMMAND_GROUPS:
        names = [name for name, (g, _) in COMMANDS.items() if g == group]
        if not names:
            continue
        emit(f"{group}  [dim]{COMMAND_GROUPS[group]}[/dim]", style="bold cyan")
        for name in names:
            emit(f"  {name.ljust(pad)}[dim]{COMMANDS[name][1]}[/dim]")
        emit()
    emit("examples", style="bold cyan")
    for command, what in EXAMPLES:
        emit(f"  {command}", style="cyan")
        emit(f"  {' ' * 2}{what}", style="dim")
    emit()
    emit(f"{'─' * min(width, 72)}", style="dim")
    emit("[dim]`lai <command> --help` for options · first time? `lai setup` · broken? `lai doctor`[/dim]")


def _default_command(argv: list[str]) -> list[str]:
    """What bare `lai` should do.

    Someone typing `lai` for the first time wants to be told how to start, not
    dropped at a prompt that will fail on the first request because there is no
    API key. So: never set up → the wizard; set up → the full interface, or the
    plain REPL if textual is not installed.
    """
    try:
        from .setup_wizard import needs_setup  # noqa: PLC0415

        if needs_setup():
            return ["setup", *argv]
    except Exception:
        pass

    try:
        import textual  # noqa: F401, PLC0415

        # A full-screen app needs a real terminal; piped output gets the REPL,
        # which degrades to reading EOF and exiting rather than hanging.
        if sys.stdout.isatty():
            return ["tui", *argv]
    except ImportError:
        pass
    return ["repl", *argv]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[0:1] in (["--help"], ["-h"], ["help"]):
        if len(argv) == 1:
            _print_overview()
            return 0
        # `lai help do` → the same help as `lai do --help`
        argv = [argv[1], "--help"]
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse signals --help, --version and usage errors by exiting; main()
        # is also called in-process (tests, the daemon), so return the code.
        return exc.code if isinstance(exc.code, int) else 1
    if not getattr(args, "command", None):
        args = parser.parse_args(_default_command(list(argv or [])))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130
    except LaiError as exc:
        sys.stderr.write(f"error: {exc}\n")
        _hint_for(exc)
        return 1


def _hint_for(exc: LaiError) -> None:
    """Turn a failure into a next step. An error the user cannot act on is a bug."""
    text = str(exc).lower()
    if "provider" in text or "api key" in text or "backend" in text:
        sys.stderr.write("\nhint: run `lai setup` to add a model backend.\n")
    elif "display" in text or "x11" in text:
        sys.stderr.write("\nhint: run `lai doctor` — LAI needs a graphical X11 session.\n")
    elif "xdotool" in text:
        sys.stderr.write("\nhint: run `lai doctor --fix` to install what is missing.\n")



if __name__ == "__main__":
    raise SystemExit(main())
