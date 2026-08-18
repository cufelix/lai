"""The chat interface — what you get when you type ``lai``.

A conversation, not a command line: you say what you want, watch the agent do
it to the actual desktop, and say the next thing. Slash commands cover
everything you would otherwise have to leave for — switching model, changing
how much it may do unattended, checking the machine.

Uses prompt_toolkit when it is installed (history, completion, a live status
line) and plain ``input()`` when it is not, because a missing optional
dependency must never be the difference between a usable tool and none.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..agent.session import Session
from ..errors import Interrupted, LaiError
from ..osl.lock import DesktopBusy
from . import backends
from .commands import COMMANDS, NEW, QUIT, Context
from .commands import run as run_command

BANNER = "LAI"
# A single-glyph prompt keeps pasted multi-line tasks readable.
PROMPT = "\n\x1b[1;36m\u203a\x1b[0m "


class Reader:
    """Line input. prompt_toolkit if available, stdin otherwise."""

    def __init__(self, history_file: Path | None = None) -> None:
        self.session = None
        # prompt_toolkit needs a real terminal; over a pipe it warns and then
        # misbehaves, so piped input goes straight to stdin. That path is what
        # makes `echo "/status" | lai chat` and CI usable at all.
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return
        try:
            from prompt_toolkit import PromptSession  # noqa: PLC0415
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory  # noqa: PLC0415
            from prompt_toolkit.completion import WordCompleter  # noqa: PLC0415
            from prompt_toolkit.history import FileHistory, InMemoryHistory  # noqa: PLC0415
        except ImportError:
            return

        history = InMemoryHistory()
        if history_file is not None:
            try:
                history_file.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(history_file))
            except OSError:
                pass
        self.session = PromptSession(
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(
                [f"/{name}" for name, spec in COMMANDS.items() if not spec[2]],
                sentence=True,
            ),
            complete_while_typing=True,
        )

    def read(self, prompt: str, *, bottom: str = "") -> str:
        if self.session is None:
            # Without prompt_toolkit the prompt is echoed literally, so the
            # colour codes would be printed rather than rendered.
            return input(_plain(prompt))
        from prompt_toolkit.formatted_text import ANSI  # noqa: PLC0415

        return self.session.prompt(
            ANSI(prompt), bottom_toolbar=(lambda: bottom) if bottom else None
        )


def _plain(text: str) -> str:
    import re  # noqa: PLC0415

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def status_line(runtime) -> str:
    info = backends.describe(runtime)
    mode = runtime.config.safety.mode
    text = f" {info['name']}/{info['model']} · {mode} "
    standbys = info["chain"][1:]
    if standbys:
        text += f"· ↻ {standbys[0]}" + (f" +{len(standbys) - 1}" if len(standbys) > 1 else "") + " "
    return text


def run_chat(runtime, *, out=None, task: str = "", verbose: bool = False) -> int:
    """The read → run → render loop. Returns a process exit code."""
    from ..cli import Out, _interactive_approver, _make_reporter  # noqa: PLC0415

    out = out or Out()
    if runtime.provider is None:
        from ..cli import _no_provider  # noqa: PLC0415

        return _no_provider(out, runtime)

    reader = Reader(Path(runtime.config.home) / "history")
    session = Session()
    runtime.extra["chat_session"] = session
    approver = _interactive_approver(out)
    reporter = _make_reporter(out, verbose=verbose)
    ctx = Context(runtime=runtime, ask=_asker(out, reader))

    _welcome(out, runtime)

    pending = task.strip()
    while True:
        if pending:
            line, pending = pending, ""
        else:
            try:
                line = reader.read(PROMPT, bottom=status_line(runtime)).strip()
            except EOFError:
                out.write("[dim]bye[/dim]")
                return 0
            except KeyboardInterrupt:
                out.write("[dim](ctrl+d to quit)[/dim]")
                continue

        if not line:
            continue

        if line.startswith("/"):
            answer = run_command(ctx, line)
            if answer == QUIT:
                out.write("[dim]bye[/dim]")
                return 0
            if answer == NEW:
                session = Session()
                runtime.extra["chat_session"] = session
                out.write("[dim]new session — the agent has forgotten what came before[/dim]")
                continue
            if answer:
                out.write(answer)
            continue

        agent = runtime.agent(session=session, approver=approver, on_event=reporter)
        try:
            result = agent.run(line)
            out.write(result.render())
        except KeyboardInterrupt:
            agent.interrupt()
            out.write("\n[yellow]interrupted[/yellow]")
        except Interrupted:
            out.write("\n[yellow]interrupted[/yellow]")
        except DesktopBusy as exc:
            # Somebody else is driving. That ends this task, not the session.
            out.write(f"[yellow]⏸ {exc}[/yellow]")
            out.write("[dim]one desktop, one agent — try again when it finishes[/dim]")
        except LaiError as exc:
            out.error(str(exc))
            if "provider" in str(exc).lower() or "backend" in str(exc).lower():
                out.write("[dim]try [bold]/model[/bold] to switch backend[/dim]")


def _asker(out, reader: Reader):
    """A numbered menu. Returns the chosen index, or -1 when cancelled."""

    def ask(question: str, options: list[str]) -> int:
        out.write(f"\n[bold]{question}[/bold]")
        for index, option in enumerate(options, 1):
            out.write(f"  [cyan]{index}[/cyan]  {option}")
        try:
            answer = reader.read("  choose [1] ").strip()
        except (EOFError, KeyboardInterrupt):
            return -1
        if not answer:
            return 0
        if not answer.isdigit() or not 1 <= int(answer) <= len(options):
            out.write("[dim]cancelled[/dim]")
            return -1
        return int(answer) - 1

    return ask


def _welcome(out, runtime) -> None:
    from .. import __version__  # noqa: PLC0415

    info = backends.describe(runtime)
    out.write(f"[bold]{BANNER}[/bold] [dim]v{__version__} — your desktop, driven[/dim]")
    out.write(
        f"[dim]{info['name']}/{info['model']} · mode {runtime.config.safety.mode} · "
        f"{len(runtime.registry)} tools · {len(runtime.skills)} skills[/dim]"
    )
    standbys = info["chain"][1:]
    if standbys:
        out.write(f"[dim]failover ready: {' → '.join(standbys[:3])}[/dim]")
    out.write("")
    out.write("[dim]Say what you want done. [bold]/help[/bold] for commands.[/dim]")
    if not sys.stdin.isatty():
        out.write("[yellow]stdin is not a terminal — approvals will be refused[/yellow]")
