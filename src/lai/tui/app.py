"""Full-screen terminal interface.

A desktop agent is hard to supervise from scrolling log lines: you want to see
what it is doing to your machine *while* it does it, and you want the permission
prompt to arrive somewhere you are already looking.

So: a live activity feed on the left, and on the right the three things that
answer "should I let this continue?" — the plan, what is actually focused on the
desktop right now, and what the run has cost so far.

The agent loop is synchronous and blocking, so it runs on a worker thread and
every UI update crosses back through ``call_from_thread``. Approvals invert
that: the worker blocks on an event while the UI thread shows the prompt.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, OptionList, RichLog, Static, TextArea

from ..agent.session import Session
from ..config import PERMISSION_MODES
from .palette import Palette, line_prefix, search

REFRESH_INTERVAL = 2.0


@dataclass(slots=True)
class PendingAnswer:
    """A question the UI must answer while a worker thread waits for it.

    The same inversion the approval prompt uses: the worker blocks on an event
    while the interface shows a dialog. Slash commands need it too — choosing a
    backend or pasting a key are questions, and they arrive from a thread.
    """

    question: str
    options: tuple = ()
    kind: str = "choice"
    """choice | confirm | secret"""
    event: threading.Event = field(default_factory=threading.Event)
    value: object = None


class ChoiceScreen(ModalScreen[int]):
    """Pick one of a numbered list — the same shape the chat menu has."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, pending: PendingAnswer) -> None:
        super().__init__()
        self.pending = pending

    def compose(self) -> ComposeResult:
        with Vertical(id="choice-box"):
            yield Label(self.pending.question, id="choice-title")
            yield OptionList(*[str(option) for option in self.pending.options], id="choice-list")
            yield Label("enter to choose · esc to cancel", classes="panel-title")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    @on(OptionList.OptionSelected)
    def _chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_index)

    def action_cancel(self) -> None:
        self.dismiss(-1)


class AskScreen(ModalScreen[str]):
    """A yes/no question, or something typed that must not be echoed."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, pending: PendingAnswer) -> None:
        super().__init__()
        self.pending = pending

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-box"):
            yield Label(self.pending.question, id="ask-title")
            if self.pending.kind == "secret":
                yield Input(password=True, id="ask-input", placeholder="paste here — not echoed")
            else:
                with Horizontal(id="ask-buttons"):
                    yield Button("Yes  (y)", variant="success", id="yes")
                    yield Button("No  (n)", variant="error", id="no")

    def on_mount(self) -> None:
        if self.pending.kind == "secret":
            self.query_one(Input).focus()

    @on(Input.Submitted)
    def _typed(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss("yes")

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")


@dataclass(slots=True)
class PendingApproval:
    tool: str
    arguments: dict
    reason: str
    event: threading.Event = field(default_factory=threading.Event)
    granted: bool = False


class ApprovalScreen(ModalScreen[bool]):
    """Blocking permission prompt."""

    BINDINGS = [
        Binding("y", "approve", "Allow"),
        Binding("n", "refuse", "Refuse"),
        Binding("escape", "refuse", "Refuse"),
    ]

    def __init__(self, pending: PendingApproval) -> None:
        super().__init__()
        self.pending = pending

    def compose(self) -> ComposeResult:
        preview = ", ".join(f"{k}={v!r}" for k, v in list(self.pending.arguments.items())[:6])
        with Vertical(id="approval-box"):
            yield Label("Approval needed", id="approval-title")
            yield Label(self.pending.tool, id="approval-tool")
            yield Static(preview[:400] or "(no arguments)", id="approval-args")
            yield Label(self.pending.reason, id="approval-reason")
            with Horizontal(id="approval-buttons"):
                yield Button("Allow  (y)", variant="success", id="allow")
                yield Button("Refuse (n)", variant="error", id="refuse")

    @on(Button.Pressed, "#allow")
    def _allow(self) -> None:
        self.action_approve()

    @on(Button.Pressed, "#refuse")
    def _refuse(self) -> None:
        self.action_refuse()

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_refuse(self) -> None:
        self.dismiss(False)


class Composer(TextArea):
    """Where you type. Multiline, because tasks are often a paragraph.

    Enter sends and Shift+Enter breaks the line — the opposite of a text
    editor's default, and the right way round for something you talk to.

    While the command palette is open the arrow keys and Enter belong to it:
    you are choosing a command, not writing one.
    """

    BINDINGS = [Binding("escape", "app.focus_feed", "Back", show=False)]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class Completed(Message):
        """A command was picked out of the palette."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    def _palette(self):
        try:
            palette = self.app.query_one(Palette)
        except Exception:
            return None
        return palette if palette.open else None

    async def _on_key(self, event) -> None:
        palette = self._palette()
        if palette is not None:
            if event.key in ("down", "up"):
                event.prevent_default()
                event.stop()
                palette.action_cursor_down() if event.key == "down" else palette.action_cursor_up()
                return
            if event.key in ("enter", "tab"):
                event.prevent_default()
                event.stop()
                name = palette.chosen()
                if name:
                    self.post_message(self.Completed(name))
                return
            if event.key == "escape":
                event.prevent_default()
                event.stop()
                palette.close()
                return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
            return
        if event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)


class StatusBar(Static):
    """Top line: who is answering, under what permissions, and at what cost."""

    provider = reactive("")
    mode = reactive("ask")
    steps = reactive(0)
    of_steps = reactive(0)
    tokens = reactive(0)
    context = reactive(0)
    context_budget = reactive(0)
    elapsed = reactive(0.0)
    busy = reactive(False)

    def render(self) -> Text:
        text = Text()
        text.append(" LAI ", style="bold white on dark_blue")
        text.append(f" {self.provider} ", style="cyan")
        mode_style = {
            "readonly": "green", "ask": "yellow", "auto": "dark_orange", "yolo": "bold red"
        }.get(self.mode, "white")
        text.append(f" {self.mode} ", style=mode_style)
        if self.busy:
            of = f"/{self.of_steps}" if self.of_steps else ""
            text.append(f" step {self.steps}{of} ", style="bold")
            text.append(f" {self.elapsed:.0f}s ", style="dim")
        else:
            text.append(" idle ", style="dim")
        text.append(f" {self.tokens:,} tok ", style="dim")
        if self.context_budget:
            # How full the transcript is. Amber from three-quarters, because
            # that is where compaction starts costing a round trip.
            share = min(self.context / self.context_budget, 1.0)
            style = "red" if share >= 0.95 else "yellow" if share >= 0.75 else "dim"
            text.append(f" ctx {share * 100:.0f}% ", style=style)
        return text


class DesktopPanel(Static):
    """What the agent is actually looking at, refreshed on a timer."""

    def on_mount(self) -> None:
        self.update("(reading the desktop…)")
        self.set_interval(REFRESH_INTERVAL, self.refresh_desktop)
        self.refresh_desktop()

    def refresh_desktop(self) -> None:
        app: LaiApp = self.app  # type: ignore[assignment]
        desktop = app.runtime.desktop
        try:
            active = desktop.windows.active_window()
            windows = desktop.windows.list_windows()
        except Exception as exc:
            self.update(Text(f"desktop unavailable: {exc}", style="red"))
            return

        text = Text()
        if active:
            text.append("focused\n", style="bold")
            text.append(f"  {active.title[:44]}\n", style="white")
            text.append(f"  {active.wm_class}  {active.bounds.width}x{active.bounds.height}\n", style="dim")
        else:
            text.append("focused\n  (none)\n", style="dim")
        text.append(f"\nwindows ({len(windows)})\n", style="bold")
        for window in windows[:7]:
            marker = "→ " if window.active else "  "
            text.append(f"{marker}{(window.wm_class or '?')[:14]:14} {window.title[:24]}\n", style="dim")
        self.update(text)


class PlanPanel(Static):
    """The agent's own plan, as recorded by the plan_update tool."""

    def on_mount(self) -> None:
        self.update(Text("no plan yet", style="dim"))

    def show(self, steps: list[str], current: int) -> None:
        text = Text()
        for index, step in enumerate(steps):
            if index < current:
                text.append(f"  ✓ {step}\n", style="green dim")
            elif index == current:
                text.append(f"  → {step}\n", style="bold yellow")
            else:
                text.append(f"    {step}\n", style="dim")
        self.update(text or Text("no plan yet", style="dim"))


# Concrete starting points. A new user facing an empty prompt does not think
# "I could ask it to file my invoices" — they think "what does this even take?".
# These are picked to be safe, fast, and obviously about *this* desktop.
EXAMPLE_TASKS = [
    "what windows do I have open?",
    "open the calculator and work out 12 * 34",
    "open the text editor and write me a haiku about X11",
    "take a screenshot and tell me what you see",
]


class LaiApp(App):
    """The interactive face of LAI."""

    CSS = """
    Screen { layout: vertical; }
    #status { height: 1; dock: top; }
    #body { height: 1fr; }
    #feed { width: 2fr; border-right: solid $panel; padding: 0 1; }
    #side { width: 42; }
    #plan-box, #desktop-box { border: round $panel; padding: 0 1; }
    #plan-box { height: 1fr; }
    #desktop-box { height: 2fr; }
    .panel-title { color: $text-muted; text-style: bold; }
    #input-area { dock: bottom; height: auto; }
    #prompt { height: auto; min-height: 3; max-height: 12; border: tall $accent; }
    #choice-box, #ask-box {
        width: 78; height: auto; max-height: 24; padding: 1 2;
        background: $surface; border: thick $accent;
    }
    #choice-title, #ask-title { color: $accent; text-style: bold; margin-bottom: 1; }
    #choice-list { height: auto; max-height: 16; }
    #ask-buttons { height: auto; margin-top: 1; }
    #ask-buttons Button { margin-right: 2; }
    #approval-box {
        width: 70; height: auto; padding: 1 2;
        background: $surface; border: thick $warning;
    }
    #approval-title { color: $warning; text-style: bold; }
    #approval-tool { color: $accent; text-style: bold; }
    #approval-args { color: $text-muted; margin: 1 0; }
    #approval-buttons { height: auto; margin-top: 1; }
    #approval-buttons Button { margin-right: 2; }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+n", "new_session", "New session"),
        Binding("ctrl+r", "pick_session", "Resume"),
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("f2", "cycle_mode", "Cycle mode"),
        Binding("f5", "observe", "Observe"),
    ]

    def __init__(self, runtime, *, task: str = "") -> None:
        super().__init__()
        self.runtime = runtime
        self.session = Session()
        self.initial_task = task
        self.agent = None
        self.pending: PendingApproval | None = None
        self._run_started = 0.0
        self._streaming = False

    # -- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status")
        with Horizontal(id="body"):
            yield RichLog(id="feed", wrap=True, markup=True, highlight=False)
            with Vertical(id="side"):
                with VerticalScroll(id="plan-box"):
                    yield Label("PLAN", classes="panel-title")
                    yield PlanPanel(id="plan")
                with VerticalScroll(id="desktop-box"):
                    yield Label("DESKTOP", classes="panel-title")
                    yield DesktopPanel(id="desktop")
        with Vertical(id="input-area"):
            yield Palette(id="palette")
            yield Composer(id="prompt", soft_wrap=True, show_line_numbers=False)
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.session.bind(self.runtime.config.sessions_dir)
        except Exception:
            pass
        bar = self.query_one(StatusBar)
        provider = self.runtime.provider
        bar.provider = f"{provider.name}/{provider.model}" if provider else "no model"
        bar.mode = self.runtime.config.safety.mode

        self._welcome()
        self.query_one(Composer).focus()
        self.set_interval(0.5, self._tick)

        if self.initial_task:
            self.submit(self.initial_task)

    def _welcome(self) -> None:
        """The first screen. Says what this is, and what to type."""
        self.write("[bold]LAI[/bold] — native desktop agent")
        self.write(
            f"[dim]{len(self.runtime.registry)} tools · {len(self.runtime.skills)} skills · "
            f"mode {self.runtime.config.safety.mode}[/dim]"
        )

        if self.runtime.provider is None:
            self.write("")
            self.write(f"[red]No model backend:[/red] {self.runtime.provider_error}")
            self.write("[yellow]Quit (ctrl+q) and run [bold]lai setup[/bold] to add one.[/yellow]")
            return

        self.write("")
        self.write("[bold]Try asking for:[/bold]")
        for example in EXAMPLE_TASKS:
            self.write(f"  [cyan]{example}[/cyan]")
        self.write("")
        self.write(
            "[dim]Plain language is the interface — describe the outcome, not the clicks.[/dim]"
        )
        self.write(
            "[dim]ctrl+c interrupt · f2 permission mode · f5 look again · "
            "ctrl+n new session · /help[/dim]\n"
        )

    # -- feed ------------------------------------------------------------

    def write(self, renderable) -> None:
        self.query_one("#feed", RichLog).write(renderable)

    def _tick(self) -> None:
        bar = self.query_one(StatusBar)
        if bar.busy:
            bar.elapsed = time.monotonic() - self._run_started

    # -- running ---------------------------------------------------------

    @on(Composer.Submitted)
    def _on_submit(self, event: Composer.Submitted) -> None:
        self.query_one(Palette).close()
        self.submit(event.text)

    @on(Composer.Changed)
    def _on_typed(self) -> None:
        """Open the palette on `/`, narrow it as the name is typed, close after."""
        composer = self.query_one(Composer)
        palette = self.query_one(Palette)
        prefix = line_prefix(composer.text)
        if prefix is None:
            palette.close()
            return
        palette.show(search(prefix))

    @on(Composer.Completed)
    def _on_completed(self, event: Composer.Completed) -> None:
        """Finish the word and hand the line back — no command runs by itself."""
        composer = self.query_one(Composer)
        rest = composer.text.split("\n", 1)
        tail = f"\n{rest[1]}" if len(rest) > 1 else ""
        self.query_one(Palette).close()
        composer.text = f"/{event.name} {tail}".rstrip() + (" " if not tail else "")
        composer.move_cursor(composer.document.end)
        composer.focus()

    @on(Palette.OptionSelected)
    def _palette_clicked(self, event: Palette.OptionSelected) -> None:
        self.post_message(Composer.Completed(str(event.option.id or "")))

    def action_focus_feed(self) -> None:
        self.query_one("#feed", RichLog).focus()

    def submit(self, text: str) -> None:
        if self.query_one(StatusBar).busy:
            self.write("[yellow]Still working — ctrl+c to interrupt.[/yellow]")
            return
        if self.runtime.provider is None:
            self.write("[red]No model backend configured; run `lai doctor`.[/red]")
            return
        if text.startswith("/"):
            self._slash(text)
            return
        self.write(f"\n[bold cyan]▸ you[/bold cyan] {text}")
        self.run_task(text)

    @work(thread=True, exclusive=True)
    def run_task(self, task: str) -> None:
        """Blocking agent run, off the UI thread."""
        self._run_started = time.monotonic()
        self.call_from_thread(self._set_busy, True)
        try:
            agent = self.runtime.agent(
                session=self.session,
                approver=self._approver,
                on_event=lambda kind, payload: self.call_from_thread(self._event, kind, payload),
            )
            self.agent = agent
            result = agent.run(task)
            self.call_from_thread(self._finished, result)
        except Exception as exc:
            self.call_from_thread(self.write, f"[red]run failed: {type(exc).__name__}: {exc}[/red]")
        finally:
            self.agent = None
            self.call_from_thread(self._set_busy, False)

    def _set_busy(self, busy: bool) -> None:
        bar = self.query_one(StatusBar)
        bar.busy = busy
        if not busy:
            bar.elapsed = 0.0

    def _event(self, kind: str, payload: dict) -> None:
        bar = self.query_one(StatusBar)
        if kind == "step":
            bar.steps = payload.get("step", 0)
            bar.of_steps = int(payload.get("of") or 0)
            bar.context = int(payload.get("context") or 0)
            bar.context_budget = int(payload.get("context_budget") or 0)
        elif kind == "text":
            delta = payload.get("delta", "")
            if delta:
                self._streaming = True
                self.write(Text(delta, style="white", end=""))
        elif kind == "assistant" and not self._streaming:
            self.write(payload.get("text", ""))
        elif kind == "tool_call":
            name = payload.get("name", "")
            args = ", ".join(f"{k}={v!r}" for k, v in list(payload.get("input", {}).items())[:3])
            self.write(f"[cyan]▸ {name}[/cyan] [dim]{args[:110]}[/dim]")
            self._streaming = False
            if name == "plan_update":
                steps = payload.get("input", {}).get("steps", [])
                current = payload.get("input", {}).get("current", 0)
                self.query_one(PlanPanel).show([str(s) for s in steps], int(current))
        elif kind == "tool_result":
            ok = payload.get("ok")
            mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
            summary = (payload.get("summary") or "").strip().splitlines()
            head = summary[0][:120] if summary else ""
            extra = f" [dim](+{len(summary) - 1} lines)[/dim]" if len(summary) > 1 else ""
            image = " [magenta]+img[/magenta]" if payload.get("images") else ""
            self.write(f"  {mark} [dim]{head}[/dim]{extra}{image}")
        elif kind == "provider_switch":
            self.write(
                f"[yellow]↻ {payload['from']} stepped aside[/yellow] "
                f"[dim]({payload.get('reason') or 'unavailable'})[/dim]"
            )
            self.write(f"  [green]continuing on {payload['to']}/{payload.get('model', '')}[/green]")
            bar.provider = f"{payload['to']}/{payload.get('model', '')}"
        elif kind == "repeating":
            self.write(f"[yellow]↺ refused a repeated {payload.get('name')}[/yellow]")
        elif kind == "yielding":
            self.write("[yellow]⏸ you are using the machine — waiting[/yellow]")
        elif kind == "resumed":
            self.write(f"[dim]▶ carrying on after {payload.get('waited', 0):.0f}s[/dim]")
        elif kind == "learned":
            titles = payload.get("titles") or payload.get("notes") or []
            self.write("[magenta]✎ learned:[/magenta] " + "; ".join(str(t) for t in titles[:3]))
        elif kind == "compacting":
            self.write("[dim]… compacting context[/dim]")
        elif kind == "error":
            self.write(f"[red]! {payload.get('error', '')}[/red]")
        bar.tokens = self.session.usage.total

    def _finished(self, result) -> None:
        icon = {
            "completed": "[green]✓[/green]", "blocked": "[yellow]⊘[/yellow]",
            "budget_exceeded": "[yellow]⏱[/yellow]", "error": "[red]✗[/red]",
            "interrupted": "[yellow]■[/yellow]",
        }.get(result.status, "•")
        self.write(f"\n{icon} [bold]{result.status.replace('_', ' ')}[/bold] "
                   f"[dim]{result.steps} steps · {result.elapsed:.0f}s[/dim]")
        if result.summary:
            self.write(result.summary)
        if result.verification:
            self.write(f"[dim]verified: {result.verification}[/dim]")
        if result.artifacts:
            self.write(f"[dim]files: {', '.join(result.artifacts)}[/dim]")
        if result.error:
            self.write(f"[red]{result.error}[/red]")
        self.write("")

    # -- approvals -------------------------------------------------------

    def _approver(self, tool: str, arguments: dict, verdict) -> bool:
        """Called on the worker thread; blocks it until the UI answers."""
        pending = PendingApproval(tool=tool, arguments=dict(arguments), reason=verdict.reason)
        self.pending = pending
        self.call_from_thread(self._ask, pending)
        pending.event.wait(timeout=300)
        self.pending = None
        return pending.granted

    def _ask(self, pending: PendingApproval) -> None:
        def answered(granted: bool | None) -> None:
            pending.granted = bool(granted)
            pending.event.set()
            self.write(
                f"  [green]allowed[/green] {pending.tool}" if pending.granted
                else f"  [yellow]refused[/yellow] {pending.tool}"
            )

        self.push_screen(ApprovalScreen(pending), answered)

    # -- actions ---------------------------------------------------------

    def action_interrupt(self) -> None:
        agent = self.agent
        if agent is None:
            self.write("[dim]nothing running[/dim]")
            return
        agent.interrupt()
        self.write("[yellow]interrupting…[/yellow]")

    def action_new_session(self) -> None:
        self.session = Session()
        try:
            self.session.bind(self.runtime.config.sessions_dir)
        except Exception:
            pass
        self.query_one(StatusBar).tokens = 0
        self.query_one(PlanPanel).update(Text("no plan yet", style="dim"))
        self.write("[dim]— new session —[/dim]")

    def action_cycle_mode(self) -> None:
        from dataclasses import replace  # noqa: PLC0415

        current = self.runtime.config.safety.mode
        index = (PERMISSION_MODES.index(current) + 1) % len(PERMISSION_MODES)
        mode = PERMISSION_MODES[index]
        self.runtime.config = self.runtime.config.with_overrides(
            safety=replace(self.runtime.config.safety, mode=mode)
        )
        self.runtime.policy.config = self.runtime.config.safety
        self.query_one(StatusBar).mode = mode
        self.write(f"[dim]permission mode → {mode}[/dim]")

    def action_command_palette(self) -> None:
        """Ctrl+P is the muscle memory; it just starts the line for you."""
        composer = self.query_one(Composer)
        composer.focus()
        if line_prefix(composer.text) is None:
            composer.text = "/"
            composer.move_cursor(composer.document.end)
        self.query_one(Palette).show(search(""))

    def action_pick_session(self) -> None:
        """Recent conversations, newest first — the list `/sessions` prints,
        except you can choose from it instead of copying an id."""
        # An empty transcript — this session, or one abandoned at the prompt —
        # has no task line and cannot be resumed, so it does not belong in a
        # list of things to resume.
        listing = [
            entry for entry in Session.list_sessions(self.runtime.config.sessions_dir, limit=30)
            if (entry.get("task") or "").strip() and entry["id"] != self.session.id
        ][:15]
        if not listing:
            self.write("[dim]no past sessions yet[/dim]")
            return
        labels = [
            f"{time.strftime('%d %b %H:%M', time.localtime(entry['modified']))}  {entry['task'][:70]}"
            for entry in listing
        ]
        pending = PendingAnswer(question="Resume which conversation?", options=tuple(labels))

        def chosen(index: int | None) -> None:
            if index is None or index < 0:
                return
            self._resume(listing[index]["id"])

        self.push_screen(ChoiceScreen(pending), chosen)

    def action_observe(self) -> None:
        try:
            observation = self.runtime.desktop.observe(screenshot=False)
        except Exception as exc:
            self.write(f"[red]observe failed: {exc}[/red]")
            return
        self.write(observation.summary(max_elements=25))

    # -- slash commands --------------------------------------------------

    @work(thread=True)
    def _slash(self, line: str) -> None:
        """Run a slash command on a worker, so its questions can use modals.

        The same table the chat uses. Keeping a second, smaller one here is how
        the full-screen interface ended up less capable than the plain one —
        you could switch backend from `lai chat` and not from `lai tui`.
        """
        from ..chat.commands import NEW, QUIT, RESUME, Context, run  # noqa: PLC0415

        context = Context(
            runtime=self.runtime,
            ask=self._ask_choice,
            secret=self._ask_secret,
            confirm=self._ask_confirm,
            say=lambda text: self.call_from_thread(self.write, text),
        )
        try:
            answer = run(context, line)
        except Exception as exc:
            self.call_from_thread(self.write, f"[red]{type(exc).__name__}: {exc}[/red]")
            return

        if answer == QUIT:
            self.call_from_thread(self.exit)
        elif answer == NEW:
            self.call_from_thread(self.action_new_session)
        elif answer.startswith(RESUME):
            self.call_from_thread(self._resume, answer[len(RESUME):])
        elif answer:
            self.call_from_thread(self.write, answer)
        self.call_from_thread(self._refresh_status)

    def _resume(self, wanted: str) -> None:
        from ..chat.session_pick import resume as pick  # noqa: PLC0415

        found, why = pick(self.runtime.config.sessions_dir, wanted)
        if found is None:
            self.write(f"[yellow]{why}[/yellow]")
            return
        self.session = found
        self.write(f"[green]↺ {why}[/green]")

    def _refresh_status(self) -> None:
        """A command may have changed the backend or the mode."""
        bar = self.query_one(StatusBar)
        provider = self.runtime.provider
        bar.provider = f"{provider.name}/{provider.model}" if provider else "no model"
        bar.mode = self.runtime.config.safety.mode

    # -- questions from a worker thread ----------------------------------

    def _answer(self, pending: PendingAnswer, screen) -> object:
        """Show a modal and block the calling worker until it is answered."""
        def answered(value) -> None:
            pending.value = value
            pending.event.set()

        self.call_from_thread(self.push_screen, screen, answered)
        pending.event.wait(timeout=600)
        return pending.value

    def _ask_choice(self, question: str, options: list) -> int:
        pending = PendingAnswer(question=question, options=tuple(options), kind="choice")
        value = self._answer(pending, ChoiceScreen(pending))
        return int(value) if isinstance(value, int) else -1

    def _ask_secret(self, question: str) -> str:
        pending = PendingAnswer(question=question, kind="secret")
        return str(self._answer(pending, AskScreen(pending)) or "")

    def _ask_confirm(self, question: str) -> bool:
        pending = PendingAnswer(question=question, kind="confirm")
        return bool(self._answer(pending, AskScreen(pending)))


def run_tui(runtime, *, task: str = "") -> None:
    LaiApp(runtime, task=task).run()
