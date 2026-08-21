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
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, RichLog, Static

from ..agent.session import Session
from ..config import PERMISSION_MODES

REFRESH_INTERVAL = 2.0


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


class StatusBar(Static):
    """Top line: who is answering, under what permissions, and at what cost."""

    provider = reactive("")
    mode = reactive("ask")
    steps = reactive(0)
    tokens = reactive(0)
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
            text.append(f" step {self.steps} ", style="bold")
            text.append(f" {self.elapsed:.0f}s ", style="dim")
        else:
            text.append(" idle ", style="dim")
        text.append(f" {self.tokens:,} tok ", style="dim")
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
    #prompt { dock: bottom; height: 3; border: tall $accent; }
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
        yield Input(placeholder="Tell me what to do on this desktop…", id="prompt")
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
        self.query_one(Input).focus()
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

    @on(Input.Submitted, "#prompt")
    def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            self.submit(text)

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

    def action_observe(self) -> None:
        try:
            observation = self.runtime.desktop.observe(screenshot=False)
        except Exception as exc:
            self.write(f"[red]observe failed: {exc}[/red]")
            return
        self.write(observation.summary(max_elements=25))

    # -- slash commands --------------------------------------------------

    def _slash(self, line: str) -> None:
        command, _, rest = line[1:].partition(" ")
        command = command.lower()
        if command in ("quit", "q", "exit"):
            self.exit()
        elif command == "new":
            self.action_new_session()
        elif command == "observe":
            self.action_observe()
        elif command == "mode":
            mode = rest.strip()
            if mode in PERMISSION_MODES:
                from dataclasses import replace  # noqa: PLC0415

                self.runtime.config = self.runtime.config.with_overrides(
                    safety=replace(self.runtime.config.safety, mode=mode)
                )
                self.runtime.policy.config = self.runtime.config.safety
                self.query_one(StatusBar).mode = mode
                self.write(f"[dim]permission mode → {mode}[/dim]")
            else:
                self.write(f"[red]mode must be one of {', '.join(PERMISSION_MODES)}[/red]")
        elif command == "tools":
            specs = self.runtime.registry.specs()
            if rest.strip():
                specs = [s for s in specs if rest.strip().lower() in s.name.lower()]
            for spec in specs[:60]:
                self.write(f"  [cyan]{spec.name}[/cyan] [dim]({spec.risk.value})[/dim] {spec.description[:70]}")
            self.write(f"[dim]{len(specs)} tool(s)[/dim]")
        elif command == "skills":
            found = self.runtime.skills.search(rest.strip()) if rest.strip() else self.runtime.skills.list()
            for skill in found[:40]:
                self.write(f"  [green]{skill.name}[/green] [dim]{skill.description[:70]}[/dim]")
            self.write(f"[dim]{len(found)} skill(s)[/dim]")
        elif command == "session":
            self.write(str(self.session.summary()))
        elif command in ("help", "h", "?"):
            self.write(
                "[bold]/help /quit /new /observe /mode <m> /tools [filter] /skills [q] /session[/bold]\n"
                "[dim]ctrl+c interrupt · f2 cycle mode · f5 observe · ctrl+n new session[/dim]"
            )
            self.write("[bold]Example tasks:[/bold]")
            for example in EXAMPLE_TASKS:
                self.write(f"  [cyan]{example}[/cyan]")
        else:
            self.write(f"[red]unknown command /{command}[/red] — try /help")


def run_tui(runtime, *, task: str = "") -> None:
    LaiApp(runtime, task=task).run()
