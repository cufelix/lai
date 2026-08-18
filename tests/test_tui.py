"""TUI: layout, event rendering, mode control and the approval modal."""

from __future__ import annotations

import threading

import pytest

from lai.tui.app import ApprovalScreen, DesktopPanel, LaiApp, PendingApproval, PlanPanel, StatusBar

pytestmark = pytest.mark.slow


class FakeResult:
    def __init__(self, status="completed", **kwargs):
        self.status = status
        self.summary = kwargs.get("summary", "did it")
        self.verification = kwargs.get("verification", "")
        self.artifacts = kwargs.get("artifacts", [])
        self.steps = kwargs.get("steps", 2)
        self.elapsed = kwargs.get("elapsed", 1.0)
        self.error = kwargs.get("error", "")


class FakeAgent:
    def __init__(self, *, on_event=None, approver=None, result=None, hook=None, **kwargs):
        self.on_event = on_event
        self.approver = approver
        self.result = result or FakeResult()
        self.hook = hook
        self.interrupted = False

    def run(self, task):
        if self.hook:
            self.hook(self)
        if self.on_event:
            self.on_event("step", {"step": 1, "of": 5})
            self.on_event("tool_call", {"name": "ui_snapshot", "input": {"scope": "focused"}})
            self.on_event("tool_result", {"name": "ui_snapshot", "ok": True, "summary": "12 elements"})
        return self.result

    def interrupt(self):
        self.interrupted = True


class RuntimeProxy:
    """Delegates to a real Runtime but hands out fake agents.

    Runtime is a slotted dataclass, so it cannot be monkeypatched directly;
    a proxy keeps the real desktop/registry/skills while faking the model.
    """

    def __init__(self, real):
        self._real = real
        self.config = real.config
        self.policy = real.policy
        self.desktop = real.desktop
        self.registry = real.registry
        self.skills = real.skills
        self.provider = type("P", (), {"name": "fake", "model": "m1", "close": lambda self: None})()
        self.provider_error = ""
        self.factory_holder = {"factory": FakeAgent}

    def __getattr__(self, name):
        return getattr(self._real, name)

    def agent(self, **kwargs):
        return self.factory_holder["factory"](**kwargs)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    from lai.runtime import build_runtime

    real = build_runtime(with_provider=False, with_mcp=False)
    try:
        yield RuntimeProxy(real)
    finally:
        real.close()


def panel_text(app, widget_type) -> str:
    return str(app.query_one(widget_type).content)


# -- layout --------------------------------------------------------------


async def test_app_mounts_with_every_panel(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        assert bar.provider == "fake/m1"
        assert bar.mode == runtime.config.safety.mode
        assert not bar.busy
        assert app.query_one(PlanPanel) is not None
        assert app.query_one(DesktopPanel) is not None


async def test_status_bar_renders_each_mode(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        for mode in ("readonly", "ask", "auto", "yolo"):
            bar.mode = mode
            assert mode in bar.render().plain


async def test_status_bar_shows_progress_when_busy(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        bar.busy, bar.steps, bar.elapsed = True, 4, 12.0
        rendered = bar.render().plain
        assert "step 4" in rendered and "12s" in rendered


async def test_desktop_panel_reads_the_real_desktop(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert "focused" in panel_text(app, DesktopPanel)


async def test_desktop_panel_survives_a_broken_desktop(runtime):
    class Boom:
        def active_window(self):
            raise RuntimeError("x server gone")

        def list_windows(self):
            raise RuntimeError("x server gone")

    runtime.desktop.windows = Boom()
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(DesktopPanel).refresh_desktop()
        await pilot.pause()
        assert "unavailable" in panel_text(app, DesktopPanel)


async def test_plan_panel_marks_progress(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(PlanPanel).show(["open editor", "type text", "save file"], 1)
        await pilot.pause()
        text = panel_text(app, PlanPanel)
        assert "✓ open editor" in text
        assert "→ type text" in text
        assert "save file" in text


# -- event rendering -----------------------------------------------------


async def test_event_stream_updates_the_ui(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._event("step", {"step": 3, "of": 10})
        app._event("tool_call", {"name": "ui_click", "input": {"name": "Save"}})
        app._event("tool_result", {"name": "ui_click", "ok": True, "summary": "clicked"})
        app._event("error", {"error": "hiccup"})
        app._event("compacting", {"estimated_tokens": 1000})
        await pilot.pause()
        assert app.query_one(StatusBar).steps == 3


async def test_plan_update_tool_call_fills_the_plan_panel(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._event("tool_call", {"name": "plan_update", "input": {"steps": ["a", "b"], "current": 1}})
        await pilot.pause()
        text = panel_text(app, PlanPanel)
        assert "✓ a" in text and "→ b" in text


async def test_finished_renders_every_status(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for status in ("completed", "blocked", "budget_exceeded", "error", "interrupted"):
            app._finished(FakeResult(status, error="boom" if status == "error" else ""))
        await pilot.pause()  # must not raise


async def test_finished_shows_artifacts_and_verification(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._finished(FakeResult(verification="re-read the file", artifacts=["/tmp/a.txt"]))
        await pilot.pause()


# -- running a task ------------------------------------------------------


async def test_submitting_a_task_runs_the_agent(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.submit("open the editor")
        for _ in range(80):
            await pilot.pause()
            if not app.query_one(StatusBar).busy and app.query_one(StatusBar).steps:
                break
        assert app.query_one(StatusBar).steps == 1


async def test_submitting_while_busy_is_refused(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(StatusBar).busy = True
        app.submit("another")  # must not raise or start a second run
        await pilot.pause()


async def test_no_provider_is_reported(runtime):
    runtime.provider = None
    runtime.provider_error = "no backend configured"
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.submit("do something")
        await pilot.pause()
        assert app.query_one(StatusBar).provider == "no model"


async def test_interrupt_without_a_run(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_interrupt()  # must not raise
        await pilot.pause()


async def test_interrupt_stops_a_running_agent(runtime):
    release = threading.Event()
    agents: list[FakeAgent] = []

    def factory(**kwargs):
        agent = FakeAgent(hook=lambda a: release.wait(timeout=5), **kwargs)
        agents.append(agent)
        return agent

    runtime.factory_holder["factory"] = factory
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.submit("slow task")
        for _ in range(80):
            await pilot.pause()
            if app.agent is not None:
                break
        app.action_interrupt()
        await pilot.pause()
        assert agents and agents[0].interrupted
        release.set()


# -- modes and commands --------------------------------------------------


async def test_cycle_mode_advances_and_applies(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        before = app.query_one(StatusBar).mode
        app.action_cycle_mode()
        await pilot.pause()
        after = app.query_one(StatusBar).mode
        assert after != before
        assert runtime.config.safety.mode == after
        assert runtime.policy.config.mode == after


async def test_new_session_resets_state(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        before = app.session.id
        app.action_new_session()
        await pilot.pause()
        assert app.session.id != before
        assert app.query_one(StatusBar).tokens == 0


async def test_observe_action(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_observe()  # must not raise
        await pilot.pause()


@pytest.mark.parametrize(
    "command", ["/help", "/tools window", "/skills", "/session", "/observe", "/new", "/frobnicate"]
)
async def test_slash_commands_do_not_crash(runtime, command):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.submit(command)
        await pilot.pause()


async def test_slash_mode_validates(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.submit("/mode yolo")
        await pilot.pause()
        assert app.query_one(StatusBar).mode == "yolo"
        app.submit("/mode banana")
        await pilot.pause()
        assert app.query_one(StatusBar).mode == "yolo", "an invalid mode must be rejected"


# -- approvals -----------------------------------------------------------


async def test_approval_modal_allows_with_a_keypress(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pending = PendingApproval("computer_click", {"x": 1, "y": 2}, "needs confirmation")
        app._ask(pending)
        await pilot.pause()
        assert isinstance(app.screen, ApprovalScreen)
        await pilot.press("y")
        await pilot.pause()
        assert pending.granted and pending.event.is_set()


async def test_approval_modal_refuses_with_a_keypress(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pending = PendingApproval("shell_exec", {"command": "rm -rf x"}, "destructive")
        app._ask(pending)
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        assert not pending.granted and pending.event.is_set()


async def test_approval_modal_escape_refuses(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pending = PendingApproval("computer_type", {"text": "x"}, "needs confirmation")
        app._ask(pending)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not pending.granted and pending.event.is_set()


async def test_approval_modal_shows_the_tool_and_reason(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pending = PendingApproval("shell_exec", {"command": "apt install x"}, "needs a look")
        app._ask(pending)
        await pilot.pause()
        rendered = app.screen.query_one("#approval-tool").content
        assert "shell_exec" in str(rendered)
        await pilot.press("n")


async def test_approver_blocks_the_worker_until_answered(runtime):
    """The whole point: the agent thread waits for a human decision."""
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        verdict = type("V", (), {"reason": "needs confirmation"})()
        answer: dict = {}

        def worker():
            answer["granted"] = app._approver("computer_click", {"x": 1, "y": 1}, verdict)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        for _ in range(80):
            await pilot.pause()
            if isinstance(app.screen, ApprovalScreen):
                break
        assert "granted" not in answer, "the worker must still be blocked"
        await pilot.press("y")
        thread.join(timeout=5)
        assert answer.get("granted") is True


# -- first impression ----------------------------------------------------


def feed_text(app) -> str:
    """Everything written to the log, as plain text."""
    from textual.widgets import RichLog

    log = app.query_one("#feed", RichLog)
    return "\n".join(str(line) for line in log.lines)


async def test_the_first_screen_suggests_what_to_type(runtime):
    """An empty prompt is the worst thing a new user can be shown."""
    from lai.tui.app import EXAMPLE_TASKS

    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text = feed_text(app)
        assert "native desktop agent" in text
        assert any(example.split()[0] in text for example in EXAMPLE_TASKS)
        assert "windows" in text, "the safest example should be offered first"


async def test_the_first_screen_shows_the_keys(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text = feed_text(app)
        assert "ctrl+c" in text and "f2" in text and "/help" in text


async def test_without_a_backend_it_points_at_setup(runtime):
    """The one failure a new user hits most must name the command that fixes it."""
    runtime.provider = None
    runtime.provider_error = "no model backend available"
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text = feed_text(app)
        assert "lai setup" in text
        assert "no model backend available" in text
        assert "Try asking for" not in text, "do not suggest tasks that cannot run"


async def test_help_repeats_the_examples(runtime):
    app = LaiApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._slash("/help")
        await pilot.pause()
        text = feed_text(app)
        assert "Example tasks" in text
        assert "/mode" in text
