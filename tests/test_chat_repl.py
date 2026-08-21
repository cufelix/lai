"""The chat loop itself — what `lai` runs when you type nothing else.

This is the front door, so the behaviours worth pinning are the ones a user
meets on a bad day: a task that fails must not end the session, Ctrl+C must
interrupt the task rather than the conversation, an unknown slash command must
not be sent to the model as work, and Ctrl+D must always get you out.
"""

from __future__ import annotations

import pytest

from lai.agent.loop import RunResult
from lai.chat.repl import run_chat, status_line
from lai.errors import Interrupted, LaiError, ProviderError


class FakeOut:
    def __init__(self):
        self.lines: list[str] = []

    def write(self, text="", **kwargs):
        self.lines.append(str(text))

    def error(self, text):
        self.lines.append(f"error: {text}")

    def rule(self, title=""):
        pass

    def raw(self, text):
        self.lines.append(str(text))

    def spinner(self, text):
        return None

    @property
    def text(self):
        return "\n".join(self.lines)


class FakeAgent:
    def __init__(self, result=None, raises=None):
        self.result = result or RunResult(status="completed", summary="did it", steps=1)
        self.raises = raises
        self.interrupted = False
        self.tasks: list[str] = []

    def run(self, task):
        self.tasks.append(task)
        if self.raises is not None:
            raise self.raises
        return self.result

    def interrupt(self):
        self.interrupted = True


class FakeRuntime:
    def __init__(self, tmp_path, agent=None, provider=True):
        from lai.config import load_config

        self.config = load_config().with_overrides(home=tmp_path)
        self.provider = type("P", (), {"name": "zai", "model": "glm-5"})() if provider else None
        self.provider_error = "no key anywhere"
        self.registry = type("R", (), {"__len__": lambda self: 7, "specs": lambda self: []})()
        self.skills = type("S", (), {"__len__": lambda self: 2, "list": lambda self: [],
                                     "search": lambda self, q: []})()
        self.policy = type("Pol", (), {"config": self.config.safety})()
        self.desktop = None
        self.journal = None
        self.extra: dict = {}
        self._agent = agent or FakeAgent()

    def agent(self, **kwargs):
        return self._agent


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    for name in ("LAI_MODE", "LAI_PROVIDER", "LAI_MODEL", "LAI_FALLBACK"):
        monkeypatch.delenv(name, raising=False)


def scripted(monkeypatch, *lines):
    """Feed the reader a fixed sequence, then EOF."""
    queue = list(lines)

    def read(self, prompt, **kwargs):
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("lai.chat.repl.Reader.read", read)


# -- the basics ----------------------------------------------------------


def test_a_task_is_run_and_its_result_shown(tmp_path, monkeypatch):
    agent = FakeAgent()
    scripted(monkeypatch, "open the editor")
    out = FakeOut()
    assert run_chat(FakeRuntime(tmp_path, agent), out=out) == 0
    assert agent.tasks == ["open the editor"]
    assert "did it" in out.text


def test_a_first_task_from_the_command_line_runs_immediately(tmp_path, monkeypatch):
    agent = FakeAgent()
    scripted(monkeypatch)
    run_chat(FakeRuntime(tmp_path, agent), out=FakeOut(), task="count the windows")
    assert agent.tasks == ["count the windows"]


def test_blank_input_is_ignored(tmp_path, monkeypatch):
    agent = FakeAgent()
    scripted(monkeypatch, "", "   ")
    run_chat(FakeRuntime(tmp_path, agent), out=FakeOut())
    assert agent.tasks == []


def test_ctrl_d_leaves(tmp_path, monkeypatch):
    scripted(monkeypatch)
    out = FakeOut()
    assert run_chat(FakeRuntime(tmp_path), out=out) == 0
    assert "bye" in out.text


def test_quit_leaves(tmp_path, monkeypatch):
    scripted(monkeypatch, "/quit", "never reached")
    agent = FakeAgent()
    assert run_chat(FakeRuntime(tmp_path, agent), out=FakeOut()) == 0
    assert agent.tasks == []


# -- surviving a bad day -------------------------------------------------


def test_a_failing_task_does_not_end_the_session(tmp_path, monkeypatch):
    agent = FakeAgent(raises=LaiError("the desktop exploded"))
    scripted(monkeypatch, "do something", "do something else")
    out = FakeOut()
    assert run_chat(FakeRuntime(tmp_path, agent), out=out) == 0
    assert agent.tasks == ["do something", "do something else"], "it kept going"
    assert "exploded" in out.text


def test_a_provider_failure_points_at_the_way_out(tmp_path, monkeypatch):
    agent = FakeAgent(raises=ProviderError("every backend refused"))
    scripted(monkeypatch, "do something")
    out = FakeOut()
    run_chat(FakeRuntime(tmp_path, agent), out=out)
    assert "/model" in out.text, "an error the user cannot act on is a bug"


def test_ctrl_c_interrupts_the_task_not_the_conversation(tmp_path, monkeypatch):
    agent = FakeAgent(raises=KeyboardInterrupt())
    scripted(monkeypatch, "a long task", "another one")
    out = FakeOut()
    assert run_chat(FakeRuntime(tmp_path, agent), out=out) == 0
    assert agent.interrupted
    assert "interrupted" in out.text
    assert len(agent.tasks) == 2


def test_an_interrupted_run_is_reported(tmp_path, monkeypatch):
    agent = FakeAgent(raises=Interrupted("stopped by user"))
    scripted(monkeypatch, "a task")
    out = FakeOut()
    run_chat(FakeRuntime(tmp_path, agent), out=out)
    assert "interrupted" in out.text


def test_ctrl_c_at_the_prompt_says_how_to_leave(tmp_path, monkeypatch):
    queue = [KeyboardInterrupt(), EOFError()]

    def read(self, prompt, **kwargs):
        raise queue.pop(0)

    monkeypatch.setattr("lai.chat.repl.Reader.read", read)
    out = FakeOut()
    assert run_chat(FakeRuntime(tmp_path), out=out) == 0
    assert "ctrl+d" in out.text


def test_with_no_backend_it_says_so_before_asking_anything(tmp_path, monkeypatch):
    """It used to exit here. Exiting sends you to the same advice from a place
    where you cannot act on it."""
    scripted(monkeypatch, "a task")
    out = FakeOut()
    assert run_chat(FakeRuntime(tmp_path, provider=False), out=out) == 0
    assert "No model yet" in out.text and "/model" in out.text


# -- slash commands ------------------------------------------------------


def test_a_slash_command_is_never_sent_to_the_model(tmp_path, monkeypatch):
    agent = FakeAgent()
    scripted(monkeypatch, "/help", "/status")
    run_chat(FakeRuntime(tmp_path, agent), out=FakeOut())
    assert agent.tasks == []


def test_an_unknown_command_is_answered_not_executed(tmp_path, monkeypatch):
    agent = FakeAgent()
    scripted(monkeypatch, "/nonsense")
    out = FakeOut()
    run_chat(FakeRuntime(tmp_path, agent), out=out)
    assert agent.tasks == []
    assert "unknown command" in out.text


def test_new_starts_a_fresh_session(tmp_path, monkeypatch):
    runtime = FakeRuntime(tmp_path)
    scripted(monkeypatch, "first", "/new", "second")
    out = FakeOut()
    run_chat(runtime, out=out)
    assert "forgotten" in out.text
    assert runtime.extra["chat_session"].messages == []


def test_resume_swaps_in_an_earlier_conversation(tmp_path, monkeypatch):
    from lai.agent.providers.base import Message
    from lai.agent.session import Session

    runtime = FakeRuntime(tmp_path)
    earlier = Session()
    earlier.task = "an earlier task"
    earlier.bind(runtime.config.sessions_dir)
    earlier.append(Message.user("something from before"))

    scripted(monkeypatch, "/resume")
    out = FakeOut()
    run_chat(runtime, out=out)
    assert runtime.extra["chat_session"].id == earlier.id
    assert "continuing session" in out.text


def test_resuming_nothing_says_so_and_carries_on(tmp_path, monkeypatch):
    scripted(monkeypatch, "/resume", "/quit")
    out = FakeOut()
    assert run_chat(FakeRuntime(tmp_path), out=out) == 0
    assert "no past sessions" in out.text


def test_the_welcome_names_the_backend_and_the_standbys(tmp_path, monkeypatch):
    runtime = FakeRuntime(tmp_path)
    runtime.provider = type("P", (), {
        "name": "zai", "model": "glm-5", "chain": ["zai", "cli:claude"], "failures": {},
    })()
    scripted(monkeypatch)
    out = FakeOut()
    run_chat(runtime, out=out)
    assert "zai/glm-5" in out.text
    assert "cli:claude" in out.text and "failover" in out.text


def test_the_status_line_survives_a_provider_without_a_chain(tmp_path):
    runtime = FakeRuntime(tmp_path)
    assert "zai/glm-5" in status_line(runtime)


# -- starting with nothing configured ------------------------------------


def test_with_no_backend_the_chat_still_opens(tmp_path, monkeypatch):
    """`/model` is how this gets fixed, and you cannot reach it from a command
    that exited."""
    scripted(monkeypatch, "/help")
    out = FakeOut()
    assert run_chat(FakeRuntime(tmp_path, provider=False), out=out) == 0
    assert "/model" in out.text
    assert "No model yet" in out.text


def test_a_task_without_a_backend_points_at_the_menu(tmp_path, monkeypatch):
    agent = FakeAgent()
    scripted(monkeypatch, "open the editor")
    out = FakeOut()
    run_chat(FakeRuntime(tmp_path, agent, provider=False), out=out)
    assert agent.tasks == [], "nothing is attempted without a model"
    assert "/model" in out.text


def test_the_menu_is_reachable_with_no_backend(tmp_path, monkeypatch):
    """The whole point: the fix is one keystroke away, not a restart."""
    from lai.chat.commands import COMMANDS

    scripted(monkeypatch, "/model")
    runtime = FakeRuntime(tmp_path, provider=False)
    monkeypatch.setattr(
        "lai.chat.backends.catalogue",
        lambda rt=None, **kw: [],
    )
    out = FakeOut()
    assert run_chat(runtime, out=out) == 0
    assert "model" in COMMANDS
