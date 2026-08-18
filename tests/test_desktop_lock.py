"""The cross-process desktop claim.

The failure this prevents is not three half-finished tasks — it is one mangled
desktop and three agents confidently reporting on a screen somebody else just
changed. The properties that matter: a second process is refused and told who
holds it, a crashed holder never keeps the desktop forever, and one process
claiming twice does not deadlock against itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from lai.osl.lock import DesktopBusy, DesktopLock, Holder


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "desktop.lock"


def test_a_free_desktop_is_claimed(lock_path):
    lock = DesktopLock(lock_path, task="open the editor")
    assert lock.acquire() is True
    assert lock.held
    lock.release()
    assert not lock.held


def test_the_claim_records_who_holds_it(lock_path):
    with DesktopLock(lock_path, task="draw a house"):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["task"] == "draw a house"
        assert payload["since"] > 0


def test_a_second_process_is_refused_and_told_who_has_it(lock_path):
    """The whole point: the refusal has to be actionable, not just 'busy'."""
    helper = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(_src())!r})
        from lai.osl.lock import DesktopLock
        lock = DesktopLock({str(lock_path)!r}, task="a long running task")
        lock.acquire()
        print("held", flush=True)
        time.sleep(30)
    """)
    child = subprocess.Popen(
        [sys.executable, "-c", helper], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "held"
        with pytest.raises(DesktopBusy) as info:
            DesktopLock(lock_path).acquire()
        message = str(info.value)
        assert str(child.pid) in message
        assert "a long running task" in message
        assert info.value.holder.pid == child.pid
    finally:
        child.kill()
        child.wait(timeout=5)


def test_the_desktop_is_freed_when_the_holder_dies(lock_path):
    """A crashed agent must never claim the desktop forever — hence flock."""
    helper = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(_src())!r})
        from lai.osl.lock import DesktopLock
        DesktopLock({str(lock_path)!r}, task="about to be killed").acquire()
        print("held", flush=True)
        time.sleep(30)
    """)
    child = subprocess.Popen([sys.executable, "-c", helper], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "held"
    child.kill()
    child.wait(timeout=5)

    lock = DesktopLock(lock_path)
    for _ in range(40):  # the kernel drops it as the process is reaped
        try:
            lock.acquire()
            break
        except DesktopBusy:
            time.sleep(0.05)
    assert lock.held, "a dead holder must not keep the desktop"
    lock.release()


def test_waiting_gives_up_with_a_useful_error(lock_path):
    helper = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(_src())!r})
        from lai.osl.lock import DesktopLock
        DesktopLock({str(lock_path)!r}, task="holding").acquire()
        print("held", flush=True)
        time.sleep(30)
    """)
    child = subprocess.Popen([sys.executable, "-c", helper], stdout=subprocess.PIPE, text=True)
    try:
        child.stdout.readline()
        started = time.monotonic()
        with pytest.raises(DesktopBusy):
            DesktopLock(lock_path).acquire(wait=0.5)
        assert time.monotonic() - started >= 0.4, "it should actually have waited"
    finally:
        child.kill()
        child.wait(timeout=5)


def test_claiming_twice_in_one_process_does_not_deadlock(lock_path):
    """The daemon may hold it when a tool spawns a subagent."""
    lock = DesktopLock(lock_path, task="outer")
    lock.acquire()
    lock.acquire()
    assert lock.held
    lock.release()
    assert lock.held, "the outer claim is still in force"
    lock.release()
    assert not lock.held


def test_the_lock_can_be_used_as_a_context_manager(lock_path):
    with DesktopLock(lock_path):
        assert DesktopLock(lock_path).holder().pid == os.getpid()


def test_releasing_a_lock_that_was_never_taken_is_harmless(lock_path):
    DesktopLock(lock_path).release()


def test_an_acquired_lock_survives_losing_its_last_reference(lock_path):
    """flock lives on an open file handle: if the garbage collector closes it,
    the desktop is released while an agent is still driving. It must not."""
    import gc

    DesktopLock(lock_path, task="nobody kept a reference").acquire()
    gc.collect()
    with pytest.raises(DesktopBusy):
        DesktopLock(lock_path).acquire()


def test_a_missing_lock_file_has_no_holder(tmp_path):
    assert DesktopLock(tmp_path / "nothing.lock").holder() == Holder()


def test_a_corrupt_lock_file_has_no_holder(lock_path):
    lock_path.write_text("not json at all", encoding="utf-8")
    assert DesktopLock(lock_path).holder() == Holder()


def test_the_holder_describes_itself_usefully():
    holder = Holder(pid=42, task="open firefox", since=time.time() - 10)
    text = holder.describe()
    assert "42" in text and "open firefox" in text and "s" in text
    assert Holder().describe() == "another process"


def test_the_directory_is_created_if_it_is_missing(tmp_path):
    lock = DesktopLock(tmp_path / "fresh" / "desktop.lock")
    lock.acquire()
    assert (tmp_path / "fresh" / "desktop.lock").is_file()
    lock.release()


def _src() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / "src")


# -- the agent holds it for the length of a run --------------------------


def _agent(config, lock, provider=None):
    """A bare Agent wired only enough to run one turn."""
    from lai.agent.loop import Agent, RunResult
    from lai.agent.session import Session

    agent = Agent.__new__(Agent)
    agent.config = config
    agent.desktop_lock = lock
    agent.session = Session()
    agent.journal = None
    agent._run = lambda task, max_steps=None: RunResult(status="completed", steps=1)
    return agent


def test_an_acting_run_claims_the_desktop(lock_path, tmp_path):
    from lai.config import load_config

    config = load_config().with_overrides(home=tmp_path)
    seen = {}
    lock = DesktopLock(lock_path, task="t")
    agent = _agent(config, lock)
    inner = agent._run
    agent._run = lambda task, max_steps=None: seen.update(held=lock.held) or inner(task)

    agent.run("open the editor")
    assert seen["held"] is True, "the desktop must be claimed while the agent drives it"
    assert not lock.held, "and released afterwards"


@pytest.mark.parametrize(("mode", "dry_run"), [("readonly", False), ("ask", True)])
def test_a_run_that_cannot_change_anything_does_not_queue(lock_path, tmp_path, mode, dry_run):
    """Reading the screen while another agent works is useful, not dangerous."""
    from dataclasses import replace

    from lai.config import load_config

    config = load_config().with_overrides(home=tmp_path)
    config = config.with_overrides(safety=replace(config.safety, mode=mode, dry_run=dry_run))

    seen = {}
    lock = DesktopLock(lock_path)
    agent = _agent(config, lock)
    inner = agent._run
    agent._run = lambda task, max_steps=None: seen.update(held=lock.held) or inner(task)

    agent.run("what is on screen?")
    assert seen["held"] is False


def test_the_lock_is_released_even_when_a_run_explodes(lock_path, tmp_path):
    from lai.config import load_config

    config = load_config().with_overrides(home=tmp_path)
    lock = DesktopLock(lock_path)
    agent = _agent(config, lock)

    def explode(task, max_steps=None):
        raise RuntimeError("the desktop vanished")

    agent._run = explode
    with pytest.raises(RuntimeError):
        agent.run("do a thing")
    assert not lock.held, "a crashed run must not keep the desktop claimed"


def test_the_refusal_names_the_task_that_is_running(lock_path, tmp_path):
    """'busy' is an obstacle; 'busy opening firefox' is information."""
    from lai.config import load_config

    config = load_config().with_overrides(home=tmp_path)
    lock = DesktopLock(lock_path)
    agent = _agent(config, lock)
    seen = {}
    inner = agent._run
    agent._run = lambda task, max_steps=None: seen.update(holder=DesktopLock(lock_path).holder()) or inner(task)

    agent.run("open firefox and check the weather")
    assert "open firefox" in seen["holder"].task


# -- the interfaces cope with a busy desktop -----------------------------


def test_the_chat_reports_a_busy_desktop_without_ending_the_session(tmp_path, monkeypatch):
    """A task refused for lack of the desktop must not close the conversation."""
    from lai.chat.repl import run_chat
    from lai.config import load_config

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    written: list[str] = []

    class FakeOut:
        def write(self, text="", **kwargs):
            written.append(str(text))

        def error(self, text):
            written.append(f"error: {text}")

        def rule(self, title=""):
            pass

        def raw(self, text):
            pass

        def spinner(self, text):
            return None

    class Busy:
        def run(self, task):
            raise DesktopBusy(Holder(pid=4242, task="drawing a house"))

        def interrupt(self):
            pass

    class FakeRuntime:
        def __init__(self):
            self.config = load_config().with_overrides(home=tmp_path)
            self.provider = type("P", (), {"name": "zai", "model": "glm"})()
            self.provider_error = ""
            self.registry = type("R", (), {"__len__": lambda self: 1})()
            self.skills = type("S", (), {"__len__": lambda self: 0})()
            self.extra: dict = {}

        def agent(self, **kwargs):
            return Busy()

    def eof(self, prompt, **kwargs):
        raise EOFError

    monkeypatch.setattr("lai.chat.repl.Reader.read", eof)
    code = run_chat(FakeRuntime(), out=FakeOut(), task="open firefox")
    text = "\n".join(written)
    assert code == 0, "the session survives"
    assert "4242" in text and "drawing a house" in text
    assert "one desktop, one agent" in text.lower()
