"""A desktop of the agent's own.

Everything else in LAI drives *your* screen — one mouse, one focus, taken in
turns. A virtual display removes the turn-taking, and the properties that make
that true are: applications land on the other display, the lock is per-screen
so the two agents do not queue behind each other, and the server is always
cleaned up, because a leaked X server is a leaked X server forever.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from lai.errors import BackendUnavailable
from lai.osl.virtual import (
    SERVERS,
    VirtualDisplay,
    _server_command,
    available,
    free_display,
)

# -- picking a display ---------------------------------------------------


def test_a_free_display_number_is_found(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    assert free_display(start=90) == ":90"


def test_taken_display_numbers_are_skipped(monkeypatch):
    taken = {"/tmp/.X90-lock", "/tmp/.X11-unix/X91"}
    monkeypatch.setattr(os.path, "exists", lambda path: path in taken)
    assert free_display(start=90) == ":92"


def test_running_out_of_numbers_is_an_error(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    with pytest.raises(BackendUnavailable, match="no free X display"):
        free_display(start=90, end=92)


# -- the server command --------------------------------------------------


def test_xvfb_is_asked_for_an_off_screen_screen():
    command = _server_command("Xvfb", ":90", (1920, 1080), 24)
    assert command[:2] == ["Xvfb", ":90"]
    assert "1920x1080x24" in command
    assert "-nolisten" in command, "a virtual screen has no business on the network"


def test_xephyr_is_asked_for_a_window():
    command = _server_command("Xephyr", ":90", (1280, 800), 24)
    assert command[:2] == ["Xephyr", ":90"]
    assert "1280x800" in command


def test_off_screen_is_preferred_when_both_exist(monkeypatch):
    """A nested window still sits on the desktop you are trying to keep."""
    from lai.osl.virtual import _preferred_server

    monkeypatch.setattr("lai.osl.virtual.shutil.which", lambda name: "/usr/bin/" + name)
    assert available() == list(SERVERS)
    assert _preferred_server() == "Xvfb"


def test_the_nested_server_is_used_when_it_is_all_there_is(monkeypatch):
    from lai.osl.virtual import _preferred_server

    monkeypatch.setattr("lai.osl.virtual.shutil.which",
                        lambda name: "/usr/bin/Xephyr" if name == "Xephyr" else None)
    assert _preferred_server() == "Xephyr"


def test_no_server_installed_says_what_to_install(monkeypatch):
    monkeypatch.setattr("lai.osl.virtual.shutil.which", lambda name: None)
    with pytest.raises(BackendUnavailable, match="no virtual X server"):
        VirtualDisplay().start()


# -- lifecycle -----------------------------------------------------------


class FakeProcess:
    def __init__(self, alive=True):
        self.alive = alive
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def _fake_start(monkeypatch, *, server_alive=True, socket_exists=True):
    started: list = []

    def popen(command, **kwargs):
        started.append(command)
        return FakeProcess(alive=server_alive)

    monkeypatch.setattr("lai.osl.virtual.shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(os.path, "exists", lambda path: socket_exists and "X11-unix" in path)
    monkeypatch.setattr("lai.osl.virtual.time.sleep", lambda seconds: None)
    return started


def test_starting_brings_up_a_server_and_a_window_manager(monkeypatch):
    """Without a window manager, apps have no decorations and never get focus —
    which looks exactly like a broken agent."""
    started = _fake_start(monkeypatch)
    screen = VirtualDisplay(display=":90")
    assert screen.start() == ":90"
    assert screen.running
    assert started[0][0].endswith("Xvfb")
    assert len(started) == 2, "the window manager is not optional"


def test_the_window_manager_can_be_left_out(monkeypatch):
    started = _fake_start(monkeypatch)
    VirtualDisplay(display=":90").start(window_manager=False)
    assert len(started) == 1


def test_a_server_that_dies_immediately_is_reported(monkeypatch):
    _fake_start(monkeypatch, server_alive=False, socket_exists=False)
    with pytest.raises(BackendUnavailable, match="exited immediately"):
        VirtualDisplay(display=":90").start()


def test_a_failed_start_leaves_nothing_running(monkeypatch):
    _fake_start(monkeypatch, server_alive=False, socket_exists=False)
    screen = VirtualDisplay(display=":90")
    with pytest.raises(BackendUnavailable):
        screen.start()
    assert not screen.running


def test_stopping_terminates_everything(monkeypatch):
    _fake_start(monkeypatch)
    screen = VirtualDisplay(display=":90")
    screen.start()
    server, manager = screen._server_process, screen._wm_process
    screen.stop()
    assert server.terminated and manager.terminated
    assert not screen.running


def test_it_is_a_context_manager(monkeypatch):
    _fake_start(monkeypatch)
    with VirtualDisplay(display=":90") as screen:
        assert screen.running
    assert not screen.running


def test_starting_twice_does_not_start_twice(monkeypatch):
    started = _fake_start(monkeypatch)
    screen = VirtualDisplay(display=":90")
    screen.start()
    screen.start()
    assert len(started) == 2, "the second call is a no-op"


def test_the_environment_points_applications_at_the_new_display(monkeypatch):
    _fake_start(monkeypatch)
    screen = VirtualDisplay(display=":90")
    screen.start()
    assert screen.env["DISPLAY"] == ":90"


# -- it does not contend with the human ----------------------------------


def test_the_lock_is_per_display(tmp_path):
    """An agent on its own screen takes nothing from the person using the real
    one, so making it queue behind them would defeat the whole point."""
    from lai.config import load_config
    from lai.runtime import _open_desktop_lock

    config = load_config().with_overrides(home=tmp_path)
    real = _open_desktop_lock(config, display=":0")
    virtual = _open_desktop_lock(config, display=":90")
    assert real.path != virtual.path

    real.acquire()
    try:
        virtual.acquire()  # must not block or raise
        assert virtual.held
        virtual.release()
    finally:
        real.release()


@pytest.mark.x11
def test_a_real_virtual_display_runs_an_application():
    """The end-to-end claim: an app launched on it lands there, not on yours."""
    import time

    if not available():
        pytest.skip("no virtual X server installed")

    screen = VirtualDisplay(size=(800, 600))
    screen.start()
    try:
        subprocess.Popen(  # noqa: S607
            ["gnome-calculator"], env=screen.env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 20
        found = ""
        while time.time() < deadline and not found:
            result = subprocess.run(  # noqa: S607
                ["xdotool", "search", "--name", "Calculator"],
                env=screen.env, capture_output=True, text=True, check=False,
            )
            found = result.stdout.strip()
            if not found:
                time.sleep(0.5)
        assert found, "the calculator never appeared on the virtual display"

        shot = screen.screenshot()
        assert shot.size == (800, 600)
        assert shot.png[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        screen.stop()
