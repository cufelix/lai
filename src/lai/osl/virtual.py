"""A desktop of the agent's own.

Everything else in LAI drives *your* screen: one mouse, one keyboard, one
focus, shared with whoever is sitting at the machine. That is the right default
— the whole point is acting on the desktop you actually use — but it means the
agent and its owner take turns.

A virtual display removes the turn-taking. The agent gets a second X server
with its own root window, its own pointer and its own focus; applications it
launches live there; screenshots come from there. You keep typing in yours and
nothing the agent does reaches your keyboard, your clipboard focus or your
window stack.

Two servers can provide it, and which one you want differs:

* **Xvfb** — entirely off-screen. Nothing appears anywhere; you watch through
  `lai web` or a screenshot if you want to. The right choice for work you want
  done rather than watched.
* **Xephyr** — a nested server inside a window on your desktop. You can see
  the agent working, and it still cannot touch your session. The right choice
  the first few times, when you want to see what it is doing.

A window manager is started alongside, because without one applications have
no decorations, cannot be maximised, and frequently never receive focus at
all — which looks exactly like a broken agent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

from ..errors import BackendUnavailable

DEFAULT_SIZE = (1920, 1080)
DEFAULT_DEPTH = 24
FIRST_DISPLAY = 90
LAST_DISPLAY = 120
STARTUP_TIMEOUT = 10.0

# Any of these will do. The list is ordered by how little they get in the way:
# a plain reparenting manager is all that is needed, and a full desktop shell
# would spend its time drawing a panel nobody is looking at.
WINDOW_MANAGERS = (
    ("metacity", ("--sm-disable", "--no-composite")),
    ("marco", ("--sm-disable",)),
    ("muffin", ("--sm-disable",)),
    ("openbox", ()),
    ("i3", ()),
    ("fluxbox", ()),
    ("icewm", ()),
    ("jwm", ()),
    ("twm", ()),
)

SERVERS = ("Xvfb", "Xephyr")


def available() -> list[str]:
    """Which X servers this machine can start a virtual display with."""
    return [name for name in SERVERS if shutil.which(name)]


def free_display(start: int = FIRST_DISPLAY, end: int = LAST_DISPLAY) -> str:
    """A display number nothing is using.

    The lock file is how X itself decides, so it is what to look at — a socket
    can linger after a crash while the lock is already gone, and vice versa.
    """
    for number in range(start, end):
        if not os.path.exists(f"/tmp/.X{number}-lock") and not os.path.exists(  # noqa: S108
            f"/tmp/.X11-unix/X{number}"  # noqa: S108
        ):
            return f":{number}"
    raise BackendUnavailable(
        "no free X display number",
        detail=f"tried :{start} through :{end - 1}",
    )


@dataclass(slots=True)
class VirtualDisplay:
    """A second X server, and the window manager that makes it usable."""

    display: str = ""
    size: tuple = DEFAULT_SIZE
    depth: int = DEFAULT_DEPTH
    server: str = ""
    """Xvfb (off-screen) or Xephyr (nested in a window). Empty picks the best available."""
    _server_process: object = field(default=None, repr=False)
    _wm_process: object = field(default=None, repr=False)
    _started: bool = False
    _restore_display: object = field(default=None, repr=False)
    """Whatever ``DISPLAY`` said before this server existed — the human's."""

    @property
    def host_display(self) -> str:
        """The display the person is actually sitting in front of."""
        return str(self._restore_display or "")

    @property
    def running(self) -> bool:
        process = self._server_process
        return bool(self._started and process is not None and process.poll() is None)

    @property
    def env(self) -> dict:
        """Environment for anything that should run on this display."""
        environment = dict(os.environ)
        environment["DISPLAY"] = self.display
        # A nested server draws into the host display, so it needs the host's
        # authority; the child applications need only the new DISPLAY.
        return environment

    def start(self, *, window_manager: bool = True) -> str:
        """Start the server and a window manager. Returns the display name."""
        if self.running:
            return self.display

        chosen = self.server or _preferred_server()
        if not chosen:
            raise BackendUnavailable(
                "no virtual X server installed",
                detail="install one: sudo apt-get install -y xvfb   (or xserver-xephyr to watch it)",
            )
        self.server = chosen
        self.display = self.display or free_display()
        # Downstream libraries — Xlib, mss, GTK, AT-SPI, xdotool — are all
        # steered by the process-wide DISPLAY, so opening a Desktop on this
        # screen sets it. Remembering the old value is what stops the variable
        # outliving the server and pointing everything at a dead socket.
        self._restore_display = os.environ.get("DISPLAY")
        # A window appearing takes your focus mid-sentence. Note where it was
        # so it can be handed straight back.
        focused = _focused_window() if chosen == "Xephyr" else None

        self._server_process = subprocess.Popen(  # noqa: S603
            _server_command(chosen, self.display, self.size, self.depth),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=_die_with_parent,  # noqa: PLW1509 - see the function
        )
        self._started = True
        try:
            _wait_for_display(self.display, self._server_process, STARTUP_TIMEOUT)
        except Exception:
            self.stop()
            raise

        if window_manager:
            self._start_window_manager()
        if chosen == "Xephyr":
            _give_focus_back(focused)
        return self.display

    def stop(self) -> None:
        """Shut it down. Applications on it die with it, which is the point."""
        for attribute in ("_wm_process", "_server_process"):
            process = getattr(self, attribute)
            setattr(self, attribute, None)
            if process is None:
                continue
            try:
                process.terminate()
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass
        self._started = False
        self._put_display_back()

    def _put_display_back(self) -> None:
        """Undo the environment change, so the next thing to look sees yours."""
        if os.environ.get("DISPLAY") != self.display:
            return  # somebody changed it since; not ours to reset
        previous = self._restore_display
        if previous is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = previous

    def screenshot(self):
        """A capture of this display, independent of whatever is on the real one."""
        from .screen import ScreenCapture  # noqa: PLC0415

        previous = os.environ.get("DISPLAY")
        os.environ["DISPLAY"] = self.display
        capture = ScreenCapture()
        try:
            return capture.grab()
        finally:
            capture.close()
            if previous is None:
                os.environ.pop("DISPLAY", None)
            else:
                os.environ["DISPLAY"] = previous

    def __enter__(self) -> VirtualDisplay:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- internals ---------------------------------------------------------

    def _start_window_manager(self) -> None:
        """Without one, windows have no decorations and often never get focus."""
        for name, arguments in WINDOW_MANAGERS:
            path = shutil.which(name)
            if not path:
                continue
            try:
                self._wm_process = subprocess.Popen(  # noqa: S603
                    [path, *arguments],
                    env=self.env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=_die_with_parent,  # noqa: PLW1509 - see the function
                )
            except OSError:
                continue
            time.sleep(0.4)  # let it take the screen before anything is launched
            return


def _preferred_server() -> str:
    """Off-screen if we can, nested if that is all there is."""
    for name in SERVERS:
        if shutil.which(name):
            return name
    return ""


WATCH_TITLE = "LAI — the agent's screen · minimise this, it keeps working"
WATCH_CLASS = "lai-agent-screen"


def _server_command(server: str, display: str, size: tuple, depth: int) -> list[str]:
    geometry = f"{size[0]}x{size[1]}x{depth}"
    if server == "Xvfb":
        return ["Xvfb", display, "-screen", "0", geometry, "-nolisten", "tcp", "-noreset"]
    return [
        "Xephyr", display,
        "-screen", f"{size[0]}x{size[1]}",
        "-resizeable",
        "-nolisten", "tcp",
        # Without this Xephyr grabs your keyboard and mouse the moment the
        # pointer crosses into the window — which turns "I want to watch it"
        # into "it took my keyboard again". The whole point of a window you can
        # glance at is that glancing costs nothing.
        "-no-host-grab",
        # Draw the agent's pointer inside the window instead of borrowing the
        # host's, so you can see where it is actually clicking. It genuinely
        # has its own mouse; this is what makes that visible.
        "-sw-cursor",
        "-title", WATCH_TITLE,
        "-name", WATCH_CLASS,
    ]


def _die_with_parent() -> None:
    """Ask the kernel to kill this child when its parent dies.

    ``stop()`` handles the ordinary case, but it only runs on an orderly exit.
    Kill the process holding a display — Ctrl+C at the wrong moment, a SIGKILL,
    a crash — and the X server it started outlives it with nothing left that
    knows the number, forever. Six of them accumulated on this machine in an
    afternoon. PR_SET_PDEATHSIG closes that hole in the kernel, where it cannot
    be skipped.

    Best effort: on anything but Linux this is simply not available, and a
    leaked server is better than a failed launch.
    """
    try:
        import ctypes  # noqa: PLC0415
        import signal as signals  # noqa: PLC0415

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signals.SIGTERM)
    except Exception:
        pass


def _focused_window():
    """Whatever the human is typing into right now, if it can be determined."""
    try:
        from .windows import WindowManager  # noqa: PLC0415

        windows = WindowManager()
        try:
            active = windows.active_window()
            return active.id if active else None
        finally:
            windows.close()
    except Exception:
        return None


def _give_focus_back(window_id) -> None:
    """Hand focus back to where it was before the watch window appeared."""
    if window_id is None:
        return
    try:
        from .windows import WindowManager  # noqa: PLC0415

        windows = WindowManager()
        try:
            windows.focus(window_id)
        finally:
            windows.close()
    except Exception:
        pass  # a WM that refuses is not worth failing a run over


def _wait_for_display(display: str, process, timeout: float) -> None:
    """Block until the server answers, or say why it never will."""
    socket_path = f"/tmp/.X11-unix/X{display.lstrip(':').split('.')[0]}"  # noqa: S108
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BackendUnavailable(
                f"the X server for {display} exited immediately",
                detail="a nested server needs a running desktop to draw into; "
                "use Xvfb for a headless machine",
            )
        if os.path.exists(socket_path):
            time.sleep(0.3)  # the socket appears a moment before it accepts
            return
        time.sleep(0.1)
    raise BackendUnavailable(f"{display} did not come up within {timeout:.0f}s")
