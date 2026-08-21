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

        self._server_process = subprocess.Popen(  # noqa: S603
            _server_command(chosen, self.display, self.size, self.depth),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._started = True
        try:
            _wait_for_display(self.display, self._server_process, STARTUP_TIMEOUT)
        except Exception:
            self.stop()
            raise

        if window_manager:
            self._start_window_manager()
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


def _server_command(server: str, display: str, size: tuple, depth: int) -> list[str]:
    geometry = f"{size[0]}x{size[1]}x{depth}"
    if server == "Xvfb":
        return ["Xvfb", display, "-screen", "0", geometry, "-nolisten", "tcp", "-noreset"]
    return ["Xephyr", display, "-screen", f"{size[0]}x{size[1]}", "-resizeable", "-nolisten", "tcp"]


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
