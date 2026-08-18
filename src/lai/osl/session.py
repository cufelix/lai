"""X session discovery.

LAI is often started by something that does not hand down a graphical
environment: an MCP client (the MCP SDK deliberately passes a restricted env
that omits ``DISPLAY``), a systemd unit, a cron job, an ssh session. Without
``DISPLAY`` every OS-layer call fails with "no X11 session to attach to", which
is a confusing way to say "I was started without a desktop".

So: find the user's X session rather than giving up. Check the environment,
then the X11 socket directory, then the environment of a process that is
already talking to the display.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

X11_SOCKET_DIR = Path("/tmp/.X11-unix")  # noqa: S108 - the X server's real socket path, not a temp file we create
DISPLAY_PATTERN = re.compile(r"^X(\d+)$")
# Processes that reliably run inside a graphical session and hold its env.
SESSION_PROCESS_HINTS = (
    "cinnamon-session", "gnome-session", "xfce4-session", "mate-session",
    "plasmashell", "lxsession", "xfwm4", "mutter", "kwin_x11", "openbox",
    "cinnamon", "xterm", "Xorg",
)


def candidate_displays() -> list[str]:
    """Display names that look plausible on this machine, best first."""
    found: list[str] = []

    current = os.environ.get("DISPLAY", "").strip()
    if current:
        found.append(current)

    if X11_SOCKET_DIR.is_dir():
        numbers: list[int] = []
        try:
            for entry in X11_SOCKET_DIR.iterdir():
                match = DISPLAY_PATTERN.match(entry.name)
                if match:
                    numbers.append(int(match.group(1)))
        except OSError:
            pass
        for number in sorted(numbers):
            name = f":{number}"
            if name not in found:
                found.append(name)

    for name in _displays_from_processes():
        if name not in found:
            found.append(name)

    return found


def _displays_from_processes() -> list[str]:
    """Read DISPLAY out of a running session process owned by this user."""
    found: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    uid = os.getuid()
    try:
        entries = [p for p in proc.iterdir() if p.name.isdigit()]
    except OSError:
        return found

    for entry in entries:
        try:
            if entry.stat().st_uid != uid:
                continue
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
            if not any(hint in comm for hint in SESSION_PROCESS_HINTS):
                continue
            raw = (entry / "environ").read_bytes()
        except (OSError, PermissionError):
            continue
        for item in raw.split(b"\0"):
            if item.startswith(b"DISPLAY="):
                value = item[8:].decode("utf-8", errors="replace").strip()
                if value and value not in found:
                    found.append(value)
                break
    return found


def probe_display(name: str, *, timeout: float = 2.0) -> bool:
    """True if we can actually open this display."""
    try:
        from Xlib import display as xdisplay  # noqa: PLC0415
    except ImportError:
        return False
    try:
        connection = xdisplay.Display(name)
    except Exception:
        return False
    try:
        connection.screen()
        return True
    except Exception:
        return False
    finally:
        try:
            connection.close()
        except Exception:
            pass


def ensure_display(preferred: str = "") -> str:
    """Make sure ``DISPLAY`` (and ``XAUTHORITY``) point at a usable X session.

    Returns the resolved display name, or "" if none could be found. Sets the
    environment as a side effect so every downstream library — Xlib, mss, GTK,
    AT-SPI, xdotool — sees the same session.
    """
    _ensure_xauthority()

    candidates = ([preferred] if preferred else []) + candidate_displays()
    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        if probe_display(name):
            os.environ["DISPLAY"] = name
            return name

    # Nothing probed clean. Still set a best guess so the resulting error
    # message names a display instead of complaining that DISPLAY is unset.
    for name in candidates:
        if name:
            os.environ.setdefault("DISPLAY", name)
            return ""
    return ""


def _ensure_xauthority() -> None:
    """Point XAUTHORITY at the user's cookie file when it is not inherited."""
    if os.environ.get("XAUTHORITY"):
        return
    home = Path(os.path.expanduser("~"))
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    candidates = [home / ".Xauthority"]
    if runtime:
        candidates.insert(0, Path(runtime) / "gdm" / "Xauthority")
    for path in candidates:
        try:
            if path.is_file():
                os.environ["XAUTHORITY"] = str(path)
                return
        except OSError:
            continue


def session_info() -> dict:
    """Summary used by ``lai doctor`` and error messages."""
    return {
        "display": os.environ.get("DISPLAY", ""),
        "xauthority": os.environ.get("XAUTHORITY", ""),
        "session_type": os.environ.get("XDG_SESSION_TYPE", ""),
        "candidates": candidate_displays(),
    }
