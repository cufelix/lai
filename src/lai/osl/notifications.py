"""Desktop notifications: reading them and sending them.

Two capabilities share this module because they are two directions of the
same channel:

* :class:`NotificationMonitor` watches ``org.freedesktop.Notifications`` on
  the DBus session bus so the agent can notice what the desktop is telling
  the human — a build finished, a message arrived, a battery warning fired —
  without polling application state itself.
* :func:`send_notification` lets the agent tell the human something back,
  without stealing keyboard/mouse focus the way a dialog box would.

``NotificationMonitor`` *eavesdrops* rather than acting as the notification
daemon: being the daemon would mean reimplementing popup rendering and would
break every app that already expects a real notification to appear. An
eavesdropping DBus match rule (``eavesdrop=true``) lets us observe ``Notify``
calls in transit without intercepting them — the real daemon still renders
the popup exactly as before. Some distributions' DBus policy blocks
eavesdropping for unprivileged processes; when the match rule is refused (or
DBus itself is unreachable), this degrades to ``available = False`` rather
than raising, because "no notifications observed" must never be confused
with "the process crashed".
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

MATCH_RULE = "eavesdrop=true,interface='org.freedesktop.Notifications',member='Notify'"
DEFAULT_HISTORY = 200
START_TIMEOUT = 3.0
STOP_TIMEOUT = 2.0
NOTIFY_SEND_TIMEOUT = 5.0

# org.freedesktop.Notifications hint values -> human-readable urgency.
_URGENCY_NAMES = {0: "low", 1: "normal", 2: "critical"}
_URGENCY_VALUES = {"low": 0, "normal": 1, "critical": 2}


@dataclass(frozen=True, slots=True)
class Notification:
    """One observed (or sent) desktop notification."""

    app_name: str
    summary: str
    body: str
    id: int
    at: float
    urgency: str = "normal"

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "summary": self.summary,
            "body": self.body,
            "id": self.id,
            "at": self.at,
            "urgency": self.urgency,
        }


def _notification_from_args(args: list, *, id: int) -> Notification:
    """Build a :class:`Notification` from a captured ``Notify`` call's args.

    Pure and dbus-free on purpose, so the parsing logic can be unit tested
    without a live bus or a real ``dbus.lowlevel.Message``. Signature per the
    spec: ``Notify(app_name, replaces_id, app_icon, summary, body, actions,
    hints, expire_timeout)``. ``id`` is our own capture-sequence number, not
    the id the daemon eventually returns — eavesdropping the method *call*
    never sees its return value, so we cannot know the daemon-assigned id.
    """
    app_name = str(args[0]) if len(args) > 0 else ""
    summary = str(args[3]) if len(args) > 3 else ""
    body = str(args[4]) if len(args) > 4 else ""
    hints = args[6] if len(args) > 6 and isinstance(args[6], dict) else {}
    urgency_raw = hints.get("urgency")
    urgency = "normal"
    if urgency_raw is not None:
        try:
            urgency = _URGENCY_NAMES.get(int(urgency_raw), "normal")
        except (TypeError, ValueError):
            urgency = "normal"
    return Notification(app_name=app_name, summary=summary, body=body, id=id, at=time.time(), urgency=urgency)


class NotificationMonitor:
    """Eavesdrops on ``org.freedesktop.Notifications.Notify`` calls."""

    def __init__(self, *, history: int = DEFAULT_HISTORY) -> None:
        self._history: deque[Notification] = deque(maxlen=history)
        self._callback: Callable[[Notification], None] | None = None
        self._loop = None
        self._thread: threading.Thread | None = None
        self._next_id = 1
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._probe()
        return self._available

    @staticmethod
    def _probe() -> bool:
        try:
            import dbus  # noqa: PLC0415
        except ImportError:
            return False
        try:
            dbus.SessionBus()
        except Exception:
            return False
        return True

    def start(self, callback: Callable[[Notification], None] | None = None) -> bool:
        """Start listening in a daemon thread. Never raises — returns False
        (and leaves ``available`` False) if eavesdropping cannot be set up."""
        if self._thread is not None:
            return True
        if not self.available:
            return False

        self._callback = callback
        ready = threading.Event()
        outcome = {"ok": False}

        def run() -> None:
            try:
                import dbus  # noqa: PLC0415
                import dbus.mainloop.glib  # noqa: PLC0415
                from gi.repository import GLib  # noqa: PLC0415

                dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
                bus = dbus.SessionBus()
                bus.add_match_string(MATCH_RULE)
                bus.add_message_filter(self._on_message)
                self._loop = GLib.MainLoop()
                outcome["ok"] = True
            except Exception:
                outcome["ok"] = False
            finally:
                ready.set()
            if outcome["ok"]:
                try:
                    self._loop.run()
                except Exception:
                    pass

        self._thread = threading.Thread(target=run, daemon=True, name="lai-notification-monitor")
        self._thread.start()
        ready.wait(timeout=START_TIMEOUT)
        if not outcome["ok"]:
            self._thread = None
            self._available = False
            return False
        return True

    def _on_message(self, bus, message) -> None:
        try:
            if message.get_interface() != "org.freedesktop.Notifications" or message.get_member() != "Notify":
                return
            note = _notification_from_args(list(message.get_args_list()), id=self._next_id)
        except Exception:
            return
        self._next_id += 1
        self._history.append(note)
        if self._callback is not None:
            try:
                self._callback(note)
            except Exception:
                pass  # a caller's callback must never take down the mainloop

    def stop(self) -> None:
        if self._loop is not None:
            try:
                self._loop.quit()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=STOP_TIMEOUT)
        self._thread = None
        self._loop = None

    def recent(self, limit: int = 20) -> list[Notification]:
        """Most recently observed notifications first, bounded to ``limit``."""
        items = list(self._history)[-limit:]
        return list(reversed(items))


def _send_via_libnotify(summary: str, body: str, urgency: str, icon: str, timeout_ms: int, app_name: str) -> bool:
    try:
        import gi  # noqa: PLC0415

        gi.require_version("Notify", "0.7")
        from gi.repository import Notify  # noqa: PLC0415
    except Exception:
        return False

    try:
        if not Notify.is_initted() and not Notify.init(app_name):
            return False
        note = Notify.Notification.new(summary, body, icon or None)
        urgency_enum = {
            "low": Notify.Urgency.LOW,
            "normal": Notify.Urgency.NORMAL,
            "critical": Notify.Urgency.CRITICAL,
        }.get(urgency, Notify.Urgency.NORMAL)
        note.set_urgency(urgency_enum)
        note.set_timeout(int(timeout_ms))
        return bool(note.show())
    except Exception:
        return False


def _send_via_notify_send(summary: str, body: str, urgency: str, icon: str, timeout_ms: int) -> bool:
    binary = shutil.which("notify-send")
    if not binary:
        return False
    args = [
        binary,
        "-u", urgency if urgency in _URGENCY_VALUES else "normal",
        "-t", str(int(timeout_ms)),
    ]
    if icon:
        args += ["-i", icon]
    args += [summary, body]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=NOTIFY_SEND_TIMEOUT, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def send_notification(
    summary: str,
    body: str = "",
    *,
    urgency: str = "normal",
    icon: str = "",
    timeout_ms: int = 5000,
    app_name: str = "LAI",
) -> bool:
    """Show a desktop notification without stealing keyboard/mouse focus.

    Tries libnotify (``gi.repository.Notify``) first; if that is unavailable
    or ``Notify.init`` fails (no notification daemon registered, missing gi
    typelib, ...) falls back to the ``notify-send`` binary, which talks to
    the same daemon without requiring the gi bindings. Returns ``False``
    rather than raising if neither path works — a missing notification
    daemon should not be able to crash the agent loop.
    """
    if _send_via_libnotify(summary, body, urgency, icon, timeout_ms, app_name):
        return True
    return _send_via_notify_send(summary, body, urgency, icon, timeout_ms)
