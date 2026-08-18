"""Tests for lai.osl.notifications and the notification tools in
lai.tools.perception.

Everything DBus/GLib-related is monkeypatched: the point of these tests is
the parsing and fallback logic, not exercising a real notification daemon
(which would also pop a visible notification during CI). ``_notification_
from_args`` is pure and is tested directly against hand-built argument
lists, matching org.freedesktop.Notifications.Notify's signature. The
eavesdrop-message-filter path is exercised with a tiny fake DBus message
object rather than a real bus connection.
"""

from __future__ import annotations

import subprocess

import pytest

from lai.config import Config
from lai.osl.notifications import (
    Notification,
    NotificationMonitor,
    _notification_from_args,
    _send_via_libnotify,
    _send_via_notify_send,
    send_notification,
)
from lai.tools.base import ToolContext, ToolRegistry
from lai.tools.perception import register as register_perception


def context(**kwargs) -> ToolContext:
    kwargs.setdefault("config", Config())
    kwargs.setdefault("extra", {})
    return ToolContext(**kwargs)


# -- Notification dataclass ------------------------------------------------------


def test_notification_to_dict_shape():
    note = Notification(app_name="Thunderbird", summary="New mail", body="from x", id=1, at=100.0, urgency="normal")
    assert note.to_dict() == {
        "app_name": "Thunderbird", "summary": "New mail", "body": "from x",
        "id": 1, "at": 100.0, "urgency": "normal",
    }


# -- _notification_from_args: pure parsing of the Notify() call signature ------------------------------------------------------


def test_notification_from_args_basic_fields():
    args = ["Thunderbird", 0, "icon", "New mail", "from x", [], {}, 5000]
    note = _notification_from_args(args, id=7)
    assert note.app_name == "Thunderbird"
    assert note.summary == "New mail"
    assert note.body == "from x"
    assert note.id == 7
    assert note.urgency == "normal"


@pytest.mark.parametrize("value,expected", [(0, "low"), (1, "normal"), (2, "critical")])
def test_notification_from_args_maps_urgency_hint(value, expected):
    args = ["App", 0, "", "s", "b", [], {"urgency": value}, 0]
    assert _notification_from_args(args, id=1).urgency == expected


def test_notification_from_args_unknown_urgency_defaults_to_normal():
    args = ["App", 0, "", "s", "b", [], {"urgency": 99}, 0]
    assert _notification_from_args(args, id=1).urgency == "normal"


def test_notification_from_args_missing_hints_defaults_to_normal():
    args = ["App", 0, "", "s", "b"]
    assert _notification_from_args(args, id=1).urgency == "normal"


def test_notification_from_args_tolerates_short_arg_lists():
    note = _notification_from_args([], id=1)
    assert note.app_name == "" and note.summary == "" and note.body == ""


def test_notification_from_args_non_dict_hints_are_ignored():
    args = ["App", 0, "", "s", "b", [], "not-a-dict", 0]
    assert _notification_from_args(args, id=1).urgency == "normal"


# -- NotificationMonitor.available / probing ------------------------------------------------------


def test_available_false_when_probe_fails(monkeypatch):
    monkeypatch.setattr(NotificationMonitor, "_probe", staticmethod(lambda: False))
    monitor = NotificationMonitor()
    assert monitor.available is False


def test_available_true_when_probe_succeeds(monkeypatch):
    monkeypatch.setattr(NotificationMonitor, "_probe", staticmethod(lambda: True))
    monitor = NotificationMonitor()
    assert monitor.available is True


def test_available_is_cached_after_first_check(monkeypatch):
    calls = []
    monkeypatch.setattr(NotificationMonitor, "_probe", staticmethod(lambda: calls.append(1) or True))
    monitor = NotificationMonitor()
    assert monitor.available and monitor.available
    assert len(calls) == 1


def test_start_returns_false_without_raising_when_unavailable(monkeypatch):
    monkeypatch.setattr(NotificationMonitor, "_probe", staticmethod(lambda: False))
    monitor = NotificationMonitor()
    assert monitor.start() is False


# -- NotificationMonitor._on_message: the eavesdrop message filter ------------------------------------------------------


class _FakeMessage:
    def __init__(self, interface: str, member: str, args: list):
        self._interface, self._member, self._args = interface, member, args

    def get_interface(self) -> str:
        return self._interface

    def get_member(self) -> str:
        return self._member

    def get_args_list(self) -> list:
        return self._args


def test_on_message_captures_a_matching_notify_call():
    monitor = NotificationMonitor()
    message = _FakeMessage(
        "org.freedesktop.Notifications", "Notify", ["App", 0, "", "Summary", "Body", [], {}, 0]
    )
    monitor._on_message(None, message)
    captured = monitor.recent()
    assert len(captured) == 1
    assert captured[0].summary == "Summary"


def test_on_message_ignores_other_members():
    monitor = NotificationMonitor()
    message = _FakeMessage("org.freedesktop.Notifications", "CloseNotification", [1])
    monitor._on_message(None, message)
    assert monitor.recent() == []


def test_on_message_ignores_other_interfaces():
    monitor = NotificationMonitor()
    message = _FakeMessage("org.freedesktop.DBus", "Notify", ["x"])
    monitor._on_message(None, message)
    assert monitor.recent() == []


def test_on_message_never_raises_on_malformed_args():
    monitor = NotificationMonitor()
    message = _FakeMessage("org.freedesktop.Notifications", "Notify", None)  # get_args_list() -> TypeError
    monitor._on_message(None, message)  # must not raise
    assert monitor.recent() == []


def test_on_message_invokes_the_callback():
    seen = []
    monitor = NotificationMonitor()
    monitor._callback = seen.append
    message = _FakeMessage("org.freedesktop.Notifications", "Notify", ["App", 0, "", "s", "b", [], {}, 0])
    monitor._on_message(None, message)
    assert len(seen) == 1 and seen[0].summary == "s"


def test_on_message_survives_a_raising_callback():
    def boom(_note):
        raise RuntimeError("callback exploded")

    monitor = NotificationMonitor()
    monitor._callback = boom
    message = _FakeMessage("org.freedesktop.Notifications", "Notify", ["App", 0, "", "s", "b", [], {}, 0])
    monitor._on_message(None, message)  # must not raise
    assert len(monitor.recent()) == 1


# -- NotificationMonitor.recent: ordering and bounding ------------------------------------------------------


def test_recent_returns_newest_first():
    monitor = NotificationMonitor()
    for i in range(3):
        message = _FakeMessage("org.freedesktop.Notifications", "Notify", [f"app{i}", 0, "", f"s{i}", "", [], {}, 0])
        monitor._on_message(None, message)
    assert [n.summary for n in monitor.recent()] == ["s2", "s1", "s0"]


def test_recent_respects_the_limit():
    monitor = NotificationMonitor()
    for i in range(5):
        message = _FakeMessage("org.freedesktop.Notifications", "Notify", ["app", 0, "", f"s{i}", "", [], {}, 0])
        monitor._on_message(None, message)
    assert len(monitor.recent(limit=2)) == 2


def test_history_is_bounded_by_history_size():
    monitor = NotificationMonitor(history=3)
    for i in range(10):
        message = _FakeMessage("org.freedesktop.Notifications", "Notify", ["app", 0, "", f"s{i}", "", [], {}, 0])
        monitor._on_message(None, message)
    assert len(monitor.recent(limit=100)) == 3
    assert [n.summary for n in monitor.recent(limit=100)] == ["s9", "s8", "s7"]


# -- send_notification: libnotify then notify-send fallback ------------------------------------------------------


def test_send_via_libnotify_returns_false_when_gi_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gi":
            raise ImportError("no gi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _send_via_libnotify("s", "b", "normal", "", 1000, "LAI") is False


def test_send_via_notify_send_returns_false_without_the_binary(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _send_via_notify_send("s", "b", "normal", "", 1000) is False


def test_send_via_notify_send_builds_the_expected_args(monkeypatch):
    seen = {}

    def fake_which(name):
        return "/usr/bin/notify-send"

    def fake_run(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)
    ok = _send_via_notify_send("Title", "Body", "critical", "dialog-info", 3000)
    assert ok is True
    args = seen["args"]
    assert args[0] == "/usr/bin/notify-send"
    assert "-u" in args and "critical" in args
    assert "-t" in args and "3000" in args
    assert "-i" in args and "dialog-info" in args
    assert args[-2:] == ["Title", "Body"]


def test_send_via_notify_send_false_on_timeout(monkeypatch):
    def fake_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="notify-send", timeout=5)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/notify-send")
    monkeypatch.setattr("subprocess.run", fake_timeout)
    assert _send_via_notify_send("s", "b", "normal", "", 1000) is False


def test_send_notification_falls_back_to_notify_send_when_libnotify_fails(monkeypatch):
    monkeypatch.setattr("lai.osl.notifications._send_via_libnotify", lambda *a, **k: False)
    monkeypatch.setattr("lai.osl.notifications._send_via_notify_send", lambda *a, **k: True)
    assert send_notification("hi") is True


def test_send_notification_prefers_libnotify_when_it_works(monkeypatch):
    calls = []
    monkeypatch.setattr("lai.osl.notifications._send_via_libnotify", lambda *a, **k: calls.append("libnotify") or True)
    monkeypatch.setattr(
        "lai.osl.notifications._send_via_notify_send", lambda *a, **k: calls.append("notify-send") or True
    )
    assert send_notification("hi") is True
    assert calls == ["libnotify"]


def test_send_notification_returns_false_when_both_paths_fail(monkeypatch):
    monkeypatch.setattr("lai.osl.notifications._send_via_libnotify", lambda *a, **k: False)
    monkeypatch.setattr("lai.osl.notifications._send_via_notify_send", lambda *a, **k: False)
    assert send_notification("hi") is False


# -- tools: notify_user / notifications_recent ------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_perception(reg)
    return reg


def test_notify_user_tool_success(monkeypatch, registry):
    monkeypatch.setattr("lai.tools.perception.send_notification", lambda *a, **k: True)
    result = registry.call("notify_user", {"summary": "Build finished"}, context())
    assert result.ok is True
    assert "Build finished" in result.content


def test_notify_user_tool_failure_path_is_clear_not_a_crash(monkeypatch, registry):
    monkeypatch.setattr("lai.tools.perception.send_notification", lambda *a, **k: False)
    result = registry.call("notify_user", {"summary": "hi"}, context())
    assert result.ok is False
    assert "notification" in result.content.lower()


def test_notifications_recent_tool_failure_path_when_monitor_unavailable(monkeypatch, registry):
    monkeypatch.setattr(NotificationMonitor, "_probe", staticmethod(lambda: False))
    result = registry.call("notifications_recent", {}, context())
    assert result.ok is False
    assert "unavailable" in result.content.lower()


def test_notifications_recent_tool_reports_captured_notifications(registry):
    ctx = context()
    monitor = NotificationMonitor()
    monitor._on_message(
        None, _FakeMessage("org.freedesktop.Notifications", "Notify", ["App", 0, "", "Hi", "there", [], {}, 0])
    )
    monkeypatch_monitor = monitor
    monkeypatch_monitor.start = lambda callback=None: True  # already "running"
    ctx.extra["notification_monitor"] = monkeypatch_monitor

    result = registry.call("notifications_recent", {"limit": 5}, ctx)
    assert result.ok is True
    assert "Hi" in result.content
    assert result.data["notifications"][0]["summary"] == "Hi"


def test_notifications_recent_tool_empty_but_ok_when_nothing_seen(registry):
    ctx = context()
    monitor = NotificationMonitor()
    monitor.start = lambda callback=None: True
    ctx.extra["notification_monitor"] = monitor
    result = registry.call("notifications_recent", {}, ctx)
    assert result.ok is True
    assert "No notifications" in result.content
