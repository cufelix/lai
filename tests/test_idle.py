"""Tests for lai.osl.idle and the user_idle tool in lai.tools.perception.

Pure tests fake the MIT-SCREEN-SAVER reply directly (a tiny stand-in with an
``.idle`` attribute in milliseconds, exactly what ``screensaver_query_info()``
returns) so idle-seconds conversion, the active/idle threshold and
wait_for_idle's polling loop are all exercised without an X11 connection.
The @pytest.mark.x11 test checks the real extension against the live
display named in this environment (``d.has_extension('MIT-SCREEN-SAVER')``
is known to be True there).
"""

from __future__ import annotations

import time

import pytest

from lai.config import Config
from lai.errors import BackendUnavailable
from lai.osl.geometry import Point
from lai.osl.idle import DEFAULT_ACTIVE_THRESHOLD, IdleMonitor, IdleState
from lai.tools.base import ToolContext, ToolRegistry
from lai.tools.perception import register as register_perception


def context(**kwargs) -> ToolContext:
    kwargs.setdefault("config", Config())
    kwargs.setdefault("extra", {})
    return ToolContext(**kwargs)


class _FakeQueryInfo:
    def __init__(self, idle_ms: int) -> None:
        self.idle = idle_ms


class _FakeRoot:
    def __init__(self, idle_ms: int = 0) -> None:
        self.idle_ms = idle_ms
        self.pointer = (0, 0)

    def screensaver_query_info(self) -> _FakeQueryInfo:
        return _FakeQueryInfo(self.idle_ms)

    def query_pointer(self):
        x, y = self.pointer

        class _Pointer:
            root_x, root_y = x, y

        return _Pointer()


def _wired_monitor(idle_ms: int = 0) -> tuple[IdleMonitor, _FakeRoot]:
    """An IdleMonitor whose extension check is pre-satisfied, bypassing _conn()."""
    monitor = IdleMonitor()
    root = _FakeRoot(idle_ms)
    monitor._root = root
    monitor._ext_ok = True
    return monitor, root


# -- IdleState ------------------------------------------------------


def test_idle_state_to_dict_rounds_seconds():
    state = IdleState(idle_seconds=12.3456, active=False, threshold=3.0)
    assert state.to_dict() == {"idle_seconds": 12.35, "active": False, "threshold": 3.0}


# -- idle_seconds: milliseconds -> seconds conversion ------------------------------------------------------


@pytest.mark.parametrize("idle_ms,expected", [(0, 0.0), (1500, 1.5), (25432, 25.432), (60000, 60.0)])
def test_idle_seconds_converts_milliseconds(idle_ms, expected):
    monitor, _root = _wired_monitor(idle_ms)
    assert monitor.idle_seconds() == pytest.approx(expected)


def test_idle_seconds_raises_when_extension_unavailable():
    monitor = IdleMonitor()
    monitor._ext_ok = False
    with pytest.raises(BackendUnavailable, match="MIT-SCREEN-SAVER"):
        monitor.idle_seconds()


# -- user_active / state ------------------------------------------------------


def test_user_active_true_below_threshold():
    monitor, _root = _wired_monitor(idle_ms=500)
    assert monitor.user_active(threshold=3.0) is True


def test_user_active_false_above_threshold():
    monitor, _root = _wired_monitor(idle_ms=5000)
    assert monitor.user_active(threshold=3.0) is False


def test_user_active_uses_the_default_threshold_when_not_given():
    monitor, _root = _wired_monitor(idle_ms=int(DEFAULT_ACTIVE_THRESHOLD * 1000) + 500)
    assert monitor.user_active() is False


def test_state_reports_idle_seconds_and_active_flag_consistently():
    monitor, _root = _wired_monitor(idle_ms=10_000)
    state = monitor.state(threshold=3.0)
    assert state.idle_seconds == pytest.approx(10.0)
    assert state.active is False
    assert state.threshold == 3.0


# -- available: caching and graceful failure ------------------------------------------------------


def test_available_is_false_when_conn_raises(monkeypatch):
    monitor = IdleMonitor()
    monkeypatch.setattr(monitor, "_conn", lambda: (_ for _ in ()).throw(BackendUnavailable("no xlib")))
    assert monitor.available is False


def test_available_is_cached(monkeypatch):
    monitor = IdleMonitor()
    calls = []

    class _FakeDisplay:
        def has_extension(self, _name: str) -> bool:
            return True

    def fake_conn():
        calls.append(1)
        return _FakeDisplay()

    monkeypatch.setattr(monitor, "_conn", fake_conn)
    assert monitor.available and monitor.available
    assert len(calls) == 1


# -- wait_for_idle ------------------------------------------------------


def test_wait_for_idle_returns_true_once_threshold_reached(monkeypatch):
    monitor, _root = _wired_monitor(idle_ms=0)
    # Simulate the user going idle mid-poll without any real sleeping.
    readings = iter([0.0, 1.0, 4.0])
    monkeypatch.setattr(monitor, "idle_seconds", lambda: next(readings))
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    assert monitor.wait_for_idle(3.0, timeout=5.0, poll=0.0) is True


def test_wait_for_idle_returns_false_on_timeout(monkeypatch):
    monitor, _root = _wired_monitor(idle_ms=0)
    monkeypatch.setattr(monitor, "idle_seconds", lambda: 0.0)
    deadline = {"t": 0.0}

    def fake_monotonic():
        deadline["t"] += 0.05
        return deadline["t"]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    assert monitor.wait_for_idle(3.0, timeout=0.1, poll=0.0) is False


# -- pointer_moved_by_user ------------------------------------------------------


def test_pointer_moved_by_user_false_within_tolerance(monkeypatch):
    monitor, root = _wired_monitor()
    root.pointer = (101, 99)
    monkeypatch.setattr(monitor, "_conn", lambda: None)
    assert monitor.pointer_moved_by_user(Point(100, 100), tolerance=3) is False


def test_pointer_moved_by_user_true_outside_tolerance(monkeypatch):
    monitor, root = _wired_monitor()
    root.pointer = (200, 100)
    monkeypatch.setattr(monitor, "_conn", lambda: None)
    assert monitor.pointer_moved_by_user(Point(100, 100), tolerance=3) is True


# -- user_idle tool ------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_perception(reg)
    return reg


def test_user_idle_tool_reports_state(registry):
    ctx = context()
    monitor, _root = _wired_monitor(idle_ms=7000)
    ctx.extra["idle_monitor"] = monitor
    result = registry.call("user_idle", {"threshold": 3}, ctx)
    assert result.ok is True
    assert result.data["idle_seconds"] == pytest.approx(7.0)
    assert result.data["active"] is False
    assert "idle" in result.content.lower()


def test_user_idle_tool_failure_path_is_clear_not_a_crash(registry):
    ctx = context()
    monitor = IdleMonitor()
    monitor._ext_ok = False
    ctx.extra["idle_monitor"] = monitor
    result = registry.call("user_idle", {}, ctx)
    assert result.ok is False
    assert "MIT-SCREEN-SAVER" in result.content


# -- x11: real display ------------------------------------------------------


@pytest.mark.x11
def test_real_idle_monitor_reports_a_non_negative_idle_time():
    monitor = IdleMonitor()
    assert monitor.available is True
    seconds = monitor.idle_seconds()
    assert seconds >= 0
