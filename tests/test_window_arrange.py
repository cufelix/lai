"""Moving and resizing windows honestly.

Two failures hid here, both found by giving the agent a real task on a real
desktop. A window manager acknowledges a geometry change immediately and
finishes it several frames later, so reading the window straight away reports
a shape it is halfway out of. And the accessibility tree carries the toolkit's
idea of where the window is, which lags an X11 move — clip hard to the window
rectangle and buttons the user can plainly see disappear from the tree.
"""

from __future__ import annotations

import pytest

from lai.osl.desktop import CLIP_MARGIN, _clip_region
from lai.osl.geometry import Rect
from lai.tools.window import ARRANGE_SLACK, _arrange_summary


class FakeWindow:
    def __init__(self, bounds, states=(), title="Calculator"):
        self.bounds = bounds
        self.states = states
        self.title = title


# -- reporting what actually happened ------------------------------------


def test_a_granted_request_is_reported_plainly():
    wanted = Rect(120, 140, 600, 480)
    got = FakeWindow(Rect(120, 140, 600, 480))
    assert "Moved" in _arrange_summary(FakeWindow(wanted), wanted, got)


def test_decoration_sized_differences_are_not_worth_mentioning():
    wanted = Rect(120, 140, 600, 480)
    got = FakeWindow(Rect(120 + ARRANGE_SLACK, 140, 600, 480 - ARRANGE_SLACK))
    assert "Moved" in _arrange_summary(FakeWindow(wanted), wanted, got)


def test_a_refused_resize_says_so_and_gives_the_real_bounds():
    """Reporting the requested size is how an agent clicks where nothing is."""
    wanted = Rect(120, 140, 600, 480)
    got = FakeWindow(Rect(172, 186, 720, 972))
    summary = _arrange_summary(FakeWindow(wanted), wanted, got)
    assert "not what was asked for" in summary
    assert "width 600 → 720" in summary and "height 480 → 972" in summary
    assert "minimum size" in summary
    assert "Use these bounds" in summary


def test_a_maximized_window_is_told_why_it_ignored_the_geometry():
    wanted = Rect(0, 0, 400, 300)
    got = FakeWindow(Rect(0, 0, 2880, 1720), states=("maximized_horz", "maximized_vert"))
    assert "maximized or fullscreen" in _arrange_summary(FakeWindow(wanted), wanted, got)


# -- clipping the accessibility tree -------------------------------------


def test_the_clip_region_is_generous_about_a_lagging_toolkit():
    """AT-SPI geometry lags an X11 move; a tight clip deletes real buttons."""
    region = _clip_region(FakeWindow(Rect(172, 186, 720, 972)))
    assert region.x == 172 - CLIP_MARGIN
    assert region.y == 186 - CLIP_MARGIN
    assert region.width == 720 + CLIP_MARGIN * 2
    assert region.height == 972 + CLIP_MARGIN * 2


def test_an_element_at_the_windows_pre_move_position_survives_the_clip():
    """The exact case observed: the window moved, the tree had not caught up."""
    before, after = Rect(0, 0, 720, 972), Rect(172, 186, 720, 972)
    stale_button = Rect(44, 308, 60, 40)  # '7', still reported at the old origin
    assert stale_button.intersects(before)
    assert not stale_button.intersects(after), "which is why a tight clip dropped it"
    assert stale_button.intersects(_clip_region(FakeWindow(after)))


# -- settling ------------------------------------------------------------


class FakeManager:
    """A window manager that takes a few polls to finish a move."""

    def __init__(self, readings):
        self.readings = list(readings)
        self.calls = 0

    def get(self, window_id):
        self.calls += 1
        index = min(self.calls - 1, len(self.readings) - 1)
        return FakeWindow(self.readings[index])


def test_settle_waits_for_the_motion_to_stop(monkeypatch):
    from lai.osl.windows import WindowManager

    manager = FakeManager([
        Rect(0, 0, 720, 972),
        Rect(90, 90, 720, 972),
        Rect(172, 186, 720, 972),
        Rect(172, 186, 720, 972),
    ])
    monkeypatch.setattr("lai.osl.windows.time.sleep", lambda seconds: None)
    settled = WindowManager.settle(manager, 1, timeout=5.0, poll=0.0)
    assert settled.bounds == Rect(172, 186, 720, 972)
    assert manager.calls == 4, "it should stop as soon as two readings agree"


def test_settle_gives_up_rather_than_hanging_on_a_restless_window(monkeypatch):
    from itertools import count

    from lai.osl.windows import WindowManager

    class Restless:
        def __init__(self):
            self.counter = count()

        def get(self, window_id):
            step = next(self.counter)
            return FakeWindow(Rect(step, step, 100, 100))

    ticks = iter([0.0, 0.1, 0.2, 99.0, 99.0])
    monkeypatch.setattr("lai.osl.windows.time.sleep", lambda seconds: None)
    monkeypatch.setattr("lai.osl.windows.time.monotonic", lambda: next(ticks, 99.0))
    result = WindowManager.settle(Restless(), 1, timeout=1.0, poll=0.0)
    assert result is not None, "a window that never stops must not hang the agent"


@pytest.mark.parametrize("states", [(), ("maximized_horz",)])
def test_settle_notices_a_state_change_too(monkeypatch, states):
    from lai.osl.windows import WindowManager

    class Changing:
        def __init__(self):
            self.calls = 0

        def get(self, window_id):
            self.calls += 1
            return FakeWindow(Rect(0, 0, 100, 100), states=states if self.calls > 1 else ())

    monkeypatch.setattr("lai.osl.windows.time.sleep", lambda seconds: None)
    manager = Changing()
    WindowManager.settle(manager, 1, timeout=5.0, poll=0.0)
    assert manager.calls >= 2
