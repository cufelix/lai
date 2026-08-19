"""Standing aside while the human is using their own machine.

Two hands on one mouse does not split the work: the agent's click lands in
whatever window the human just switched to. So the agent waits — but only for
the tools that actually move the mouse, only for a bounded time, and never on
a machine that cannot tell whether anyone is there.
"""

from __future__ import annotations

import pytest

from lai.safety.yielding import POLL_INTERVAL, Yielded, wait_for_the_human


class Clock:
    """A hand-cranked clock, so waiting is instant and exact in tests."""

    def __init__(self):
        self.now = 0.0

    def sleep(self, seconds):
        self.now += seconds

    def time(self):
        return self.now


def test_an_idle_human_costs_no_wait():
    clock = Clock()
    waited = wait_for_the_human(lambda: 30.0, settle=4.0, sleep=clock.sleep, now=clock.time)
    assert waited == 0.0


def test_it_waits_until_the_human_stops():
    clock = Clock()
    readings = iter([0.5, 1.0, 2.0, 9.0])

    waited = wait_for_the_human(
        lambda: next(readings), settle=4.0, sleep=clock.sleep, now=clock.time
    )
    assert waited == pytest.approx(3 * POLL_INTERVAL)


def test_it_gives_up_rather_than_hanging_forever():
    """A human who simply keeps working must not stall the run until timeout."""
    clock = Clock()
    with pytest.raises(Yielded) as info:
        wait_for_the_human(lambda: 0.0, settle=4.0, limit=10.0, sleep=clock.sleep, now=clock.time)
    assert info.value.waited >= 10.0
    assert "the desktop is yours" in str(info.value)
    assert "yield_to_user" in str(info.value), "and how to turn it off"


def test_a_machine_that_cannot_tell_is_not_blocked():
    """No idle extension must mean an agent that works, not one that refuses."""

    def broken():
        raise RuntimeError("MIT-SCREEN-SAVER is not available")

    assert wait_for_the_human(broken, settle=4.0) == 0.0


def test_the_wait_is_announced_once_not_every_poll():
    clock = Clock()
    readings = iter([0.0, 0.0, 0.0, 9.0])
    seen: list = []
    wait_for_the_human(
        lambda: next(readings), settle=4.0, on_wait=seen.append,
        sleep=clock.sleep, now=clock.time,
    )
    assert len(seen) == 1, "a status line, not a stream"


# -- the loop only waits for tools that move the mouse -------------------


def _agent(monkeypatch, tmp_path, *, risk, idle, yield_to_user=True):
    from dataclasses import replace

    from lai.agent.loop import Agent
    from lai.config import load_config
    from lai.safety.policy import Risk

    config = load_config().with_overrides(home=tmp_path)
    config = config.with_overrides(
        safety=replace(config.safety, yield_to_user=yield_to_user,
                       user_idle_seconds=4.0, max_yield_seconds=1.0)
    )
    spec = type("S", (), {"name": "t", "risk": risk})()

    agent = Agent.__new__(Agent)
    agent.config = config
    agent.registry = type("R", (), {"get": lambda self, name: spec})()
    agent.audit = type("A", (), {"write": lambda self, *a, **k: None})()
    agent.events: list = []
    agent._emit = lambda kind, payload: agent.events.append((kind, payload))
    agent._idle_probe = idle
    return agent, Risk


def test_an_input_tool_waits_for_the_human(tmp_path, monkeypatch):
    from lai.safety.policy import Risk

    readings = iter([0.0, 9.0, 9.0, 9.0])
    agent, _ = _agent(monkeypatch, tmp_path, risk=Risk.INPUT, idle=lambda: next(readings))
    monkeypatch.setattr("lai.safety.yielding.time.sleep", lambda s: None)

    agent._yield_to_human("computer_click")
    kinds = [kind for kind, _ in agent.events]
    assert "yielding" in kinds and "resumed" in kinds


def test_reading_the_screen_never_waits(tmp_path, monkeypatch):
    """An agent that cannot even look at a desktop in use is useless."""
    from lai.safety.policy import Risk

    def busy():
        raise AssertionError("a read must not consult the idle monitor")

    agent, _ = _agent(monkeypatch, tmp_path, risk=Risk.READ, idle=busy)
    agent._yield_to_human("computer_screenshot")
    assert agent.events == []


def test_the_feature_can_be_turned_off(tmp_path, monkeypatch):
    from lai.safety.policy import Risk

    def busy():
        raise AssertionError("yielding is off; the monitor must not be consulted")

    agent, _ = _agent(monkeypatch, tmp_path, risk=Risk.INPUT, idle=busy, yield_to_user=False)
    agent._yield_to_human("computer_click")
    assert agent.events == []


def test_a_human_who_keeps_working_ends_the_turn(tmp_path, monkeypatch):
    from lai.safety.policy import Risk

    agent, _ = _agent(monkeypatch, tmp_path, risk=Risk.INPUT, idle=lambda: 0.0)
    monkeypatch.setattr("lai.safety.yielding.time.sleep", lambda s: None)
    with pytest.raises(Yielded):
        agent._yield_to_human("computer_click")


def test_a_setting_survives_the_round_trip(tmp_path, monkeypatch):
    from lai.config import load_config

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        "[safety]\nyield_to_user = false\nuser_idle_seconds = 9.5\n", encoding="utf-8"
    )
    safety = load_config().safety
    assert safety.yield_to_user is False and safety.user_idle_seconds == 9.5
