"""Noticing when the agent is repeating itself.

The most expensive failure in an autonomous run is the *same* wrong action
taken twelve times, each costing a full model turn. But false positives here
are worse than the disease: polling a window list, waiting twice, taking two
screenshots are all identical and none of them are stuck. So the rule is
narrow — identical arguments, identical failure, and a success clears it.
"""

from __future__ import annotations

import pytest

from lai.agent.repetition import MAX_TRACKED, REFUSE_AT, WARN_AT, Repetition


@pytest.fixture
def tracker():
    return Repetition()


def fail(tracker, name="ui_click", **arguments):
    return tracker.record(name, arguments, ok=False, content="element_not_found")


# -- warning, then refusing ----------------------------------------------


def test_the_first_failure_passes_without_comment(tracker):
    """One failure is information, not a pattern."""
    assert fail(tracker, name="ui_click") == ""


def test_the_second_identical_failure_is_pointed_out(tracker):
    fail(tracker, name="ui_click", target="Save")
    warning = fail(tracker, name="ui_click", target="Save")
    assert "attempt 2" in warning
    assert "task_blocked" in warning, "and the way out is named"


def test_the_third_is_refused_rather_than_run(tracker):
    """A message the model can ignore does not break the cycle."""
    for _ in range(REFUSE_AT - 1):
        fail(tracker, name="ui_click", target="Save")
    refusal = tracker.should_refuse("ui_click", {"target": "Save"})
    assert refusal
    assert "will not produce a different result" in refusal
    assert "ui_snapshot" in refusal, "the alternatives are spelled out"


def test_nothing_is_refused_before_the_evidence_is_in(tracker):
    fail(tracker, name="ui_click", target="Save")
    assert tracker.should_refuse("ui_click", {"target": "Save"}) == ""


# -- what must NOT be treated as repetition ------------------------------


def test_success_is_never_repetition(tracker):
    """Polling, waiting and screenshotting are identical and legitimate."""
    for _ in range(10):
        assert tracker.record("window_list", {}, ok=True) == ""
    assert tracker.should_refuse("window_list", {}) == ""


def test_a_success_clears_the_history(tracker):
    """The world moved; what failed before is no longer evidence about now."""
    fail(tracker, name="ui_click", target="Save")
    fail(tracker, name="ui_click", target="Save")
    tracker.record("ui_click", {"target": "Save"}, ok=True)
    assert fail(tracker, name="ui_click", target="Save") == "", "counting starts again"


def test_different_arguments_are_exploration_not_repetition(tracker):
    for target in ("Save", "Save As", "OK", "Apply"):
        assert fail(tracker, name="ui_click", target=target) == ""


def test_different_tools_are_counted_separately(tracker):
    fail(tracker, name="ui_click", target="Save")
    assert fail(tracker, name="ui_find", target="Save") == ""


def test_argument_order_does_not_hide_a_repeat(tracker):
    tracker.record("ui_click", {"a": 1, "b": 2}, ok=False)
    warning = tracker.record("ui_click", {"b": 2, "a": 1}, ok=False)
    assert "attempt 2" in warning


# -- staying bounded -----------------------------------------------------


def test_a_long_run_does_not_grow_without_bound(tracker):
    for index in range(MAX_TRACKED + 50):
        tracker.record("t", {"i": index}, ok=False)
    assert len(tracker.failures) <= MAX_TRACKED + 1


def test_unserialisable_arguments_do_not_crash_it(tracker):
    assert tracker.record("t", {"weird": object()}, ok=False) == ""


def test_stuck_calls_are_reportable(tracker):
    for _ in range(WARN_AT):
        fail(tracker, name="ui_click", target="Save")
    fail(tracker, name="ui_find", target="Other")
    stuck = tracker.stuck_on()
    assert stuck and "ui_click" in stuck[0][0]
    assert all(count >= WARN_AT for _key, count in stuck)


# -- the loop acts on it -------------------------------------------------


def test_the_loop_refuses_the_third_attempt_without_running_it(tmp_path):
    """Refusing costs nothing; the turn it replaces costs a model call."""
    from lai.agent.loop import Agent
    from lai.agent.providers.base import ToolCall
    from lai.agent.session import Session
    from lai.config import load_config
    from lai.tools.base import ToolResult

    calls: list = []

    class Registry:
        def call(self, name, arguments, context):
            calls.append(name)
            return ToolResult.failure("element_not_found")

        def get(self, name):
            return type("S", (), {"name": name, "risk": None})()

    agent = Agent.__new__(Agent)
    agent.config = load_config().with_overrides(home=tmp_path)
    agent.registry = Registry()
    agent.repetition = Repetition()
    agent.audit = type("A", (), {"write": lambda self, *a, **k: None})()
    agent.session = Session()
    agent.stop_requested = type("E", (), {"is_set": lambda self: False})()
    agent.desktop = None
    agent.policy = None
    agent._emit = lambda kind, payload: None
    agent._yield_to_human = lambda name: None
    agent._tool_context = lambda: None

    call = ToolCall("1", "ui_click", {"target": "Save"})
    for _ in range(REFUSE_AT + 2):
        agent._run_tools([call], 1)

    assert len(calls) == REFUSE_AT - 1, "it stops actually running the doomed call"
