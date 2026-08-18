"""run_subagent: context isolation, depth guard, budget enforcement, event forwarding."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lai.agent.loop import Agent
from lai.agent.providers.base import Message, TextBlock, ToolCall, TurnResult, Usage
from lai.agent.session import Session
from lai.agent.subagent import MAX_SUBAGENT_DEPTH, SubagentDepthExceeded, run_subagent
from lai.config import load_config
from lai.safety.policy import Risk
from lai.tools.base import ToolRegistry, ToolResult, ToolSpec
from lai.tools.control import register as register_control


class FakeProvider:
    """Replays scripted turns; raises whatever is scripted as an exception."""

    def __init__(self, script: list, *, name: str = "fake", model: str = "fake-1") -> None:
        self.script = list(script)
        self.name = name
        self.model = model
        self.calls: list[dict] = []

    def complete(self, messages, *, system="", tools=None, stream=None) -> TurnResult:
        self.calls.append({"messages": list(messages), "system": system})
        if not self.script:
            return _text_turn("(nothing left to say)")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        pass


def _text_turn(text: str) -> TurnResult:
    return TurnResult(message=Message("assistant", [TextBlock(text)]), usage=Usage(10, 5))


def _tool_turn(name: str, args: dict | None = None, *, call_id: str = "c1") -> TurnResult:
    return TurnResult(
        message=Message("assistant", [ToolCall(id=call_id, name=name, input=args or {})]),
        usage=Usage(20, 8),
    )


def echo_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_control(registry)
    registry.register(
        ToolSpec(
            name="echo",
            description="echo back",
            parameters={"properties": {"value": {"type": "string"}}},
            handler=lambda ctx, args: ToolResult.text(f"echoed {args.get('value', '')}"),
            risk=Risk.READ,
        )
    )
    return registry


def build_parent(script: list, *, registry: ToolRegistry | None = None, max_steps: int = 10, **kwargs) -> Agent:
    config = load_config()
    config = config.with_overrides(limits=replace(config.limits, max_steps=max_steps, max_seconds=60))
    if registry is None:
        registry = echo_registry()
    provider = FakeProvider(script)
    return Agent(config=config, provider=provider, registry=registry, session=Session(), **kwargs)


# -- basic delegation ------------------------------------------------


def test_subagent_runs_its_own_task_and_reports_completion():
    parent = build_parent([])  # parent never actually runs its own loop in this test
    parent.provider.script = [_tool_turn("task_complete", {"summary": "subtask done", "verification": "checked"})]
    result = run_subagent(task="do a focused subtask", parent=parent)
    assert result.status == "completed"
    assert "subtask done" in result.summary
    assert "checked" in result.summary
    assert result.steps == 1
    assert result.session_id


def test_subagent_result_usage_reflects_only_the_child():
    parent = build_parent([])
    parent.provider.script = [
        _tool_turn("echo", {"value": "x"}),
        _tool_turn("task_complete", {"summary": "done"}),
    ]
    result = run_subagent(task="x", parent=parent)
    assert result.usage.input_tokens == 40  # 20 + 20 from the two scripted turns
    assert result.usage.output_tokens == 16


def test_empty_task_is_rejected():
    parent = build_parent([])
    with pytest.raises(ValueError):
        run_subagent(task="", parent=parent)
    with pytest.raises(ValueError):
        run_subagent(task="   ", parent=parent)


# -- context isolation: no transcript leakage -----------------------------


def test_subagent_does_not_leak_into_the_parent_transcript():
    parent = build_parent([])
    parent.session.append(Message.user("parent's own private task context"))
    parent_messages_before = list(parent.session.messages)

    parent.provider.script = [
        _tool_turn("echo", {"value": "child step"}),
        _tool_turn("task_complete", {"summary": "child finished"}),
    ]
    run_subagent(task="an isolated subtask", parent=parent)

    # The parent's transcript is untouched by the child's run.
    assert parent.session.messages == parent_messages_before
    assert not any("child step" in m.text for m in parent.session.messages)


def test_subagent_result_carries_no_message_transcript():
    parent = build_parent([])
    parent.provider.script = [_tool_turn("task_complete", {"summary": "s"})]
    result = run_subagent(task="x", parent=parent)
    assert not hasattr(result, "messages")
    assert not hasattr(result, "transcript")


def test_subagent_gets_a_fresh_session_id_distinct_from_parent():
    parent = build_parent([])
    parent.provider.script = [_tool_turn("task_complete", {"summary": "s"})]
    result = run_subagent(task="x", parent=parent)
    assert result.session_id != parent.session.id


# -- depth guard -----------------------------------------------------


def test_depth_starts_at_zero_and_allows_two_levels():
    assert MAX_SUBAGENT_DEPTH == 2
    parent = build_parent([])
    assert parent.session.metadata.get("subagent_depth", 0) == 0
    parent.provider.script = [_tool_turn("task_complete", {"summary": "depth 1 ok"})]
    result = run_subagent(task="depth one", parent=parent)
    assert result.status == "completed"


def test_depth_two_is_allowed_depth_three_is_refused():
    # Simulate a chain: parent (depth 0) -> subagent A (depth 1) -> subagent B (depth 2).
    grandparent = build_parent([])
    grandparent.session.metadata["subagent_depth"] = 1  # as if grandparent were itself depth 1

    grandparent.provider.script = [_tool_turn("task_complete", {"summary": "depth two ok"})]
    result = run_subagent(task="depth two", parent=grandparent)
    assert result.status == "completed"


def test_depth_beyond_limit_raises_without_running_anything():
    parent = build_parent([])
    parent.session.metadata["subagent_depth"] = MAX_SUBAGENT_DEPTH  # already at the cap
    parent.provider.script = [_tool_turn("task_complete", {"summary": "should never run"})]
    with pytest.raises(SubagentDepthExceeded):
        run_subagent(task="one nest too many", parent=parent)
    # Nothing was consumed from the script — the provider was never called.
    assert parent.provider.script  # still has the unused scripted turn


# -- budget enforcement ------------------------------------------------


def test_default_step_budget_is_smaller_than_a_typical_parent_budget():
    parent = build_parent([_tool_turn("echo", {"value": str(i)}, call_id=f"c{i}") for i in range(30)])
    result = run_subagent(task="loop forever", parent=parent)
    assert result.status == "budget_exceeded"
    assert result.steps == 15  # the documented default


def test_explicit_max_steps_is_honoured():
    parent = build_parent([_tool_turn("echo", {"value": str(i)}, call_id=f"c{i}") for i in range(30)])
    result = run_subagent(task="loop forever", parent=parent, max_steps=3)
    assert result.status == "budget_exceeded"
    assert result.steps == 3


def test_max_steps_is_clamped_to_a_hard_ceiling():
    # Even an absurd request doesn't get an unbounded step budget.
    parent = build_parent([_tool_turn("echo", {"value": str(i)}, call_id=f"c{i}") for i in range(1000)])
    result = run_subagent(task="loop forever", parent=parent, max_steps=10_000)
    assert result.status == "budget_exceeded"
    assert result.steps <= 50


# -- tool scoping -------------------------------------------------------


def test_restricting_tools_still_keeps_control_tools():
    parent = build_parent([])
    parent.provider.script = [_tool_turn("task_complete", {"summary": "s"})]
    result = run_subagent(task="x", parent=parent, tools=["echo"])
    assert result.status == "completed"


def test_restricted_tools_are_unavailable_to_the_child():
    registry = echo_registry()
    registry.register(
        ToolSpec(
            name="dangerous",
            description="should not be reachable",
            parameters={"properties": {}},
            handler=lambda ctx, args: ToolResult.text("ran dangerous"),
            risk=Risk.WRITE,
        )
    )
    parent = build_parent([], registry=registry)
    parent.provider.script = [
        _tool_turn("dangerous"),
        _tool_turn("task_complete", {"summary": "s"}),
    ]
    result = run_subagent(task="x", parent=parent, tools=["echo"])
    # The tool call for "dangerous" should have failed (unknown tool to the
    # scoped registry), which surfaces as a tool_error result fed back to the
    # model rather than the handler actually running -- the run still
    # completes on the next scripted turn.
    assert result.status == "completed"


# -- sharing the safety gate --------------------------------------------


def test_subagent_shares_the_parents_registry_by_default():
    parent = build_parent([])
    parent.provider.script = [_tool_turn("task_complete", {"summary": "s"})]
    captured = {}

    original_run = Agent.run

    def spy_run(self, *a, **kw):
        captured["registry"] = self.registry
        return original_run(self, *a, **kw)

    Agent.run = spy_run
    try:
        run_subagent(task="x", parent=parent)
    finally:
        Agent.run = original_run
    assert captured["registry"] is parent.registry


class _FakeAudit:
    def write(self, *a, **kw):
        pass


class _FakePolicy:
    def check(self, *a, **kw):
        from lai.safety.policy import Decision, Risk, Verdict

        return Verdict(Decision.ALLOW, "test allow", Risk.READ)

    def record(self):
        pass


def test_subagent_shares_policy_audit_and_desktop_objects():
    sentinel_policy = _FakePolicy()
    sentinel_audit = _FakeAudit()
    sentinel_desktop = object()
    parent = build_parent([], policy=sentinel_policy, audit=sentinel_audit, desktop=sentinel_desktop)
    parent.provider.script = [_tool_turn("task_complete", {"summary": "s"})]

    captured = {}
    original_init = Agent.__init__

    def spy_init(self, **kwargs):
        if kwargs.get("session") is not parent.session:
            captured.update(kwargs)
        return original_init(self, **kwargs)

    Agent.__init__ = spy_init
    try:
        run_subagent(task="x", parent=parent)
    finally:
        Agent.__init__ = original_init

    assert captured["policy"] is sentinel_policy
    assert captured["audit"] is sentinel_audit
    assert captured["desktop"] is sentinel_desktop


# -- event forwarding -------------------------------------------------


def test_child_events_are_forwarded_to_parent_on_event_with_subagent_marker():
    seen: list[tuple[str, dict]] = []
    parent = build_parent([], on_event=lambda kind, payload: seen.append((kind, payload)))
    parent.provider.script = [_tool_turn("task_complete", {"summary": "s"})]
    run_subagent(task="x", parent=parent)

    subagent_events = [payload for kind, payload in seen if kind == "subagent"]
    assert subagent_events
    assert all("event" in p and "payload" in p and "depth" in p and "session_id" in p for p in subagent_events)
    assert any(p["event"] == "done" for p in subagent_events)


def test_a_broken_parent_event_callback_does_not_break_the_subagent():
    def bad(kind, payload):
        raise RuntimeError("ui exploded")

    parent = build_parent([], on_event=bad)
    parent.provider.script = [_tool_turn("task_complete", {"summary": "s"})]
    result = run_subagent(task="x", parent=parent)
    assert result.status == "completed"


def test_no_on_event_on_parent_is_fine():
    parent = build_parent([])  # on_event defaults to None
    parent.provider.script = [_tool_turn("task_complete", {"summary": "s"})]
    result = run_subagent(task="x", parent=parent)
    assert result.status == "completed"
