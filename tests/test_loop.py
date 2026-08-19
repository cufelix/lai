"""The agent loop, driven by a scripted fake provider (no network, no desktop)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lai.agent.loop import Agent, RunResult
from lai.agent.providers.base import Message, TextBlock, ToolCall, ToolResultBlock, TurnResult, Usage
from lai.agent.session import Session
from lai.config import load_config
from lai.errors import ProviderError
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
        self.closed = False

    def complete(self, messages, *, system="", tools=None, stream=None) -> TurnResult:
        self.calls.append({"messages": list(messages), "system": system, "tools": tools})
        if not self.script:
            return _text_turn("(nothing left to say)")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if stream is not None and item.message.text:
            stream("text", item.message.text)
        return item

    def close(self) -> None:
        self.closed = True


def _text_turn(text: str) -> TurnResult:
    return TurnResult(
        message=Message("assistant", [TextBlock(text)]),
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _tool_turn(name: str, args: dict | None = None, *, call_id: str = "c1", text: str = "") -> TurnResult:
    blocks: list = [TextBlock(text)] if text else []
    blocks.append(ToolCall(id=call_id, name=name, input=args or {}))
    return TurnResult(
        message=Message("assistant", blocks),
        stop_reason="tool_use",
        usage=Usage(input_tokens=20, output_tokens=8),
    )


def build_agent(script: list, *, registry: ToolRegistry | None = None, max_steps: int = 10, **kwargs):
    config = load_config()
    config = config.with_overrides(limits=replace(config.limits, max_steps=max_steps, max_seconds=60))
    if registry is None:
        registry = ToolRegistry()
        register_control(registry)
    provider = FakeProvider(script)
    agent = Agent(
        config=config,
        provider=provider,
        registry=registry,
        desktop=None,
        session=Session(),
        **kwargs,
    )
    return agent, provider


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


# -- termination ---------------------------------------------------------


def test_task_complete_finishes_the_run():
    agent, _ = build_agent([
        _tool_turn("task_complete", {"summary": "all done", "verification": "checked it"})
    ])
    result = agent.run("do the thing")
    assert result.status == "completed" and result.ok
    assert result.summary == "all done"
    assert result.verification == "checked it"
    assert result.steps == 1


def test_task_complete_carries_artifacts():
    agent, _ = build_agent([
        _tool_turn("task_complete", {"summary": "s", "artifacts": ["/tmp/a.txt", "/tmp/b.txt"]})
    ])
    assert agent.run("x").artifacts == ["/tmp/a.txt", "/tmp/b.txt"]


def test_task_blocked_reports_blocked():
    agent, _ = build_agent([
        _tool_turn("task_blocked", {"reason": "app missing", "needs": "install gimp"})
    ])
    result = agent.run("x")
    assert result.status == "blocked" and not result.ok
    assert result.summary == "app missing"
    assert result.verification == "install gimp"


def test_bare_text_is_nudged_once_then_accepted():
    agent, provider = build_agent([_text_turn("here is my answer"), _text_turn("still my answer")])
    result = agent.run("what is up")
    assert result.status == "completed"
    assert result.summary == "still my answer"
    # Two model turns, and the nudge was inserted between them.
    assert len(provider.calls) == 2
    nudges = [
        m for m in agent.session.messages
        if m.role == "user" and "stopped without finishing" in m.text
    ]
    assert len(nudges) == 1


def test_a_tool_call_clears_the_nudge_flag():
    registry = echo_registry()
    agent, _ = build_agent(
        [
            _text_turn("thinking out loud"),
            _tool_turn("echo", {"value": "hi"}),
            _text_turn("done thinking"),
            _text_turn("really done"),
        ],
        registry=registry,
    )
    result = agent.run("x")
    assert result.status == "completed"
    assert result.summary == "really done"


# -- budgets and failures ------------------------------------------------


def test_step_budget_exhaustion():
    script = [_tool_turn("echo", {"value": str(i)}, call_id=f"c{i}") for i in range(10)]
    agent, _ = build_agent(script, registry=echo_registry(), max_steps=3)
    result = agent.run("loop forever")
    assert result.status == "budget_exceeded"
    assert "step budget of 3" in result.error
    assert result.steps == 3


def test_time_budget_exhaustion():
    agent, _ = build_agent([_tool_turn("echo", {"value": "x"})] * 5, registry=echo_registry())
    agent.config = agent.config.with_overrides(
        limits=replace(agent.config.limits, max_seconds=-1)
    )
    result = agent.run("x")
    assert result.status == "budget_exceeded" and "time budget" in result.error


def test_three_consecutive_provider_errors_end_the_run():
    agent, _ = build_agent([ProviderError("boom")] * 3)
    result = agent.run("x")
    assert result.status == "error"
    assert "provider failed 3 times" in result.error


def test_a_transient_provider_error_is_recovered():
    agent, _ = build_agent([
        ProviderError("temporary"),
        _tool_turn("task_complete", {"summary": "recovered"}),
    ])
    result = agent.run("x")
    assert result.status == "completed" and result.summary == "recovered"


def test_interrupt_before_the_first_step():
    agent, _ = build_agent([_tool_turn("task_complete", {"summary": "never reached"})])
    agent.interrupt()
    result = agent.run("x")
    assert result.status == "interrupted"


def test_unexpected_exception_is_contained():
    class Exploding:
        name, model = "boom", "boom-1"

        def complete(self, *a, **kw):
            raise RuntimeError("unexpected")

        def close(self):
            pass

    config = load_config()
    agent = Agent(config=config, provider=Exploding(), registry=ToolRegistry(), session=Session())
    result = agent.run("x")
    assert result.status == "error" and "RuntimeError" in result.error


# -- transcript mechanics ------------------------------------------------


def test_tool_results_are_appended_as_a_user_message():
    agent, _ = build_agent(
        [_tool_turn("echo", {"value": "hello"}), _tool_turn("task_complete", {"summary": "s"})],
        registry=echo_registry(),
    )
    agent.run("x")
    results = [
        block
        for message in agent.session.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert any("echoed hello" in block.content for block in results)
    assert all(message.role == "user" for message in agent.session.messages if any(
        isinstance(b, ToolResultBlock) for b in message.content
    ))


def test_tool_images_are_carried_into_the_transcript():
    registry = echo_registry()
    registry.register(
        ToolSpec(
            name="shoot",
            description="returns an image",
            parameters={"properties": {}},
            handler=lambda ctx, args: ToolResult(ok=True, content="shot", images=[b"\x89PNGfake"]),
            risk=Risk.READ,
        )
    )
    agent, _ = build_agent(
        [_tool_turn("shoot"), _tool_turn("task_complete", {"summary": "s"})], registry=registry
    )
    agent.run("x")
    blocks = [
        block
        for message in agent.session.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert any(block.images for block in blocks)


def test_failed_tool_result_is_flagged_as_an_error():
    registry = echo_registry()
    registry.register(
        ToolSpec(
            name="fails",
            description="always fails",
            parameters={"properties": {}},
            handler=lambda ctx, args: ToolResult.failure("nope"),
            risk=Risk.READ,
        )
    )
    agent, _ = build_agent(
        [_tool_turn("fails"), _tool_turn("task_complete", {"summary": "s"})], registry=registry
    )
    agent.run("x")
    errors = [
        block
        for message in agent.session.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.is_error
    ]
    assert errors


def test_usage_accumulates_across_turns():
    agent, _ = build_agent(
        [_tool_turn("echo", {"value": "a"}), _tool_turn("task_complete", {"summary": "s"})],
        registry=echo_registry(),
    )
    result = agent.run("x")
    assert result.usage.input_tokens == 40 and result.usage.output_tokens == 16


def test_system_prompt_and_tools_are_sent_to_the_provider():
    agent, provider = build_agent([_tool_turn("task_complete", {"summary": "s"})])
    agent.run("open something")
    first = provider.calls[0]
    assert "LAI" in first["system"]
    assert any(tool["name"] == "task_complete" for tool in first["tools"])


def test_events_are_emitted():
    seen: list[str] = []
    agent, _ = build_agent(
        [_tool_turn("echo", {"value": "a"}), _tool_turn("task_complete", {"summary": "s"})],
        registry=echo_registry(),
        on_event=lambda kind, payload: seen.append(kind),
    )
    agent.run("x")
    for expected in ("start", "step", "tool_call", "tool_result", "done"):
        assert expected in seen


def test_a_broken_event_callback_does_not_break_the_run():
    def bad(kind, payload):
        raise RuntimeError("ui exploded")

    agent, _ = build_agent([_tool_turn("task_complete", {"summary": "s"})], on_event=bad)
    assert agent.run("x").status == "completed"


def test_session_is_persisted_when_bound(tmp_path):
    agent, _ = build_agent([_tool_turn("task_complete", {"summary": "s"})])
    agent.session.bind(tmp_path)
    agent.run("persist me")
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert "persist me" in files[0].read_text()


# -- RunResult -----------------------------------------------------------


def test_run_result_dict_and_render():
    result = RunResult(
        status="completed", summary="did it", verification="checked",
        artifacts=["/tmp/x"], steps=4, elapsed=12.3, usage=Usage(1, 2),
    )
    data = result.to_dict()
    assert data["ok"] is True and data["steps"] == 4
    text = result.render()
    assert "completed" in text and "did it" in text and "/tmp/x" in text


@pytest.mark.parametrize(
    "status", ["completed", "blocked", "budget_exceeded", "error", "interrupted", "idle"]
)
def test_render_handles_every_status(status):
    assert RunResult(status=status).render()


def test_only_completed_is_ok():
    assert RunResult(status="completed").ok
    for status in ("blocked", "budget_exceeded", "error", "interrupted", "idle"):
        assert not RunResult(status=status).ok


def test_interrupt_flag_is_cleared_between_runs():
    # A pre-run interrupt is honoured exactly once; the next run starts clean.
    agent, _ = build_agent([
        _tool_turn("task_complete", {"summary": "never reached"}),
        _tool_turn("task_complete", {"summary": "second run"}),
    ])
    agent.interrupt()
    assert agent.run("first").status == "interrupted"
    assert agent.run("second").status == "completed"


# -- compaction sizes itself to the backend ------------------------------


def _agent_with(provider, max_tokens=600_000):
    """A bare Agent with just enough wiring to ask about its context budget."""
    from dataclasses import replace

    from lai.agent.loop import Agent
    from lai.config import load_config

    agent = Agent.__new__(Agent)
    config = load_config()
    agent.config = config.with_overrides(limits=replace(config.limits, max_tokens=max_tokens))
    agent.provider = provider
    return agent


def test_a_hosted_backend_keeps_the_configured_budget():
    provider = type("P", (), {"name": "zai", "model": "glm"})()
    assert _agent_with(provider)._context_budget() == 600_000


def test_a_cli_backend_shrinks_the_budget_to_what_it_can_swallow():
    """Without this the loop never compacts, so the provider cuts the middle
    out of the conversation every turn instead — and the agent forgets what it
    already tried."""
    provider = type("P", (), {"name": "cli:claude", "model": "claude", "context_chars": 96_000})()
    agent = _agent_with(provider)
    budget = agent._context_budget()
    assert 10_000 < budget < 20_000, "roughly what 96k characters of prompt holds"
    assert agent._compact_threshold() < budget, (
        "the threshold must sit below what the backend can take, or it never fires"
    )


def test_the_budget_never_collapses_to_nothing():
    from lai.agent.loop import MIN_CONTEXT_TOKENS

    provider = type("P", (), {"name": "cli:tiny", "model": "t", "context_chars": 100})()
    assert _agent_with(provider)._context_budget() >= MIN_CONTEXT_TOKENS


def test_a_hosted_backend_keeps_its_do_not_bother_floor():
    """Summarising a two-message conversation is pointless work."""
    from lai.agent.loop import COMPACT_MIN_TOKENS

    provider = type("P", (), {"name": "zai", "model": "glm"})()
    assert _agent_with(provider, max_tokens=1_000)._compact_threshold() == COMPACT_MIN_TOKENS
