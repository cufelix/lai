"""Tools that cannot possibly work right now.

Eleven per cent of every tool call this machine has logged was refused before
it ran — 90 clicks in `ask` mode with nobody to ask, 73 in `readonly`, 19 shell
commands in `auto`. Each one cost a full model turn to discover something that
was knowable before the turn started.

The model has no way to see a permission mode. So it is not told about tools it
cannot use: they are simply not offered, and it is told plainly why, once.
"""

from __future__ import annotations

import pytest

from lai.agent.toolgate import ToolGate
from lai.config import SafetyConfig
from lai.safety.policy import PolicyEngine, Risk
from lai.tools.base import ToolRegistry, ToolResult, ToolSpec


def spec(name: str, risk: Risk) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} does something",
        parameters={"properties": {}},
        handler=lambda ctx, args: ToolResult.text("ok"),
        risk=risk,
    )


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(spec("ui_snapshot", Risk.READ))
    reg.register(spec("window_list", Risk.READ))
    reg.register(spec("ui_click", Risk.INPUT))
    reg.register(spec("computer_type", Risk.INPUT))
    reg.register(spec("file_write", Risk.WRITE))
    reg.register(spec("shell_exec", Risk.DESTRUCTIVE))
    return reg


def policy(mode: str, **kwargs) -> PolicyEngine:
    return PolicyEngine(SafetyConfig(mode=mode, **kwargs))


def names(gate: ToolGate, task: str = "click the save button") -> set[str]:
    return {spec.name for spec in gate.choose(task)[0]}


# -- readonly: acting can never work -------------------------------------


def test_readonly_offers_nothing_that_acts(registry):
    gate = ToolGate(registry, policy=policy("readonly"))
    offered = names(gate)
    assert "ui_snapshot" in offered and "window_list" in offered
    assert "ui_click" not in offered
    assert "file_write" not in offered
    assert "shell_exec" not in offered


def test_readonly_says_why_rather_than_leaving_a_hole(registry):
    gate = ToolGate(registry, policy=policy("readonly"))
    said = gate.describe_forbidden()
    assert "readonly" in said.lower()
    assert "cannot" in said.lower()
    assert "ui_click" in said
    # Silence would be worse: the model would invent a way around it.
    assert "another way" in said.lower()


# -- ask, with nobody to ask ---------------------------------------------


def test_ask_mode_without_an_approver_offers_nothing_that_needs_asking(registry):
    """`lai do` has no terminal to prompt at, so every click is refused."""
    gate = ToolGate(registry, policy=policy("ask"), can_ask=False)
    offered = names(gate)
    assert "ui_snapshot" in offered
    assert "ui_click" not in offered


def test_ask_mode_with_an_approver_offers_everything(registry):
    """In chat there is somebody to ask, so asking is the whole point."""
    gate = ToolGate(registry, policy=policy("ask"), can_ask=True)
    assert "ui_click" in names(gate)


# -- auto: acting is fine, destroying still needs a human ----------------


def test_auto_without_an_approver_keeps_the_hands_but_drops_the_shell(registry):
    gate = ToolGate(registry, policy=policy("auto"), can_ask=False)
    offered = names(gate)
    assert "ui_click" in offered
    assert "file_write" in offered
    assert "shell_exec" not in offered


def test_auto_with_an_approver_keeps_the_shell(registry):
    assert "shell_exec" in names(ToolGate(registry, policy=policy("auto"), can_ask=True))


# -- yolo, dry-run and the default ---------------------------------------


def test_yolo_offers_everything(registry):
    assert "shell_exec" in names(ToolGate(registry, policy=policy("yolo"), can_ask=False))


def test_dry_run_offers_nothing_that_has_an_effect(registry):
    gate = ToolGate(registry, policy=policy("auto", dry_run=True), can_ask=False)
    offered = names(gate)
    assert "ui_snapshot" in offered
    assert "ui_click" not in offered


def test_no_policy_at_all_changes_nothing(registry):
    """The gate is used outside a run too, where there is no policy to consult."""
    assert "ui_click" in names(ToolGate(registry))


def test_a_denied_tool_is_never_offered(registry):
    config = SafetyConfig(mode="yolo", deny_tools=("shell_exec",))
    gate = ToolGate(registry, policy=PolicyEngine(config), can_ask=True)
    assert "shell_exec" not in names(gate)


def test_an_explicitly_allowed_tool_survives_readonly(registry):
    config = SafetyConfig(mode="readonly", allow_tools=("ui_click",))
    gate = ToolGate(registry, policy=PolicyEngine(config), can_ask=False)
    assert "ui_click" in names(gate)


# -- an approver that cannot reach anybody -------------------------------


def test_an_approver_with_no_terminal_does_not_count_as_one(registry, monkeypatch):
    """`lai do` in a pipeline has an approver. It refuses everything, and the
    agent spent a turn discovering that for each of ninety clicks."""
    import sys

    from lai.cli import Out, _interactive_approver

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    approver = _interactive_approver(Out(color=False))
    assert approver.can_ask is False

    gate = ToolGate(registry, policy=policy("ask"),
                    can_ask=bool(getattr(approver, "can_ask", True)))
    assert "ui_click" not in names(gate)


def test_an_approver_at_a_terminal_does(registry, monkeypatch):
    import sys

    from lai.cli import Out, _interactive_approver

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    approver = _interactive_approver(Out(color=False))
    assert approver.can_ask is True
    assert "ui_click" in names(ToolGate(registry, policy=policy("ask"), can_ask=True))
