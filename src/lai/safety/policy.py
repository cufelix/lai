"""Permission policy — the gate every action passes through.

Four modes:

``readonly``  perception only; nothing that changes the world runs.
``ask``       safe actions run; anything that writes asks the human.
``auto``      known-safe *and* ordinary write actions run; destructive ones ask.
``yolo``      everything except hard-denied patterns runs unattended.

Hard denials (destructive shell patterns, password managers) apply in **every**
mode, including yolo. That is deliberate: an autonomous agent that can wipe a
disk or type into a password prompt is not a feature.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from ..config import SafetyConfig


class Risk(str, Enum):
    """How much damage a tool can do."""

    READ = "read"
    """No side effects: screenshots, listing windows, reading files."""
    INPUT = "input"
    """Drives the UI: clicks, typing, scrolling. Reversible but visible."""
    WRITE = "write"
    """Changes durable state: writing files, launching or closing apps."""
    DESTRUCTIVE = "destructive"
    """Hard to undo: shell commands, killing processes, deleting."""


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Verdict:
    decision: Decision
    reason: str
    risk: Risk = Risk.READ
    matched: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    def to_dict(self) -> dict:
        out = {"decision": self.decision.value, "reason": self.reason, "risk": self.risk.value}
        if self.matched:
            out["matched"] = self.matched
        return out


# What each mode permits without asking.
_MODE_ALLOWANCE: dict[str, frozenset[Risk]] = {
    "readonly": frozenset({Risk.READ}),
    "ask": frozenset({Risk.READ}),
    "auto": frozenset({Risk.READ, Risk.INPUT, Risk.WRITE}),
    "yolo": frozenset({Risk.READ, Risk.INPUT, Risk.WRITE, Risk.DESTRUCTIVE}),
}


class PolicyEngine:
    """Decides allow / ask / deny for one tool invocation."""

    def __init__(self, config: SafetyConfig, *, focus_provider=None) -> None:
        self.config = config
        self._focus_provider = focus_provider
        self._deny_res = _compile(config.deny_shell_patterns)
        self._confirm_res = _compile(config.confirm_shell_patterns)
        self._actions: deque[float] = deque(maxlen=2000)

    # -- main entry point -------------------------------------------------

    def check(self, tool_name: str, tool_input: dict, *, risk: Risk = Risk.READ) -> Verdict:
        if tool_name in self.config.deny_tools:
            return Verdict(Decision.DENY, f"tool {tool_name!r} is denied by configuration", risk)

        # Rate limit before anything else — a runaway loop is itself a hazard.
        if risk is not Risk.READ:
            over, rate = self._rate_limited()
            if over:
                return Verdict(
                    Decision.DENY,
                    f"action rate limit exceeded ({rate}/min > {self.config.max_actions_per_minute}/min)",
                    risk,
                )

        # Hard denials apply in every mode, yolo included.
        blocked = self._hard_deny(tool_name, tool_input)
        if blocked is not None:
            return blocked

        if tool_name in self.config.allow_tools:
            return Verdict(Decision.ALLOW, "tool is explicitly allowed", risk)

        mode = self.config.mode
        if self.config.dry_run and risk is not Risk.READ:
            return Verdict(Decision.DENY, "dry-run mode: no side effects permitted", risk)

        if mode == "readonly" and risk is not Risk.READ:
            return Verdict(
                Decision.DENY, f"readonly mode blocks {risk.value} actions", risk
            )

        allowed = _MODE_ALLOWANCE.get(mode, _MODE_ALLOWANCE["ask"])
        if risk in allowed:
            return Verdict(Decision.ALLOW, f"{risk.value} action permitted in {mode} mode", risk)

        # Shell commands that merely warrant a look get ASK rather than DENY.
        matched = self._matches(self._confirm_res, _command_text(tool_input))
        if matched:
            return Verdict(
                Decision.ASK, f"command needs confirmation (matched {matched!r})", risk, matched
            )
        return Verdict(Decision.ASK, f"{risk.value} action needs confirmation in {mode} mode", risk)

    def record(self) -> None:
        """Note that an action actually ran (feeds the rate limiter)."""
        self._actions.append(time.monotonic())

    # -- hard rules -------------------------------------------------------

    def _hard_deny(self, tool_name: str, tool_input: dict) -> Verdict | None:
        command = _command_text(tool_input)
        if command:
            matched = self._matches(self._deny_res, command)
            if matched:
                return Verdict(
                    Decision.DENY,
                    f"command matches a destructive pattern ({matched!r})",
                    Risk.DESTRUCTIVE,
                    matched,
                )

        if _is_input_tool(tool_name):
            protected = self.protected_focus()
            if protected:
                return Verdict(
                    Decision.DENY,
                    f"refusing to send input to a protected window: {protected}",
                    Risk.INPUT,
                    protected,
                )
        return None

    def protected_focus(self) -> str | None:
        """Name of the focused window if it is on the protected list."""
        if self._focus_provider is None:
            return None
        try:
            window = self._focus_provider()
        except Exception:
            return None
        if window is None:
            return None
        app = f"{getattr(window, 'wm_class', '')} {getattr(window, 'instance', '')}".lower()
        title = str(getattr(window, "title", "")).lower()
        for needle in self.config.protected_apps:
            if needle and needle.lower() in app:
                return f"{needle} (app)"
        for needle in self.config.protected_titles:
            if needle and needle.lower() in title:
                return f"{needle} (title)"
        return None

    def _rate_limited(self) -> tuple[bool, int]:
        cutoff = time.monotonic() - 60.0
        while self._actions and self._actions[0] < cutoff:
            self._actions.popleft()
        count = len(self._actions)
        return count >= self.config.max_actions_per_minute, count

    @staticmethod
    def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> str:
        if not text:
            return ""
        for pattern in patterns:
            found = pattern.search(text)
            if found:
                return found.group(0)[:80]
        return ""


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        try:
            compiled.append(re.compile(raw))
        except re.error:
            continue  # a bad user pattern must not break the whole gate
    return tuple(compiled)


def _command_text(tool_input: dict) -> str:
    if not isinstance(tool_input, dict):
        return ""
    parts = [
        str(tool_input.get(key, ""))
        for key in ("command", "cmd", "script", "shell", "args")
        if tool_input.get(key)
    ]
    return " ".join(parts)


def _is_input_tool(tool_name: str) -> bool:
    return tool_name.startswith(("computer_", "ui_")) and tool_name not in (
        "computer_screenshot",
        "ui_snapshot",
        "ui_find",
    )
