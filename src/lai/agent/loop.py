"""The agentic loop.

Observe → act → verify, repeated under explicit budgets, with every tool call
passing through the safety gate. Termination is explicit: the model calls
``task_complete`` or ``task_blocked``; anything else that stops the loop
(budget, repeated errors, interrupt) is reported as such rather than dressed up
as success.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import Interrupted, LaiError, ProviderError
from ..safety.audit import AuditLog
from ..safety.policy import PolicyEngine
from ..tools.base import ToolContext, ToolRegistry, ToolResult
from ..tools.control import BLOCKED_MARKER, DONE_MARKER
from .prompt import build_system_prompt
from .providers.base import (
    Message,
    Provider,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
)
from .session import Session

EventCallback = Callable[[str, dict], None]

COMPACT_AT_FRACTION = 0.75
COMPACT_MIN_TOKENS = 40_000


@dataclass(slots=True)
class RunResult:
    """Outcome of one autonomous run."""

    status: str
    """completed | blocked | budget_exceeded | error | interrupted | idle"""
    summary: str = ""
    verification: str = ""
    artifacts: list[str] = field(default_factory=list)
    steps: int = 0
    elapsed: float = 0.0
    usage: Usage = field(default_factory=Usage)
    session_id: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "ok": self.ok,
            "summary": self.summary,
            "verification": self.verification,
            "artifacts": self.artifacts,
            "steps": self.steps,
            "elapsed": round(self.elapsed, 1),
            "usage": self.usage.to_dict(),
            "session_id": self.session_id,
            "error": self.error,
        }

    def render(self) -> str:
        icon = {
            "completed": "✓", "blocked": "⊘", "budget_exceeded": "⏱",
            "error": "✗", "interrupted": "■", "idle": "·",
        }.get(self.status, "?")
        lines = [f"{icon} {self.status.replace('_', ' ')} — {self.steps} steps in {self.elapsed:.0f}s"]
        if self.summary:
            lines.append(self.summary)
        if self.verification:
            lines.append(f"Verified: {self.verification}")
        if self.artifacts:
            lines.append("Files: " + ", ".join(self.artifacts))
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


class Agent:
    """Drives a provider against the tool registry until the task is done."""

    def __init__(
        self,
        *,
        config,
        provider: Provider,
        registry: ToolRegistry,
        desktop=None,
        policy: PolicyEngine | None = None,
        audit: AuditLog | None = None,
        skills=None,
        session: Session | None = None,
        approver: Callable | None = None,
        on_event: EventCallback | None = None,
        cwd: Path | None = None,
        system_extra: str = "",
    ) -> None:
        self.config = config
        self.provider = provider
        self.registry = registry
        self.desktop = desktop
        self.policy = policy
        self.audit = audit or AuditLog.disabled()
        self.skills = skills
        self.session = session or Session()
        self.approver = approver
        self.on_event = on_event
        self.cwd = Path(cwd or Path.cwd())
        self.system_extra = system_extra
        self.stop_requested = threading.Event()
        self.tool_extra: dict = {}
        """Long-lived services (memory, scheduler, this agent) handed to tools."""
        self._system_prompt: str = ""

    # -- public ----------------------------------------------------------

    def interrupt(self) -> None:
        """Ask the loop to stop after the current step."""
        self.stop_requested.set()

    def run(self, task: str, *, max_steps: int | None = None) -> RunResult:
        limits = self.config.limits
        budget_steps = max_steps or limits.max_steps
        deadline = time.monotonic() + limits.max_seconds
        started = time.monotonic()

        # Deliberately *not* cleared here: an interrupt raised between
        # submitting a task and the loop starting (the daemon's /stop racing
        # /task) must still be honoured. The flag is cleared when the run ends.
        self.session.task = self.session.task or task
        self._system_prompt = self._build_system_prompt()
        self.session.append(Message.user(task))

        self._emit("start", {"task": task, "provider": self.provider.name, "model": self.provider.model})
        self.audit.write("run_start", task=task, provider=self.provider.name, model=self.provider.model)

        consecutive_errors = 0
        result = RunResult(status="idle", session_id=self.session.id)

        try:
            for step in range(1, budget_steps + 1):
                if self.stop_requested.is_set():
                    raise Interrupted("stopped by user")
                if time.monotonic() > deadline:
                    result = self._budget_result(
                        f"time budget of {limits.max_seconds:.0f}s exhausted", step - 1
                    )
                    break
                if self.session.usage.total > limits.max_tokens:
                    result = self._budget_result(
                        f"token budget of {limits.max_tokens} exhausted", step - 1
                    )
                    break

                self.session.steps = step
                self._emit("step", {"step": step, "of": budget_steps})
                self._maybe_compact()

                try:
                    turn = self._model_turn()
                except ProviderError as exc:
                    consecutive_errors += 1
                    self._emit("error", {"error": str(exc), "recoverable": consecutive_errors < 3})
                    if consecutive_errors >= 3:
                        result = RunResult(
                            status="error",
                            error=f"provider failed {consecutive_errors} times: {exc}",
                            steps=step,
                            session_id=self.session.id,
                        )
                        break
                    time.sleep(2.0 * consecutive_errors)
                    continue

                consecutive_errors = 0
                self.session.add_usage(turn.usage)
                self.session.append(turn.message)

                if turn.text.strip():
                    self._emit("assistant", {"text": turn.text})

                if not turn.tool_calls:
                    # The model stopped without declaring completion. Nudge once;
                    # if it does so again, treat the last message as the answer.
                    if self.session.metadata.get("nudged"):
                        result = RunResult(
                            status="completed",
                            summary=turn.text.strip() or self.session.last_assistant_text(),
                            steps=step,
                            session_id=self.session.id,
                        )
                        break
                    self.session.metadata["nudged"] = True
                    self.session.append(
                        Message.user(
                            "You stopped without finishing. If the task is done, call "
                            "task_complete with your summary and how you verified it. "
                            "If you are stuck, call task_blocked. Otherwise continue working."
                        )
                    )
                    continue

                self.session.metadata.pop("nudged", None)
                outcome = self._run_tools(turn.tool_calls, step)
                if outcome is not None:
                    result = outcome
                    break
            else:
                result = self._budget_result(f"step budget of {budget_steps} exhausted", budget_steps)

        except Interrupted as exc:
            result = RunResult(
                status="interrupted", error=str(exc), steps=self.session.steps, session_id=self.session.id
            )
        except LaiError as exc:
            result = RunResult(
                status="error", error=str(exc), steps=self.session.steps, session_id=self.session.id
            )
        except Exception as exc:  # the loop must always return a result
            result = RunResult(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                steps=self.session.steps,
                session_id=self.session.id,
            )

        self.stop_requested.clear()
        result.elapsed = time.monotonic() - started
        result.usage = self.session.usage
        result.steps = result.steps or self.session.steps
        result.session_id = self.session.id
        self._emit("done", result.to_dict())
        self.audit.write("run_end", **result.to_dict())
        return result

    # -- internals -------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return build_system_prompt(
            desktop=self.desktop,
            safety=getattr(self.config, "safety", None),
            skills=self.skills,
            cwd=self.cwd,
            extra=self.system_extra,
        )

    def _model_turn(self):
        def stream(kind: str, payload: str) -> None:
            if kind in ("text", "thinking"):
                self._emit(kind, {"delta": payload})
            elif kind == "tool":
                self._emit("tool_start", {"name": payload})

        was = self.provider.name
        try:
            return self.provider.complete(
                self.session.messages,
                system=self._system_prompt,
                tools=self.registry.to_anthropic(),
                stream=stream,
            )
        finally:
            # A fallback chain answers as whichever backend actually replied, so
            # the change of identity is the only signal a switch happened — and
            # the user must be told, not quietly served by a different model.
            now = self.provider.name
            if now != was:
                reason = getattr(self.provider, "failures", {}).get(was, "")
                self._emit("provider_switch", {"from": was, "to": now,
                                               "model": self.provider.model, "reason": reason})
                self.audit.write("provider_switch", **{"from": was, "to": now, "reason": reason})

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            desktop=self.desktop,
            config=self.config,
            policy=self.policy,
            audit=self.audit,
            session=self.session,
            skills=self.skills,
            registry=self.registry,
            cwd=self.cwd,
            approver=self.approver,
            extra=self.tool_extra,
        )

    def _run_tools(self, calls: list[ToolCall], step: int) -> RunResult | None:
        """Execute one batch of tool calls. Returns a RunResult if the run ends."""
        context = self._tool_context()
        blocks: list = []
        terminal: RunResult | None = None

        for call in calls:
            if self.stop_requested.is_set():
                raise Interrupted("stopped by user")

            self._emit("tool_call", {"name": call.name, "input": call.input, "id": call.id})
            result: ToolResult = self.registry.call(call.name, call.input, context)
            self._emit(
                "tool_result",
                {
                    "name": call.name,
                    "ok": result.ok,
                    "summary": result.content[:600],
                    "images": len(result.images),
                    "duration": round(result.duration, 3),
                },
            )

            if result.content == DONE_MARKER:
                terminal = RunResult(
                    status="completed",
                    summary=result.data.get("summary", ""),
                    verification=result.data.get("verification", ""),
                    artifacts=list(result.data.get("artifacts", [])),
                    steps=step,
                    session_id=self.session.id,
                )
                blocks.append(ToolResultBlock(call.id, "Task marked complete."))
                continue
            if result.content == BLOCKED_MARKER:
                terminal = RunResult(
                    status="blocked",
                    summary=result.data.get("reason", ""),
                    verification=result.data.get("needs", ""),
                    steps=step,
                    session_id=self.session.id,
                )
                blocks.append(ToolResultBlock(call.id, "Task marked blocked."))
                continue

            blocks.append(
                ToolResultBlock(
                    tool_use_id=call.id,
                    content=result.content or "(no output)",
                    images=tuple(result.images),
                    is_error=not result.ok,
                )
            )

        if blocks:
            self.session.append(Message("user", blocks))
        return terminal

    def _maybe_compact(self) -> None:
        """Summarise old turns when the transcript gets heavy."""
        limit = self.config.limits.max_tokens
        threshold = max(COMPACT_MIN_TOKENS, int(limit * COMPACT_AT_FRACTION))
        estimated = self.session.estimate_tokens()
        if estimated < threshold:
            self.session.prune_images()
            return

        self._emit("compacting", {"estimated_tokens": estimated})
        summary = self._summarise_history()
        dropped = self.session.compact(summary)
        self.audit.write("compacted", dropped=dropped, estimated_tokens=estimated)
        self._emit("compacted", {"dropped": dropped})

    def _summarise_history(self) -> str:
        """Ask the model for a handoff summary; fall back to a mechanical one."""
        transcript = []
        for message in self.session.messages[:-4]:
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    transcript.append(f"{message.role}: {block.text[:400]}")
                elif isinstance(block, ToolCall):
                    transcript.append(f"tool_call: {block.name}({_brief(block.input)})")
                elif isinstance(block, ToolResultBlock):
                    transcript.append(f"tool_result: {block.content[:200]}")
        joined = "\n".join(transcript[-160:])

        try:
            turn = self.provider.complete(
                [
                    Message.user(
                        "Summarise this desktop-agent session so a fresh instance can continue "
                        "seamlessly. Cover: the goal, what has been done, what was learned about "
                        "the apps involved (element names, quirks), what failed and why, and what "
                        "remains. Be specific and factual.\n\n" + joined
                    )
                ],
                system="You write precise handoff notes for an autonomous agent.",
            )
            if turn.text.strip():
                self.session.add_usage(turn.usage)
                return turn.text.strip()
        except Exception:
            pass
        return "Previous steps (mechanical summary):\n" + joined[-4000:]

    def _budget_result(self, reason: str, steps: int) -> RunResult:
        return RunResult(
            status="budget_exceeded",
            summary=self.session.last_assistant_text(),
            error=reason,
            steps=steps,
            session_id=self.session.id,
        )

    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, payload)
        except Exception:
            pass


def _brief(value: dict, limit: int = 120) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in list(value.items())[:4])
    return text[:limit]
