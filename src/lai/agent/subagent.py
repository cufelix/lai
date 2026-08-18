"""Delegate an isolated subtask to a fresh, budget-limited agent.

The point of a subagent is context isolation: a long, exploratory subtask
(read ten log files and find the root cause; drive a multi-step dialog to
completion) can burn thousands of tokens of screenshots and back-and-forth
that the parent agent never needs to see. :func:`run_subagent` runs that
subtask in a brand-new :class:`~lai.agent.session.Session` — a clean
transcript — and hands the parent back a compact conclusion instead of the
whole exchange.

Isolation is about *context*, not about *safety*: the child shares the
parent's tool registry (or a filtered view of it), desktop, policy engine and
audit log. A subagent that could bypass the safety gate would defeat the
entire point of having one, so nothing here constructs a new
:class:`~lai.safety.policy.PolicyEngine` or a new audit log — it always reuses
the parent's.

Non-obvious design decision: recursion depth is tracked on the *session*
metadata (``subagent_depth``), not on the :class:`Agent` object itself. A
subagent's own ``Agent`` is otherwise indistinguishable from a top-level one —
same provider, same registry shape — so if a delegated task itself calls
``delegate`` again, the only way to know "how deep are we" is to read it back
off the session that was stamped when *this* subagent was created.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..errors import LaiError
from ..tools.base import ToolRegistry
from .loop import Agent
from .providers.base import Usage
from .session import Session

MAX_SUBAGENT_DEPTH = 2
"""A subagent may itself delegate once more (depth 2); a third level is refused."""

# Control tools a delegated agent must always retain, regardless of any
# ``tools`` filter — without these it can never signal completion and will
# just run out its step budget every time.
_ALWAYS_KEPT_TOOLS = frozenset({"task_complete", "task_blocked"})


class SubagentDepthExceeded(LaiError):
    """Refused: delegating would create a chain deeper than :data:`MAX_SUBAGENT_DEPTH`."""

    code = "subagent_depth_exceeded"


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """The compact answer a subagent hands back — never the raw transcript."""

    status: str
    summary: str
    steps: int
    usage: Usage
    session_id: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "summary": self.summary,
            "steps": self.steps,
            "usage": self.usage.to_dict(),
            "session_id": self.session_id,
        }


def run_subagent(
    *,
    task: str,
    parent: Agent,
    max_steps: int = 15,
    tools: Iterable[str] | None = None,
    system_extra: str = "",
) -> SubagentResult:
    """Run ``task`` to completion in an isolated session and return a summary.

    Raises :class:`SubagentDepthExceeded` rather than running anything if
    ``parent`` is already itself a subagent nested at the recursion limit.
    """
    if not task or not task.strip():
        raise ValueError("delegate: task must not be empty")

    depth = int((parent.session.metadata or {}).get("subagent_depth", 0)) if parent.session else 0
    new_depth = depth + 1
    if new_depth > MAX_SUBAGENT_DEPTH:
        raise SubagentDepthExceeded(
            f"refusing to delegate: this would nest subagents {new_depth} deep, "
            f"beyond the limit of {MAX_SUBAGENT_DEPTH}",
            detail="flatten the work into the current task instead of delegating again",
        )

    bounded_steps = max(1, min(int(max_steps), 50))

    child_session = Session()
    child_session.metadata["subagent_depth"] = new_depth

    wanted = (set(tools) | _ALWAYS_KEPT_TOOLS) if tools is not None else None
    child_registry = _scoped_registry(parent.registry, wanted)

    def _forward(kind: str, payload: dict) -> None:
        """Relay child events to the parent's callback, tagged so a UI can nest them."""
        if parent.on_event is None:
            return
        try:
            parent.on_event(
                "subagent",
                {"depth": new_depth, "session_id": child_session.id, "event": kind, "payload": payload},
            )
        except Exception:
            pass  # a broken UI callback must not abort the subagent's work

    child = Agent(
        config=parent.config,
        provider=parent.provider,
        registry=child_registry,
        desktop=parent.desktop,
        policy=parent.policy,
        audit=parent.audit,
        skills=parent.skills,
        session=child_session,
        approver=parent.approver,
        on_event=_forward,
        cwd=parent.cwd,
        system_extra=system_extra,
    )

    result = child.run(task, max_steps=bounded_steps)

    summary = result.summary.strip() or result.error.strip() or "(subagent produced no summary)"
    if result.verification:
        summary = f"{summary}\nVerified: {result.verification}"
    if result.artifacts:
        summary = f"{summary}\nArtifacts: {', '.join(result.artifacts)}"

    return SubagentResult(
        status=result.status,
        summary=summary,
        steps=result.steps,
        usage=result.usage,
        session_id=result.session_id,
    )


def _scoped_registry(parent_registry: ToolRegistry, wanted: set[str] | None) -> ToolRegistry:
    """A view of ``parent_registry`` limited to ``wanted`` tool names.

    Reuses the parent's :class:`~lai.tools.base.ToolSpec` objects as-is — the
    handlers still close over whatever they always did, and every call still
    goes through the shared :class:`~lai.tools.base.ToolContext` (same policy,
    same audit log) built by the child :class:`Agent`. Only the *set of names
    the model can see* changes.
    """
    if wanted is None:
        return parent_registry
    scoped = ToolRegistry(policy=parent_registry.policy)
    for spec in parent_registry.specs():
        if spec.name in wanted:
            scoped.register(spec)
    return scoped


__all__ = ["MAX_SUBAGENT_DEPTH", "SubagentDepthExceeded", "SubagentResult", "run_subagent"]
