"""Tool contract: specs, registry, dispatch.

Tools are declared once and rendered into whatever shape a provider wants
(Anthropic ``input_schema``, OpenAI ``function``, MCP ``inputSchema``). Every
call goes through :meth:`ToolRegistry.call`, which is the single choke point for
permission checks, validation, auditing and error containment — so no tool can
skip the safety gate by accident.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import InvalidToolInput, LaiError, PermissionDenied, ToolNotFound
from ..safety.policy import Decision, PolicyEngine, Risk, Verdict


@dataclass(slots=True)
class ToolResult:
    """What a tool hands back to the model."""

    ok: bool = True
    content: str = ""
    data: dict = field(default_factory=dict)
    images: list[bytes] = field(default_factory=list)
    """PNG bytes to attach as vision blocks."""
    duration: float = 0.0

    @classmethod
    def text(cls, content: str, **data: Any) -> ToolResult:
        return cls(ok=True, content=content, data=data)

    @classmethod
    def json(cls, payload: dict, *, content: str | None = None) -> ToolResult:
        return cls(
            ok=True,
            content=content if content is not None else json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            data=payload,
        )

    @classmethod
    def failure(cls, content: str, **data: Any) -> ToolResult:
        # The parameter is named ``content`` (not ``message``) on purpose:
        # callers splat ``LaiError.to_dict()`` in here, and that carries a
        # "message" key which would otherwise collide with the parameter name.
        return cls(ok=False, content=content, data={"error": True, **data})

    def truncated(self, limit: int) -> ToolResult:
        if limit <= 0 or len(self.content) <= limit:
            return self
        head = self.content[: limit - 200]
        return ToolResult(
            ok=self.ok,
            content=f"{head}\n… [truncated {len(self.content) - limit + 200} chars]",
            data=self.data,
            images=self.images,
            duration=self.duration,
        )


@dataclass(slots=True)
class ToolContext:
    """Everything a tool handler is allowed to touch."""

    desktop: Any = None
    config: Any = None
    policy: PolicyEngine | None = None
    audit: Any = None
    session: Any = None
    skills: Any = None
    registry: ToolRegistry | None = None
    cwd: Path = field(default_factory=Path.cwd)
    approver: Callable[[str, dict, Verdict], bool] | None = None
    extra: dict = field(default_factory=dict)


ToolHandler = Callable[[ToolContext, dict], ToolResult]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One callable capability exposed to the model."""

    name: str
    description: str
    parameters: dict
    """JSON Schema object describing the input."""
    handler: ToolHandler
    risk: Risk = Risk.READ
    group: str = "core"
    returns_image: bool = False

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _ensure_object_schema(self.parameters),
        }

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _ensure_object_schema(self.parameters),
            },
        }

    def to_mcp(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": _ensure_object_schema(self.parameters),
        }


class ToolRegistry:
    """Holds tool specs and dispatches calls through the safety gate."""

    def __init__(self, *, policy: PolicyEngine | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.policy = policy

    # -- registration ----------------------------------------------------

    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        if spec.name in self._tools and not replace:
            raise LaiError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        *,
        risk: Risk = Risk.READ,
        group: str = "core",
        returns_image: bool = False,
    ):
        """Decorator form: ``@registry.tool("computer_click", ...)``."""

        def decorator(handler: ToolHandler) -> ToolHandler:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=handler,
                    risk=risk,
                    group=group,
                    returns_image=returns_image,
                )
            )
            return handler

        return decorator

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # -- introspection ---------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            close = [n for n in self._tools if name.lower() in n.lower()][:5]
            raise ToolNotFound(
                f"unknown tool {name!r}",
                detail=(f"did you mean: {', '.join(close)}?" if close else f"available: {', '.join(self.names[:25])}"),
            )
        return spec

    def specs(self, *, groups: Iterable[str] | None = None, exclude: Iterable[str] = ()) -> list[ToolSpec]:
        wanted = set(groups) if groups else None
        skip = set(exclude)
        return [
            spec
            for name, spec in sorted(self._tools.items())
            if name not in skip and (wanted is None or spec.group in wanted)
        ]

    def to_anthropic(self, **kwargs) -> list[dict]:
        return [spec.to_anthropic() for spec in self.specs(**kwargs)]

    def to_openai(self, **kwargs) -> list[dict]:
        return [spec.to_openai() for spec in self.specs(**kwargs)]

    def to_mcp(self, **kwargs) -> list[dict]:
        return [spec.to_mcp() for spec in self.specs(**kwargs)]

    # -- dispatch --------------------------------------------------------

    def call(self, name: str, tool_input: dict | None, context: ToolContext) -> ToolResult:
        """Run a tool. Never raises for ordinary failure — returns a failed result."""
        started = time.monotonic()
        payload = dict(tool_input or {})
        try:
            spec = self.get(name)
        except ToolNotFound as exc:
            return ToolResult.failure(str(exc), **exc.to_dict())

        policy = context.policy or self.policy
        verdict: Verdict | None = None
        if policy is not None:
            verdict = policy.check(name, payload, risk=spec.risk)
            if verdict.decision is Decision.DENY:
                _audit(context, "tool_denied", tool=name, input=payload, verdict=verdict.to_dict())
                return ToolResult.failure(
                    f"BLOCKED by safety policy: {verdict.reason}",
                    tool=name,
                    decision="deny",
                    reason=verdict.reason,
                )
            if verdict.decision is Decision.ASK:
                approved = False
                if context.approver is not None:
                    try:
                        approved = bool(context.approver(name, payload, verdict))
                    except Exception as exc:
                        return ToolResult.failure(f"approval failed: {exc}", tool=name)
                if not approved:
                    _audit(context, "tool_refused", tool=name, input=payload, verdict=verdict.to_dict())
                    return ToolResult.failure(
                        f"NOT APPROVED by the user: {verdict.reason}. "
                        "Ask the user for permission, or choose a different approach.",
                        tool=name,
                        decision="ask",
                        reason=verdict.reason,
                    )

        try:
            validate_input(payload, spec.parameters, tool=name)
        except InvalidToolInput as exc:
            return ToolResult.failure(f"invalid input for {name}: {exc}", **exc.to_dict())

        _audit(context, "tool_call", tool=name, input=payload, risk=spec.risk.value)
        try:
            result = spec.handler(context, payload)
            if not isinstance(result, ToolResult):
                result = ToolResult.json(result if isinstance(result, dict) else {"result": result})
        except PermissionDenied as exc:
            result = ToolResult.failure(f"permission denied: {exc}", **exc.to_dict())
        except LaiError as exc:
            result = ToolResult.failure(f"{exc.code}: {exc}", **exc.to_dict())
        except Exception as exc:
            result = ToolResult.failure(
                f"tool {name!r} raised {type(exc).__name__}: {exc}", tool=name
            )

        result.duration = time.monotonic() - started
        if policy is not None and spec.risk is not Risk.READ and result.ok:
            policy.record()

        limit = getattr(getattr(context.config, "limits", None), "max_tool_output_chars", 0) or 0
        if limit:
            result = result.truncated(limit)

        _audit(
            context,
            "tool_result",
            tool=name,
            ok=result.ok,
            duration=round(result.duration, 3),
            summary=result.content[:400],
            images=len(result.images),
        )
        return result


# -- schema helpers ------------------------------------------------------


def _ensure_object_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out = dict(schema)
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def validate_input(payload: dict, schema: dict, *, tool: str = "") -> None:
    """Validate against the subset of JSON Schema our tools actually use.

    Deliberately small and dependency-free: models mostly get types right; what
    matters is catching missing required fields and bad enums with a message the
    model can act on.
    """
    if not isinstance(payload, dict):
        raise InvalidToolInput(f"{tool}: input must be an object")
    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    missing = [key for key in required if key not in payload or payload[key] is None]
    if missing:
        raise InvalidToolInput(
            f"missing required parameter(s): {', '.join(missing)}",
            detail=f"expected: {', '.join(sorted(properties))}",
        )

    for key, value in payload.items():
        rule = properties.get(key)
        if not isinstance(rule, dict) or value is None:
            continue
        expected = rule.get("type")
        if isinstance(expected, str):
            allowed = _TYPE_CHECKS.get(expected)
            # bool is a subclass of int; don't let True pass as an integer.
            if allowed and (
                not isinstance(value, allowed)
                or (expected in ("integer", "number") and isinstance(value, bool))
            ):
                raise InvalidToolInput(
                    f"parameter {key!r} must be {expected}, got {type(value).__name__}"
                )
        choices = rule.get("enum")
        if choices and value not in choices:
            raise InvalidToolInput(
                f"parameter {key!r} must be one of {choices}, got {value!r}"
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum, maximum = rule.get("minimum"), rule.get("maximum")
            if minimum is not None and value < minimum:
                raise InvalidToolInput(f"parameter {key!r} must be >= {minimum}, got {value}")
            if maximum is not None and value > maximum:
                raise InvalidToolInput(f"parameter {key!r} must be <= {maximum}, got {value}")


def _audit(context: ToolContext, kind: str, **data: Any) -> None:
    audit = getattr(context, "audit", None)
    if audit is not None:
        try:
            audit.write(kind, **data)
        except Exception:
            pass
