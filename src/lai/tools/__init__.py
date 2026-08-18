"""Tool layer: everything the agent can do, declared once and rendered per provider."""

from __future__ import annotations

import importlib

from ..safety.policy import PolicyEngine
from . import app, computer, system, ui, window
from .base import (
    ToolContext,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    validate_input,
)

CORE_MODULES = (computer, ui, window, app, system)

# Loaded lazily at build time, for two reasons: `agentic` reaches back into
# lai.agent (which imports this package, so a module-level import would be
# circular), and both families depend on optional pieces — tesseract, ffmpeg,
# a DBus session — that must degrade to "tool absent" rather than "LAI won't
# start".
OPTIONAL_MODULES = ("perception", "agentic")


def _load_optional() -> tuple[list, dict[str, str]]:
    modules, problems = [], {}
    for name in OPTIONAL_MODULES:
        try:
            modules.append(importlib.import_module(f".{name}", __package__))
        except Exception as exc:  # an optional family, by definition
            problems[name] = f"{type(exc).__name__}: {exc}"
    return modules, problems


def build_registry(
    *, policy: PolicyEngine | None = None, groups: set[str] | None = None
) -> ToolRegistry:
    """Create a registry with every available built-in tool registered.

    ``groups`` restricts which tool families load, e.g. ``{"ui", "window"}`` for a
    perception-only agent.
    """
    registry = ToolRegistry(policy=policy)
    optional, problems = _load_optional()
    registry.optional_problems = problems  # type: ignore[attr-defined]

    for module in (*CORE_MODULES, *optional):
        try:
            module.register(registry)
        except Exception as exc:  # one bad family must not break the rest
            problems[getattr(module, "__name__", "?").rsplit(".", 1)[-1]] = str(exc)

    if groups is not None:
        for spec in list(registry.specs()):
            if spec.group not in groups:
                registry.unregister(spec.name)
    return registry


__all__ = [
    "CORE_MODULES",
    "OPTIONAL_MODULES",
    "ToolContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_registry",
    "validate_input",
]
