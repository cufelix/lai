"""Error hierarchy for LAI.

Every failure that can reach a model is a :class:`LaiError` carrying a stable
``code``, so the agent loop can turn it into a structured tool result instead of
crashing the run.
"""

from __future__ import annotations


class LaiError(Exception):
    """Base class for all LAI errors."""

    code = "lai_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        out = {"error": self.code, "message": self.message}
        if self.detail:
            out["detail"] = self.detail
        return out

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.message} ({self.detail})" if self.detail else self.message


class ConfigError(LaiError):
    code = "config_error"


class BackendUnavailable(LaiError):
    """The requested OS backend (X11, AT-SPI, ...) is not usable here."""

    code = "backend_unavailable"


class DisplayError(LaiError):
    code = "display_error"


class WindowNotFound(LaiError):
    code = "window_not_found"


class ElementNotFound(LaiError):
    code = "element_not_found"


class AppNotFound(LaiError):
    code = "app_not_found"


class ToolError(LaiError):
    """A tool executed but failed in a way the model should see and react to."""

    code = "tool_error"


class ToolNotFound(ToolError):
    code = "tool_not_found"


class InvalidToolInput(ToolError):
    code = "invalid_tool_input"


class PermissionDenied(LaiError):
    """A safety policy blocked the action."""

    code = "permission_denied"


class BudgetExceeded(LaiError):
    """Step, time or token budget ran out."""

    code = "budget_exceeded"


class ProviderError(LaiError):
    code = "provider_error"


class SkillError(LaiError):
    code = "skill_error"


class Interrupted(LaiError):
    code = "interrupted"
