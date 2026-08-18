"""Model routing: not every step deserves the expensive model.

Most of an agent loop's steps are perception-heavy — read the screen, click a
button, check the result — and a fast/cheap model handles those fine. A
minority of steps (the first one, where the plan gets set; any step right
after something went wrong; a step whose task text signals real complexity)
benefit from the strongest available model. :class:`ModelRouter` picks between
providers on each step using a small, data-driven rule list rather than
hand-rolled conditionals, so the policy can grow by appending a
:class:`RoutingRule` instead of editing branching logic.

Non-obvious design decision: routing is *stateless per call* — ``choose()``
takes everything it needs as arguments rather than reading them off a shared
mutable "current state" object. That makes the router trivially safe to call
from a driving loop without worrying about staleness, and trivial to test:
every test just constructs a :class:`RouteState` worth of arguments and checks
which provider comes back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ..config import Config
from ..errors import ProviderError
from .providers.base import Provider, Usage
from .providers.registry import build_provider

# Per-provider-family "cheap" model overrides used by build_router() when the
# caller hasn't supplied its own provider map. Picked to be fast/inexpensive
# siblings of the family's default model; not exhaustive on purpose — a
# family with no known cheap sibling (e.g. a local ollama model) just routes
# everything to its one available provider.
_CHEAP_MODEL_OVERRIDES: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "zai": "glm-4.5-flash",
    "openai": "gpt-4o-mini",
    "openrouter": "anthropic/claude-haiku-4.5",
}

# Substrings in a task description that suggest a step is not routine.
# Deliberately broad and cheap to check (a single lowercase substring scan)
# rather than an attempt at real intent classification.
COMPLEXITY_SIGNALS: tuple[str, ...] = (
    "architecture", "design", "refactor", "security", "vulnerab", "algorithm",
    "concurren", "race condition", "root cause", "investigat", "compare",
    "trade-off", "tradeoff", "migrat", "complex", "strategy", "debug",
    "why does", "why is", "deadlock", "optimi",
)


@dataclass(frozen=True, slots=True)
class RouteState:
    """What a routing rule is allowed to look at."""

    step: int
    task: str
    last_tool: str = ""
    failures: int = 0


@dataclass(frozen=True, slots=True)
class RoutingRule:
    """One entry in the routing policy.

    ``target`` is a key into :attr:`ModelRouter.providers` (conventionally
    ``"cheap"`` or ``"strong"``, but the router does not care what the keys
    are named). ``predicate`` decides whether this rule fires for a given
    :class:`RouteState`; the first matching rule in the list wins.
    """

    name: str
    target: str
    reason: str
    predicate: Callable[[RouteState], bool]


def _has_complexity_signal(task: str) -> bool:
    lowered = task.lower()
    return any(signal in lowered for signal in COMPLEXITY_SIGNALS)


DEFAULT_RULES: tuple[RoutingRule, ...] = (
    RoutingRule(
        "first_step_plans", "strong",
        "the first step sets the plan for everything after it",
        lambda s: s.step <= 1,
    ),
    RoutingRule(
        "after_failure", "strong",
        "the cheap model's last attempt did not work; escalate instead of repeating it",
        lambda s: s.failures > 0,
    ),
    RoutingRule(
        "complexity_signal", "strong",
        "the task text signals nontrivial reasoning",
        lambda s: _has_complexity_signal(s.task),
    ),
)


class ModelRouter:
    """Chooses a :class:`Provider` per step from a small named pool."""

    def __init__(
        self,
        providers: dict[str, Provider],
        rules: Sequence[RoutingRule] = DEFAULT_RULES,
        *,
        default_model: str = "cheap",
    ) -> None:
        if not providers:
            raise ValueError("ModelRouter requires at least one provider")
        self.providers: dict[str, Provider] = dict(providers)
        self.rules: tuple[RoutingRule, ...] = tuple(rules)
        self.default_model = default_model if default_model in self.providers else next(iter(self.providers))
        self.usage_by_model: dict[str, Usage] = {}
        self.last_model: str = self.default_model

    def choose(self, *, step: int, task: str, last_tool: str = "", failures: int = 0) -> Provider:
        # A single-provider deployment (the common case: one API key
        # configured) must still work, unconditionally — routing degrades to
        # "there is only one choice" rather than raising or guessing.
        if len(self.providers) == 1:
            (only_key,) = self.providers.keys()
            self.last_model = only_key
            return self.providers[only_key]

        state = RouteState(step=step, task=task, last_tool=last_tool, failures=failures)
        for rule in self.rules:
            if rule.target in self.providers and rule.predicate(state):
                self.last_model = rule.target
                return self.providers[rule.target]

        self.last_model = self.default_model
        return self.providers[self.default_model]

    def record_usage(self, usage: Usage, *, model: str | None = None) -> None:
        """Attribute ``usage`` to a model key for cost tracking.

        Defaults to whichever key :meth:`choose` last returned, so a caller
        that just wants "count what I used" doesn't have to track keys itself.
        """
        key = model or self.last_model
        self.usage_by_model[key] = self.usage_by_model.get(key, Usage()) + usage

    def total_usage(self) -> Usage:
        total = Usage()
        for usage in self.usage_by_model.values():
            total = total + usage
        return total


def build_router(config: Config, *, providers: dict[str, Provider] | None = None) -> ModelRouter:
    """Construct a router, either from an explicit provider map or from config.

    With no ``providers`` supplied, this builds a "strong" provider from
    ``config.provider`` (the same way the rest of LAI does) and tries to build
    a cheaper sibling from the same credentials via :data:`_CHEAP_MODEL_OVERRIDES`.
    If that family has no known cheap sibling, or the cheap variant can't be
    built (e.g. the account lacks access to it), the router still ends up
    fully functional — both keys just point at the same provider.
    """
    if providers is not None:
        return ModelRouter(providers)

    strong = build_provider(config.provider)
    cheap_model = _CHEAP_MODEL_OVERRIDES.get(strong.name, "")
    if not cheap_model or cheap_model == strong.model:
        return ModelRouter({"strong": strong, "cheap": strong})

    cheap_config = replace(config.provider, model=cheap_model)
    try:
        cheap = build_provider(cheap_config)
    except ProviderError:
        cheap = strong
    return ModelRouter({"strong": strong, "cheap": cheap})


__all__ = [
    "COMPLEXITY_SIGNALS",
    "DEFAULT_RULES",
    "ModelRouter",
    "RouteState",
    "RoutingRule",
    "build_router",
]
