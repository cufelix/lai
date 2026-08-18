"""ModelRouter: escalation rules, single-provider degradation, usage tracking."""

from __future__ import annotations

from lai.agent.providers.base import Message, TextBlock, TurnResult, Usage
from lai.agent.router import (
    DEFAULT_RULES,
    ModelRouter,
    RouteState,
    RoutingRule,
    build_router,
)


class FakeProvider:
    def __init__(self, name: str, model: str = "fake-model") -> None:
        self.name = name
        self.model = model

    def complete(self, messages, *, system="", tools=None, stream=None) -> TurnResult:
        return TurnResult(message=Message("assistant", [TextBlock("ok")]), usage=Usage(1, 1))

    def close(self) -> None:
        pass


def two_providers() -> dict[str, FakeProvider]:
    return {"cheap": FakeProvider("cheap"), "strong": FakeProvider("strong")}


# -- single-provider degradation -----------------------------------------


def test_single_provider_is_always_returned_regardless_of_rules():
    only = FakeProvider("only")
    router = ModelRouter({"only": only})
    assert router.choose(step=1, task="anything") is only
    assert router.choose(step=99, task="design the architecture", failures=5) is only


def test_router_requires_at_least_one_provider():
    import pytest

    with pytest.raises(ValueError):
        ModelRouter({})


# -- built-in escalation rules --------------------------------------------


def test_first_step_uses_strong_model():
    providers = two_providers()
    router = ModelRouter(providers)
    chosen = router.choose(step=1, task="click the button")
    assert chosen is providers["strong"]


def test_routine_later_step_uses_cheap_model():
    providers = two_providers()
    router = ModelRouter(providers)
    chosen = router.choose(step=5, task="click the next button", failures=0)
    assert chosen is providers["cheap"]


def test_failure_escalates_to_strong_model():
    providers = two_providers()
    router = ModelRouter(providers)
    chosen = router.choose(step=5, task="click the button", failures=1)
    assert chosen is providers["strong"]


def test_complexity_signal_escalates_to_strong_model():
    providers = two_providers()
    router = ModelRouter(providers)
    chosen = router.choose(step=5, task="investigate the root cause of this race condition")
    assert chosen is providers["strong"]


def test_plain_task_text_does_not_trigger_complexity_signal():
    providers = two_providers()
    router = ModelRouter(providers)
    chosen = router.choose(step=5, task="click save")
    assert chosen is providers["cheap"]


# -- data-driven rules ---------------------------------------------------


def test_custom_rules_override_defaults():
    providers = two_providers()
    always_strong = RoutingRule("always_strong", "strong", "test rule", lambda s: True)
    router = ModelRouter(providers, rules=[always_strong])
    assert router.choose(step=10, task="anything") is providers["strong"]


def test_rule_targeting_an_unknown_key_falls_through_to_default():
    providers = two_providers()
    bogus = RoutingRule("bogus", "nonexistent", "test rule", lambda s: True)
    router = ModelRouter(providers, rules=[bogus])
    assert router.choose(step=10, task="anything") is providers["cheap"]


def test_default_model_falls_back_to_first_provider_if_key_unknown():
    providers = two_providers()
    router = ModelRouter(providers, rules=[], default_model="does-not-exist")
    chosen = router.choose(step=10, task="anything")
    assert chosen in providers.values()


def test_route_state_is_what_predicates_receive():
    seen: list[RouteState] = []
    rule = RoutingRule("capture", "strong", "test", lambda s: seen.append(s) or False)
    providers = two_providers()
    router = ModelRouter(providers, rules=[rule])
    router.choose(step=3, task="hello", last_tool="ui_click", failures=2)
    assert seen[0] == RouteState(step=3, task="hello", last_tool="ui_click", failures=2)


def test_first_matching_rule_wins():
    providers = two_providers()
    first = RoutingRule("first", "strong", "r1", lambda s: True)
    second = RoutingRule("second", "cheap", "r2", lambda s: True)
    router = ModelRouter(providers, rules=[first, second])
    assert router.choose(step=10, task="x") is providers["strong"]


def test_default_rules_are_exposed_and_ordered():
    assert isinstance(DEFAULT_RULES, tuple)
    assert DEFAULT_RULES[0].name == "first_step_plans"


# -- usage tracking --------------------------------------------------


def test_usage_by_model_starts_empty():
    router = ModelRouter(two_providers())
    assert router.usage_by_model == {}


def test_record_usage_attributes_to_last_chosen_model():
    router = ModelRouter(two_providers())
    router.choose(step=5, task="routine step")  # -> cheap
    router.record_usage(Usage(input_tokens=10, output_tokens=5))
    assert router.usage_by_model["cheap"].total == 15


def test_record_usage_accumulates_across_calls():
    router = ModelRouter(two_providers())
    router.choose(step=5, task="routine")
    router.record_usage(Usage(10, 0))
    router.record_usage(Usage(5, 0))
    assert router.usage_by_model["cheap"].input_tokens == 15


def test_record_usage_accepts_explicit_model_key():
    router = ModelRouter(two_providers())
    router.record_usage(Usage(1, 1), model="strong")
    assert "strong" in router.usage_by_model
    assert "cheap" not in router.usage_by_model


def test_total_usage_sums_every_model():
    router = ModelRouter(two_providers())
    router.record_usage(Usage(10, 0), model="cheap")
    router.record_usage(Usage(0, 20), model="strong")
    assert router.total_usage() == Usage(10, 20)


def test_single_provider_usage_is_still_tracked_by_its_own_key():
    router = ModelRouter({"only": FakeProvider("only")})
    router.choose(step=1, task="x")
    router.record_usage(Usage(3, 4))
    assert router.usage_by_model["only"].total == 7


# -- build_router ----------------------------------------------------


def test_build_router_with_explicit_providers_bypasses_config():
    providers = two_providers()
    router = build_router(config=object(), providers=providers)
    assert router.providers == providers


def test_build_router_single_injected_provider_always_wins():
    only = {"solo": FakeProvider("solo")}
    router = build_router(config=object(), providers=only)
    assert router.choose(step=1, task="design a new architecture") is only["solo"]
