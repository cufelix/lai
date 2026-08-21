"""Provider failover.

The property that matters: a backend refusing for a reason another backend
would not share — quota, auth, an outage — must move the run on rather than
end it, and the change must be visible. A request that is simply wrong must
not burn through every backend on the machine.
"""

from __future__ import annotations

import pytest

from lai.agent.providers.base import Message, TextBlock, TurnResult, Usage
from lai.agent.providers.fallback import Candidate, FallbackProvider, should_switch
from lai.errors import ProviderError


class FakeProvider:
    """A backend that fails a set number of times, then answers."""

    def __init__(self, name, *, fails_with=None, model="m", answer="ok"):
        self.name = name
        self.model = model
        self.fails_with = fails_with
        self.answer = answer
        self.calls = 0
        self.closed = False

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.fails_with is not None:
            raise self.fails_with
        return TurnResult(
            message=Message("assistant", [TextBlock(self.answer)]),
            stop_reason="end_turn", usage=Usage(), model=self.model,
        )

    def close(self):
        self.closed = True


def chain(*providers, on_switch=None):
    return FallbackProvider(
        [Candidate(p.name, (lambda p=p: p)) for p in providers], on_switch=on_switch
    )


# -- classification ------------------------------------------------------


@pytest.mark.parametrize("text", [
    "zai: HTTP 429 (rate_limit_error: Usage limit reached for 5 hour)",
    "HTTP 401 unauthorized",
    "HTTP 529 overloaded_error",
    "insufficient credit balance",
    "claude exited 1 (Please run /login)",
    "connection refused",
    "read timed out",
])
def test_a_backend_specific_failure_moves_on(text):
    assert should_switch(ProviderError(text))


@pytest.mark.parametrize("text", [
    "HTTP 400: messages.3: image content is malformed",
    "unknown tool 'window_lst'",
    "could not parse the reply",
])
def test_a_request_that_is_simply_wrong_does_not(text):
    assert not should_switch(ProviderError(text)), (
        "a bad request fails identically everywhere; switching just wastes backends"
    )


# -- switching -----------------------------------------------------------


def test_a_rate_limited_backend_hands_over():
    primary = FakeProvider("zai", fails_with=ProviderError("HTTP 429 rate_limit_error"))
    standby = FakeProvider("cli:claude", answer="done")
    provider = chain(primary, standby)

    turn = provider.complete([Message.user("hi")])
    assert turn.message.content[0].text == "done"
    assert provider.name == "cli:claude", "it must report who actually answered"
    assert standby.calls == 1


def test_the_switch_is_announced_with_a_reason():
    seen = []
    provider = chain(
        FakeProvider("zai", fails_with=ProviderError("HTTP 429 Usage limit reached for 5 hour")),
        FakeProvider("ollama"),
        on_switch=lambda a, b, why: seen.append((a, b, why)),
    )
    provider.complete([Message.user("hi")])
    assert len(seen) == 1
    origin, destination, reason = seen[0]
    assert (origin, destination) == ("zai", "ollama")
    assert "Usage limit" in reason


def test_it_stays_switched_for_the_rest_of_the_run():
    """Flapping between two models mid-task produces incoherent behaviour."""
    primary = FakeProvider("zai", fails_with=ProviderError("429 rate limit"))
    standby = FakeProvider("ollama")
    provider = chain(primary, standby)

    provider.complete([Message.user("one")])
    provider.complete([Message.user("two")])
    assert primary.calls == 1, "the exhausted backend must not be tried again"
    assert standby.calls == 2


def test_a_failure_no_other_backend_would_fix_is_raised():
    primary = FakeProvider("zai", fails_with=ProviderError("HTTP 400 malformed request"))
    standby = FakeProvider("ollama")
    provider = chain(primary, standby)

    with pytest.raises(ProviderError, match="400"):
        provider.complete([Message.user("hi")])
    assert standby.calls == 0


def test_the_last_backend_raises_rather_than_disappearing():
    provider = chain(
        FakeProvider("zai", fails_with=ProviderError("429 rate limit")),
        FakeProvider("ollama", fails_with=ProviderError("connection refused")),
    )
    with pytest.raises(ProviderError, match="connection refused"):
        provider.complete([Message.user("hi")])


def test_a_standby_that_cannot_be_built_is_skipped_not_fatal():
    def explode():
        raise ProviderError("ollama: no API key configured")

    provider = FallbackProvider([
        Candidate("zai", lambda: FakeProvider("zai", fails_with=ProviderError("429 rate limit"))),
        Candidate("ollama", explode),
        Candidate("cli:claude", lambda: FakeProvider("cli:claude", answer="from the CLI")),
    ])
    turn = provider.complete([Message.user("hi")])
    assert turn.message.content[0].text == "from the CLI"
    assert "ollama" in provider.failures


def test_the_failed_backend_is_closed_on_the_way_out():
    primary = FakeProvider("zai", fails_with=ProviderError("429 rate limit"))
    provider = chain(primary, FakeProvider("ollama"))
    provider.complete([Message.user("hi")])
    assert primary.closed, "a dropped backend must not leak its connection pool"


def test_failures_are_recorded_for_the_interface():
    provider = chain(
        FakeProvider("zai", fails_with=ProviderError("HTTP 429 Usage limit reached")),
        FakeProvider("ollama"),
    )
    provider.complete([Message.user("hi")])
    assert "Usage limit reached" in provider.failures["zai"]


def test_a_transport_blowing_up_is_still_a_backend_failure():
    class Exploding(FakeProvider):
        def complete(self, messages, **kwargs):
            self.calls += 1
            raise ConnectionResetError("connection reset by peer")

    provider = chain(Exploding("zai"), FakeProvider("ollama", answer="took over"))
    assert provider.complete([Message.user("hi")]).message.content[0].text == "took over"


def test_an_empty_chain_is_refused():
    with pytest.raises(ProviderError, match="no model backend"):
        FallbackProvider([])


def test_the_chain_is_introspectable():
    provider = chain(FakeProvider("zai"), FakeProvider("ollama"))
    assert provider.chain == ["zai", "ollama"]
    assert provider.model == "m"


# -- chain construction --------------------------------------------------


def credential(name, key="k", model="m"):
    from lai.agent.providers.registry import Credential

    return Credential(name, key, "https://example.test", model, "test")


def test_the_chain_puts_the_configured_backend_first(monkeypatch):
    from lai.agent.providers.registry import build_chain
    from lai.config import ProviderConfig

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [credential("zai"), credential("openai"), credential("ollama")],
    )
    names = [c.name for c in build_chain(ProviderConfig(name="openai"))]
    assert names[0] == "openai"
    assert set(names[1:]) == {"zai", "ollama"}


def test_an_explicit_chain_is_honoured_in_order(monkeypatch):
    from lai.agent.providers.registry import build_chain
    from lai.config import ProviderConfig

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [credential("zai"), credential("openai"), credential("ollama")],
    )
    config = ProviderConfig(name="zai", fallback=("ollama", "openai"))
    assert [c.name for c in build_chain(config)] == ["zai", "ollama", "openai"]


def test_failover_can_be_turned_off(monkeypatch):
    from lai.agent.providers.registry import build_chain
    from lai.config import ProviderConfig

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [credential("zai"), credential("openai")],
    )
    assert [c.name for c in build_chain(ProviderConfig(name="zai", fallback=()))] == ["zai"]


def test_standbys_are_not_built_until_they_are_needed(monkeypatch):
    """Building a backend can probe a socket or spawn a CLI — not on every start."""
    from lai.agent.providers.registry import build_chain
    from lai.config import ProviderConfig

    built = []
    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [credential("zai"), credential("openai")],
    )
    monkeypatch.setattr(
        "lai.agent.providers.registry._instantiate",
        lambda name, config, cred: built.append(name) or FakeProvider(name),
    )
    chain_ = build_chain(ProviderConfig(name="zai"))
    assert built == [], "constructing the chain must not construct the backends"
    chain_[0].build()
    assert built == ["zai"]


def test_a_standby_never_inherits_the_primary_key(monkeypatch):
    """A z.ai key aimed at OpenAI is not a fallback, it is a confusing 401."""
    from lai.agent.providers.registry import build_chain
    from lai.config import ProviderConfig

    seen = {}
    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [credential("zai"), credential("openai")],
    )
    monkeypatch.setattr(
        "lai.agent.providers.registry._instantiate",
        lambda name, config, cred: seen.setdefault(name, config) or FakeProvider(name),
    )
    for candidate in build_chain(ProviderConfig(name="zai", model="glm-5.2", api_key="secret")):
        candidate.build()
    assert seen["zai"].api_key == "secret" and seen["zai"].model == "glm-5.2"
    assert seen["openai"].api_key == "" and seen["openai"].model == ""


def test_config_reads_the_chain_from_the_file(tmp_path, monkeypatch):
    from lai.config import load_config

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.delenv("LAI_FALLBACK", raising=False)
    (tmp_path / "config.toml").write_text(
        '[provider]\nname = "zai"\nfallback = ["cli:claude", "ollama"]\n', encoding="utf-8"
    )
    assert load_config().provider.fallback == ("cli:claude", "ollama")


def test_failover_defaults_to_on(tmp_path, monkeypatch):
    from lai.config import load_config

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.delenv("LAI_FALLBACK", raising=False)
    assert load_config().provider.fallback == ("auto",)


@pytest.mark.parametrize("value", ["off", "none", "false"])
def test_the_environment_can_turn_failover_off(tmp_path, monkeypatch, value):
    from lai.config import load_config

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setenv("LAI_FALLBACK", value)
    assert load_config().provider.fallback == ()


# -- the loop tells the user ---------------------------------------------


def test_the_loop_announces_a_switch_between_steps():
    """A different model answering mid-run is news, not an implementation detail."""
    from lai.agent.loop import Agent

    events = []
    provider = chain(
        FakeProvider("zai", fails_with=ProviderError("HTTP 429 Usage limit reached")),
        FakeProvider("ollama"),
    )
    agent = Agent.__new__(Agent)
    agent.provider = provider
    agent.session = type("S", (), {"messages": []})()
    agent._system_prompt = ""
    agent.registry = type("R", (), {"to_anthropic": lambda self: []})()
    agent._tool_schemas = []
    agent.audit = type("A", (), {"write": lambda self, *a, **k: None})()
    agent._emit = lambda kind, payload: events.append((kind, payload))

    agent._model_turn()
    switches = [payload for kind, payload in events if kind == "provider_switch"]
    assert switches and switches[0]["from"] == "zai" and switches[0]["to"] == "ollama"
    assert "Usage limit" in switches[0]["reason"]


def test_auto_standbys_prefer_a_backend_that_can_finish_the_job(monkeypatch):
    """A 2B local model is a last resort, not the first thing tried after a quota."""
    from lai.agent.providers.registry import build_chain
    from lai.config import ProviderConfig

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [
            credential("zai"),
            credential("ollama", key=""),
            credential("cli:claude", key=""),
            credential("openai"),
        ],
    )
    names = [c.name for c in build_chain(ProviderConfig(name="zai"))]
    assert names[0] == "zai"
    assert names.index("openai") < names.index("cli:claude") < names.index("ollama")


def test_the_terminal_shows_a_switch_when_it_happens():
    """A different model taking over must be visible, not buried in a log."""
    from lai.cli import _make_reporter

    written = []
    out = type("O", (), {
        "write": lambda self, text="", **kw: written.append(text),
        "raw": lambda self, text: None,
        "rule": lambda self, title="": None,
        "spinner": lambda self, text: None,
    })()
    _make_reporter(out)("provider_switch", {
        "from": "zai", "to": "cli:claude", "model": "claude", "reason": "HTTP 429 Usage limit reached",
    })
    text = "\n".join(written)
    assert "zai stepped aside" in text and "cli:claude" in text and "429" in text


def test_a_cli_that_keeps_exiting_non_zero_hands_over():
    """Observed live: `claude` failed identically three times and killed the run.

    By the time it reaches the chain the provider has already spent its own
    retries, so whatever it blames, it is broken for this run.
    """
    detail = ('claude exited 1 ({"is_error":true,"terminal_reason":"api_error",'
              '"input_tokens":0,"output_tokens":0})')
    assert should_switch(ProviderError(detail))


def test_a_cli_producing_no_output_hands_over():
    assert should_switch(ProviderError("claude produced no output"))


# -- context budget ------------------------------------------------------


def test_the_chain_reports_the_active_backends_context():
    """Compaction has to size itself against whoever is actually answering."""
    small = FakeProvider("cli:claude")
    small.context_chars = 96_000
    provider = chain(FakeProvider("zai"), small)
    assert provider.context_chars == 0, "an API provider declares no character cap"
    provider.index, provider.active = 1, small
    assert provider.context_chars == 96_000


def test_an_exhausted_chain_explains_itself_on_the_next_turn():
    """Observed live: after five backends stepped aside, the sixth turn raised
    'NoneType object has no attribute complete' — a crash where a diagnosis
    belonged."""
    provider = chain(
        FakeProvider("zai", fails_with=ProviderError("HTTP 429 Usage limit reached")),
        FakeProvider("ollama", fails_with=ProviderError("network error: timed out")),
    )
    with pytest.raises(ProviderError, match="timed out"):
        provider.complete([Message.user("one")])

    with pytest.raises(ProviderError) as info:
        provider.complete([Message.user("two")])
    message = str(info.value)
    assert "all 2 model backend(s) failed" in message
    assert "zai" in message and "Usage limit" in message
    assert "ollama" in message and "timed out" in message
    assert "lai models" in message


def test_the_exhausted_error_survives_having_no_recorded_reasons():
    provider = chain(FakeProvider("only", fails_with=ProviderError("429 rate limit")))
    provider.active, provider.failures = None, {}
    assert "no backend was usable" in str(provider.exhausted())


def test_an_exhausted_chain_still_names_who_was_answering_last():
    """The switch event is emitted after the turn, by which point the chain may
    already be spent — reporting 'ollama/' with no model helps nobody."""
    provider = chain(
        FakeProvider("zai", fails_with=ProviderError("429 rate limit")),
        FakeProvider("ollama", model="qwen3-vl:2b", fails_with=ProviderError("timed out")),
    )
    with pytest.raises(ProviderError):
        provider.complete([Message.user("hi")])
    assert provider.name == "ollama"
    assert provider.model == "qwen3-vl:2b"
