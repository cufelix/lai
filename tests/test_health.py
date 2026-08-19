"""Remembering what each backend did last time.

A quota lasts hours. Without a memory every run rediscovers it the same way —
send, wait, 429, fail over — and the listing keeps calling a backend that has
refused all morning "ready now". The rules that matter: a stated recovery time
is believed, a backend that answers is healthy again whatever it did before,
and being wrong about any of it costs a retry rather than an outage.
"""

from __future__ import annotations

import json
import time

import pytest

from lai.agent.providers.health import (
    AUTH_COOLDOWN,
    MAX_COOLDOWN,
    QUOTA_COOLDOWN,
    TRANSIENT_COOLDOWN,
    Health,
    classify,
    cooling,
    note_failure,
    note_success,
    read,
)

QUOTA = ("zai: HTTP 429 from https://api.z.ai/api/anthropic (rate_limit_error: [1308]"
         "[Usage limit reached for 5 hour. Your limit will reset at 2099-08-19 12:01:48])")


# -- classification ------------------------------------------------------


@pytest.mark.parametrize("reason", [
    "HTTP 429 rate limit", "Usage limit reached for 5 hour",
    "insufficient credit balance", "claude hit its session limit",
])
def test_a_quota_is_recognised(reason):
    kind, cooldown = classify(reason)
    assert kind == "quota" and cooldown == QUOTA_COOLDOWN


@pytest.mark.parametrize("reason", ["HTTP 401 unauthorized", "invalid api key", "Please run /login"])
def test_an_authentication_failure_is_recognised(reason):
    kind, cooldown = classify(reason)
    assert kind == "auth" and cooldown == AUTH_COOLDOWN


def test_anything_else_gets_a_short_rest():
    kind, cooldown = classify("connection reset by peer")
    assert kind == "error" and cooldown == TRANSIENT_COOLDOWN


# -- recording -----------------------------------------------------------


def test_a_stated_reset_time_is_believed(tmp_path):
    """Vendors usually say when they recover; guessing an hour wastes the rest."""
    entry = note_failure(tmp_path, "zai", QUOTA)
    assert entry.kind == "quota"
    assert entry.until > time.time() + QUOTA_COOLDOWN, "the stated time is far away"
    assert entry.cooling


@pytest.mark.parametrize(("reason", "seconds"), [
    ("rate limited, retry in 30 seconds", 30),
    ("overloaded — try again after 5 minutes", 300),
])
def test_a_stated_retry_delay_is_believed(tmp_path, reason, seconds):
    entry = note_failure(tmp_path, "openai", reason)
    assert seconds - 10 <= entry.recovers_in <= seconds + 10


def test_a_recovery_time_in_the_past_falls_back_to_the_default(tmp_path):
    entry = note_failure(tmp_path, "zai", "quota exhausted, resets at 2001-01-01 00:00:00")
    assert entry.recovers_in > 0


def test_a_cooldown_is_capped(tmp_path):
    entry = note_failure(tmp_path, "zai", "quota, resets at 2999-01-01 00:00:00")
    assert entry.recovers_in <= MAX_COOLDOWN


def test_a_backend_that_answers_is_healthy_again(tmp_path):
    note_failure(tmp_path, "zai", QUOTA)
    note_success(tmp_path, "zai")
    assert "zai" not in read(tmp_path)


def test_only_cooling_backends_are_listed(tmp_path):
    note_failure(tmp_path, "zai", QUOTA)
    note_failure(tmp_path, "openai", "connection reset")
    (tmp_path / "backends.json").write_text(
        json.dumps({
            "zai": {"reason": "quota", "kind": "quota", "at": 0, "until": time.time() + 600},
            "old": {"reason": "x", "kind": "error", "at": 0, "until": time.time() - 10},
        }),
        encoding="utf-8",
    )
    assert set(cooling(tmp_path)) == {"zai"}


def test_the_description_says_what_and_how_long(tmp_path):
    entry = note_failure(tmp_path, "zai", "HTTP 429 rate limit")
    text = entry.describe()
    assert "out of quota" in text and "retry in" in text
    assert Health("x", until=0).describe() == ""


# -- never getting in the way --------------------------------------------


def test_a_corrupt_health_file_is_ignored(tmp_path):
    (tmp_path / "backends.json").write_text("not json {{{", encoding="utf-8")
    assert read(tmp_path) == {}
    note_failure(tmp_path, "zai", "quota")
    assert "zai" in read(tmp_path), "and it recovers on the next write"


def test_absurd_entries_are_skipped(tmp_path):
    (tmp_path / "backends.json").write_text(
        json.dumps({"good": {"until": 1}, "bad": "not a dict", "worse": {"until": "soon"}}),
        encoding="utf-8",
    )
    assert set(read(tmp_path)) == {"good"}


def test_an_unwritable_home_does_not_raise(tmp_path):
    note_failure(tmp_path / "nope" / "deeper" / "still", "zai", "quota")


# -- the chain uses it ---------------------------------------------------


def test_a_cooling_standby_is_skipped(tmp_path, monkeypatch):
    from lai.agent.providers.registry import Credential, build_chain
    from lai.config import ProviderConfig

    def credential(name):
        return Credential(name, "k", "https://x.test", "m", "test")

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [credential("zai"), credential("openai"), credential("ollama")],
    )
    note_failure(tmp_path, "openai", QUOTA)

    names = [c.name for c in build_chain(ProviderConfig(name="zai"), home=tmp_path)]
    assert "openai" not in names, "asking again before it recovers just wastes a turn"
    assert "ollama" in names


def test_the_configured_backend_is_tried_even_while_cooling(tmp_path, monkeypatch):
    """The user asked for it. Being wrong about a recovery time must cost a
    retry, never an outage."""
    from lai.agent.providers.registry import Credential, build_chain
    from lai.config import ProviderConfig

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [Credential("zai", "k", "u", "m", "t")],
    )
    note_failure(tmp_path, "zai", QUOTA)
    assert [c.name for c in build_chain(ProviderConfig(name="zai"), home=tmp_path)] == ["zai"]


def test_the_chain_writes_down_why_a_backend_stepped_aside(tmp_path):
    from lai.agent.providers.base import Message
    from lai.agent.providers.fallback import Candidate, FallbackProvider
    from lai.errors import ProviderError
    from tests.test_fallback import FakeProvider

    provider = FallbackProvider(
        [
            Candidate("zai", lambda: FakeProvider("zai", fails_with=ProviderError(QUOTA))),
            Candidate("ollama", lambda: FakeProvider("ollama")),
        ],
        home=tmp_path,
    )
    provider.complete([Message.user("hi")])

    recorded = read(tmp_path)
    assert recorded["zai"].kind == "quota"
    assert "ollama" not in recorded, "the one that answered is healthy"


def test_auto_does_not_start_on_a_backend_that_is_refusing(tmp_path, monkeypatch):
    """"auto" means "whichever works" — starting on a known-exhausted key
    spends a turn discovering what was already written down."""
    from lai.agent.providers.registry import Credential, build_chain
    from lai.config import ProviderConfig

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [Credential(n, "k", "u", "m", "t") for n in ("zai", "openai", "ollama")],
    )
    note_failure(tmp_path, "zai", QUOTA)
    chain = build_chain(ProviderConfig(name="auto"), home=tmp_path)
    assert chain[0].name == "openai"


def test_auto_still_starts_somewhere_when_everything_is_resting(tmp_path, monkeypatch):
    """A stale cooldown must never leave the machine with no agent at all."""
    from lai.agent.providers.registry import Credential, build_chain
    from lai.config import ProviderConfig

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [Credential(n, "k", "u", "m", "t") for n in ("zai", "openai")],
    )
    note_failure(tmp_path, "zai", QUOTA)
    note_failure(tmp_path, "openai", QUOTA)
    assert build_chain(ProviderConfig(name="auto"), home=tmp_path)[0].name == "zai"
