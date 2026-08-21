"""Refusing a backend outright.

A machine can have a working login for a model its owner does not want used —
on cost, on policy, or simply on preference. The point of this list is that
saying so once holds everywhere: auto-detection, the failover chain, the
listing and the menus. A denied backend that reappears in any of them is the
whole feature failing.
"""

from __future__ import annotations

import pytest

from lai.agent.providers.registry import Credential, build_chain
from lai.config import ProviderConfig, load_config
from lai.errors import ProviderError


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [Credential(name, "k", "https://x.test", "m", "test")
                 for name in ("zai", "cli:claude", "openai", "ollama")],
    )


def test_a_denied_backend_is_not_chosen_automatically():
    chain = build_chain(ProviderConfig(name="auto", deny=("zai",)))
    assert chain[0].name != "zai"


def test_a_denied_backend_is_not_a_fallback_either():
    """The whole point: it must not come back when something else fails."""
    names = [c.name for c in build_chain(ProviderConfig(name="zai", deny=("cli:claude",)))]
    assert "cli:claude" not in names
    assert "openai" in names, "the others are unaffected"


def test_asking_for_a_denied_backend_explicitly_is_an_error():
    """Silently substituting something else would be worse than refusing."""
    with pytest.raises(ProviderError, match="deny list"):
        build_chain(ProviderConfig(name="cli:claude", deny=("cli:claude",)))


def test_denying_everything_leaves_a_clear_failure():
    with pytest.raises(ProviderError, match="no model backend"):
        build_chain(ProviderConfig(name="auto", deny=("zai", "cli:claude", "openai", "ollama")))


def test_the_listing_hides_denied_backends(tmp_path, monkeypatch):
    from lai.models import discover

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    listed = {b.name for b in discover(probe_local=False, deny=("cli:claude",))}
    assert "cli:claude" not in listed


def test_names_are_matched_case_and_space_insensitively():
    names = [c.name for c in build_chain(ProviderConfig(name="zai", deny=("  CLI:Claude  ",)))]
    assert "cli:claude" not in names


def test_it_reads_from_the_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.delenv("LAI_DENY", raising=False)
    (tmp_path / "config.toml").write_text(
        '[provider]\ndeny = ["cli:claude", "anthropic"]\n', encoding="utf-8"
    )
    assert load_config().provider.deny == ("cli:claude", "anthropic")


def test_the_environment_can_deny_for_one_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setenv("LAI_DENY", "cli:claude,anthropic")
    assert load_config().provider.deny == ("cli:claude", "anthropic")


def test_nothing_is_denied_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.delenv("LAI_DENY", raising=False)
    assert load_config().provider.deny == ()
