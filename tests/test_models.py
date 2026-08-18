"""The backend catalogue and `lai models`.

Two things matter here. The listing must be honest — "installed" is not
"signed in", and "known" is not "working" — and it must never let a slow or
missing local server turn into a hang or a crash.
"""

from __future__ import annotations

import pytest

from lai.agent.providers import catalog
from lai.models import KIND_API, KIND_CLI, KIND_LOCAL, KNOWN, READY, Backend, check, discover

# -- the catalogue -------------------------------------------------------


def test_every_vendor_is_completely_specified():
    for vendor in catalog.ALL_VENDORS:
        assert vendor.name and vendor.label, vendor
        assert vendor.base_url.startswith("http"), vendor.name
        assert vendor.default_model, vendor.name
        assert vendor.local or vendor.env_keys, f"{vendor.name} is hosted but names no key"


def test_vendor_names_are_unique():
    names = [vendor.name for vendor in catalog.ALL_VENDORS]
    assert len(names) == len(set(names))


def test_hosted_vendors_link_somewhere_to_get_a_key():
    for vendor in catalog.VENDORS:
        assert vendor.signup.startswith("https://"), vendor.name


def test_local_vendors_need_no_key():
    for vendor in catalog.LOCAL_VENDORS:
        assert vendor.local
        assert not vendor.needs_key or vendor.name == "litellm"


def test_lookup_is_case_insensitive():
    assert catalog.get("GROQ") is catalog.get("groq")
    assert catalog.get("  deepseek ") is not None
    assert catalog.get("nope") is None


def test_env_key_lookup_prefers_the_first_variable():
    vendor = catalog.get("gemini")
    assert catalog.env_key_for(vendor, {"GOOGLE_API_KEY": "b"}) == "b"
    assert catalog.env_key_for(vendor, {"GEMINI_API_KEY": "a", "GOOGLE_API_KEY": "b"}) == "a"
    assert catalog.env_key_for(vendor, {}) == ""


def test_source_names_the_variable_that_was_used():
    vendor = catalog.get("groq")
    assert catalog.source_of(vendor, {"GROQ_API_KEY": "x"}) == "GROQ_API_KEY"
    assert catalog.source_of(catalog.get("ollama"), {}) == "local endpoint"


def test_a_blank_key_does_not_count():
    assert catalog.env_key_for(catalog.get("groq"), {"GROQ_API_KEY": "   "}) == ""


# -- discovery -----------------------------------------------------------


@pytest.fixture
def isolated(monkeypatch):
    """No real credentials, no real sockets."""
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    monkeypatch.setattr("lai.models._probe", lambda url, timeout: (False, "not running"))
    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which", lambda name: None)
    for vendor in catalog.ALL_VENDORS:
        for variable in vendor.env_keys:
            monkeypatch.delenv(variable, raising=False)


def test_discovery_lists_every_known_vendor(isolated):
    found = discover()
    names = {backend.name for backend in found}
    for vendor in catalog.ALL_VENDORS:
        assert vendor.name in names, f"{vendor.name} is missing from the listing"


def test_nothing_is_ready_on_a_bare_machine(isolated):
    assert [b for b in discover() if b.usable] == []


def test_a_key_in_the_environment_makes_a_vendor_ready(isolated, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    groq = next(b for b in discover() if b.name == "groq")
    assert groq.status == READY
    assert groq.detail == "GROQ_API_KEY"
    assert groq.kind == KIND_API


def test_a_running_local_server_is_ready(isolated, monkeypatch):
    monkeypatch.setattr("lai.models._probe",
                        lambda url, timeout: (True, "serving 2 model(s)") if "1234" in url else (False, "not running"))
    lmstudio = next(b for b in discover() if b.name == "lmstudio")
    assert lmstudio.status == READY and lmstudio.kind == KIND_LOCAL
    assert "serving" in lmstudio.detail


def test_an_installed_cli_shows_as_needing_a_sign_in(isolated, monkeypatch):
    monkeypatch.setattr("lai.agent.providers.cli_agent.shutil.which",
                        lambda name: "/usr/bin/claude" if name == "claude" else None)
    claude = next(b for b in discover() if b.name == "cli:claude")
    assert claude.kind == KIND_CLI
    assert claude.detail == "installed"
    assert "sign in" in claude.hint or "ANTHROPIC_API_KEY" in claude.hint


def test_a_missing_cli_is_only_known(isolated):
    assert next(b for b in discover() if b.name == "cli:codex").status == KNOWN


def test_ready_backends_come_first(isolated, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    statuses = [b.status for b in discover()]
    assert statuses == sorted(statuses, key=lambda s: {READY: 0, "auth": 1, KNOWN: 2}[s])


def test_a_detected_cli_is_marked_unverified(isolated, monkeypatch):
    """Being installed is not being signed in, and the listing must not pretend."""
    credential = type("C", (), {"provider": "cli:claude", "api_key": "", "model": "claude",
                                "source": "claude CLI on PATH"})()
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", lambda: [credential])
    claude = next(b for b in discover() if b.name == "cli:claude")
    assert claude.status == READY
    assert "not verified" in claude.detail
    assert claude.hint == "lai models test cli:claude"
    # Vision is a per-CLI fact now: claude reads staged screenshots, and the
    # listing must say so rather than blanket-denying it.
    assert claude.vision
    codex = next(b for b in discover() if b.name == "cli:codex")
    assert codex.vision, "codex gets read-access sandbox flags for the same purpose"


def test_discovery_survives_broken_credential_lookup(monkeypatch):
    def explode():
        raise RuntimeError("environment is a mess")

    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", explode)
    monkeypatch.setattr("lai.models._probe", lambda url, timeout: (False, ""))
    found = discover()
    assert any("environment is a mess" in b.detail for b in found)
    assert len(found) > 10, "one broken probe must not empty the listing"


def test_local_probing_can_be_skipped(monkeypatch):
    probed: list[str] = []
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    monkeypatch.setattr("lai.models._probe",
                        lambda url, timeout: (probed.append(url), (False, ""))[1])
    discover(probe_local=False)
    assert probed == []


def test_probe_reports_a_dead_endpoint(monkeypatch):
    import httpx

    from lai.models import _probe

    def explode(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", explode)
    assert _probe("http://127.0.0.1:9/v1", 0.1) == (False, "not running")


def test_probe_lists_served_models(monkeypatch):
    import httpx

    from lai.models import _probe

    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"id": "a"}, {"id": "b"}]}

    monkeypatch.setattr(httpx, "get", lambda url, timeout: Response())
    alive, detail = _probe("http://127.0.0.1:1234/v1", 0.1)
    assert alive and "2 model(s)" in detail and "a" in detail


def test_probe_tolerates_a_server_that_answers_with_junk(monkeypatch):
    import httpx

    from lai.models import _probe

    class Response:
        status_code = 200

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(httpx, "get", lambda url, timeout: Response())
    assert _probe("http://127.0.0.1:1234/v1", 0.1) == (True, "running")


def test_backend_serialises():
    import json

    payload = json.loads(json.dumps(Backend("x", "X", KIND_API, READY, "d", "m").to_dict()))
    assert payload["name"] == "x" and payload["status"] == READY


# -- check ---------------------------------------------------------------


def test_check_reports_a_working_backend(monkeypatch):
    class Turn:
        text = "OK"

    class Instance:
        name, model = "groq", "llama"
        closed = False

        def complete(self, messages, system=""):
            return Turn()

        def close(self):
            Instance.closed = True

    monkeypatch.setattr("lai.agent.providers.registry.build_provider", lambda config: Instance())
    works, detail = check("groq")
    assert works and "groq/llama" in detail and "OK" in detail
    assert Instance.closed


def test_check_reports_a_broken_backend(monkeypatch):
    def explode(config):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr("lai.agent.providers.registry.build_provider", explode)
    works, detail = check("groq")
    assert not works and "401" in detail


def test_check_spends_only_a_tiny_request(monkeypatch):
    seen: dict = {}

    def capture(config):
        seen["max_tokens"] = config.max_tokens
        raise RuntimeError("stop")

    monkeypatch.setattr("lai.agent.providers.registry.build_provider", capture)
    check("groq")
    assert seen["max_tokens"] <= 32
