"""Asking a backend which models it serves.

A vendor's catalogue is not something to hard-code: OpenRouter carries hundreds
and adds more weekly, and a local Ollama serves whatever its owner pulled this
morning. What matters here is that the live answer is normalised honestly —
free before paid, roomy before cramped, and anything the endpoint declines to
say stays absent rather than guessed.
"""

from __future__ import annotations

import httpx
import pytest

from lai.agent.providers.listing import ModelInfo, fetch, search
from lai.errors import ProviderError

OPENROUTER = {
    "data": [
        {"id": "big/paid", "name": "Big Paid", "context_length": 200000,
         "pricing": {"prompt": "0.000003"}},
        {"id": "small/free:free", "name": "Small Free", "context_length": 32000,
         "pricing": {"prompt": "0"}},
        {"id": "mid/cheap", "name": "Mid", "context_length": 128000,
         "pricing": {"prompt": "0.0000005"}},
    ]
}


def _transport(payload, status=200):
    def handler(request):
        return httpx.Response(status, json=payload)

    return handler


@pytest.fixture
def fake_get(monkeypatch):
    def install(payload, status=200, capture=None):
        def get(url, headers=None, timeout=None):
            if capture is not None:
                capture.update({"url": url, "headers": headers or {}})
            if isinstance(payload, Exception):
                raise payload
            return httpx.Response(status, json=payload, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", get)

    return install


# -- reading the answer --------------------------------------------------


def test_models_come_back_normalised(fake_get):
    fake_get(OPENROUTER)
    models = fetch("https://openrouter.ai/api/v1")
    identifiers = [m.id for m in models]
    assert set(identifiers) == {"big/paid", "small/free:free", "mid/cheap"}


def test_free_models_sort_first_then_cheapest(fake_get):
    """What a person actually chooses on."""
    fake_get(OPENROUTER)
    models = fetch("https://x.test/v1")
    assert models[0].id == "small/free:free"
    assert [m.id for m in models[1:]] == ["mid/cheap", "big/paid"]


def test_prices_are_quoted_per_million_tokens(fake_get):
    fake_get(OPENROUTER)
    paid = next(m for m in fetch("https://x.test/v1") if m.id == "big/paid")
    assert paid.prompt_price == pytest.approx(3.0)
    assert "$3/M in" in paid.describe()


def test_a_free_suffix_counts_as_free(fake_get):
    fake_get({"data": [{"id": "vendor/model:free", "context_length": 8000}]})
    assert fetch("https://x.test/v1")[0].free


def test_an_endpoint_that_says_nothing_about_price_does_not_invent_one(fake_get):
    fake_get({"data": [{"id": "local-model"}]})
    model = fetch("https://x.test/v1")[0]
    assert model.prompt_price == -1
    assert "$" not in model.describe()


def test_a_bare_list_of_names_is_accepted(fake_get):
    """Some local servers answer with the simplest possible shape."""
    fake_get(["model-a", "model-b"])
    assert [m.id for m in fetch("https://x.test/v1")] == ["model-a", "model-b"]


def test_the_context_window_can_hide_in_top_provider(fake_get):
    fake_get({"data": [{"id": "m", "top_provider": {"context_length": 65536}}]})
    assert fetch("https://x.test/v1")[0].context == 65536


def test_nameless_entries_are_skipped(fake_get):
    fake_get({"data": [{"id": ""}, {"nothing": True}, "", {"id": "real"}]})
    assert [m.id for m in fetch("https://x.test/v1")] == ["real"]


# -- failing usefully ----------------------------------------------------


def test_a_refused_key_says_so(fake_get):
    fake_get({}, status=401)
    with pytest.raises(ProviderError, match="refused the key"):
        fetch("https://x.test/v1", "bad-key")


def test_an_error_status_carries_the_body(fake_get):
    fake_get({"error": "model list disabled"}, status=500)
    with pytest.raises(ProviderError, match="answered 500"):
        fetch("https://x.test/v1")


def test_an_unreachable_endpoint_names_the_url(fake_get):
    fake_get(httpx.ConnectError("refused"))
    with pytest.raises(ProviderError, match="could not reach"):
        fetch("http://127.0.0.1:9/v1")


def test_an_unexpected_shape_is_refused(fake_get):
    fake_get({"models": "not a list"})
    with pytest.raises(ProviderError, match="unexpected shape"):
        fetch("https://x.test/v1")


def test_the_key_is_sent_as_a_bearer_token(fake_get):
    seen: dict = {}
    fake_get(OPENROUTER, capture=seen)
    fetch("https://x.test/v1", "secret-key")
    assert seen["headers"]["authorization"] == "Bearer secret-key"
    assert seen["url"].endswith("/models")


def test_no_key_means_no_header(fake_get):
    seen: dict = {}
    fake_get(OPENROUTER, capture=seen)
    fetch("https://x.test/v1")
    assert "authorization" not in seen["headers"]


# -- searching -----------------------------------------------------------


def test_search_matches_every_word():
    models = [
        ModelInfo(id="anthropic/claude-sonnet-4.5", label="Claude Sonnet"),
        ModelInfo(id="anthropic/claude-opus-4", label="Claude Opus"),
        ModelInfo(id="openai/gpt-4o", label="GPT-4o"),
    ]
    assert [m.id for m in search(models, "claude sonnet")] == ["anthropic/claude-sonnet-4.5"]


def test_search_looks_at_the_label_too():
    models = [ModelInfo(id="x/y", label="Mistral Large")]
    assert search(models, "mistral") == models


def test_an_empty_search_keeps_everything():
    models = [ModelInfo(id="a"), ModelInfo(id="b")]
    assert search(models, "   ") == models


# -- the endpoint a backend name resolves to -----------------------------


def test_a_catalogued_vendor_resolves_to_its_url(monkeypatch):
    from lai.models import endpoint_for

    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    base_url, _key = endpoint_for("openrouter")
    assert "openrouter.ai" in base_url


def test_a_discovered_credential_wins_because_it_carries_the_key(monkeypatch):
    from lai.agent.providers.registry import Credential
    from lai.models import endpoint_for

    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [Credential("openrouter", "live-key", "https://proxy.test/v1", "m", "test")],
    )
    assert endpoint_for("openrouter") == ("https://proxy.test/v1", "live-key")


def test_an_unknown_backend_is_a_lookup_error(monkeypatch):
    from lai.models import endpoint_for

    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    with pytest.raises(LookupError):
        endpoint_for("not-a-vendor")
