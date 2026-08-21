"""Asking a backend which models it actually serves.

A vendor's catalogue is not something to hard-code. OpenRouter alone carries
hundreds of models and adds more weekly; a local Ollama serves whatever its
owner pulled this morning. Both — and almost every other OpenAI-compatible
endpoint — answer ``GET /models`` with the live list, so the honest way to
offer a choice is to ask.

What comes back is normalised to (id, label, context, price) because that is
what a person actually chooses on, and sorted so the useful ones surface: a
free model before a paid one at the same size, a big context before a small
one. Anything the endpoint declines to say is simply absent rather than
guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

DEFAULT_TIMEOUT = 8.0
MAX_MODELS = 2_000


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One model a backend says it can serve."""

    id: str
    label: str = ""
    context: int = 0
    prompt_price: float = -1.0
    """Cost per million prompt tokens; -1 when the endpoint does not say."""
    free: bool = False

    def describe(self) -> str:
        parts = []
        if self.context:
            parts.append(f"{self.context // 1000}k ctx" if self.context >= 1000 else f"{self.context} ctx")
        if self.free:
            parts.append("free")
        elif self.prompt_price >= 0:
            parts.append(f"${self.prompt_price:g}/M in")
        return " · ".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "context": self.context,
            "prompt_price": self.prompt_price, "free": self.free,
        }


def fetch(base_url: str, api_key: str = "", *, timeout: float = DEFAULT_TIMEOUT) -> list[ModelInfo]:
    """Ask an OpenAI-compatible endpoint what it serves.

    Raises :class:`~lai.errors.ProviderError` with the reason, because "we could
    not list the models" and "there are no models" are different answers and
    only one of them is worth acting on.
    """
    from ...errors import ProviderError  # noqa: PLC0415

    url = base_url.rstrip("/") + "/models"
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ProviderError(f"could not reach {url}", detail=str(exc)) from exc
    if response.status_code == 401:
        raise ProviderError(f"{url} refused the key", detail="check the API key for this vendor")
    if response.status_code >= 400:
        raise ProviderError(f"{url} answered {response.status_code}", detail=response.text[:200])

    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError(f"{url} did not answer with JSON") from exc

    entries = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ProviderError(f"{url} answered in an unexpected shape")

    models = [parsed for entry in entries[:MAX_MODELS] if (parsed := _parse(entry)) is not None]
    models.sort(key=_rank)
    return models


def search(models: list[ModelInfo], query: str) -> list[ModelInfo]:
    """Models matching every word of the query, in id or label."""
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return models
    return [
        model for model in models
        if all(term in f"{model.id} {model.label}".lower() for term in terms)
    ]


def _parse(entry) -> ModelInfo | None:
    if isinstance(entry, str):
        return ModelInfo(id=entry.strip()) if entry.strip() else None
    if not isinstance(entry, dict):
        return None
    identifier = str(entry.get("id") or entry.get("name") or "").strip()
    if not identifier:
        return None

    context = entry.get("context_length") or entry.get("context_window") or 0
    top = entry.get("top_provider")
    if not context and isinstance(top, dict):
        context = top.get("context_length") or 0

    price, free = -1.0, False
    pricing = entry.get("pricing")
    if isinstance(pricing, dict):
        try:
            # OpenRouter quotes dollars per token; per million reads better.
            price = float(pricing.get("prompt", -1)) * 1_000_000
        except (TypeError, ValueError):
            price = -1.0
        free = price == 0
    free = free or identifier.endswith(":free")

    return ModelInfo(
        id=identifier,
        label=str(entry.get("name") or "").strip(),
        context=int(context or 0),
        prompt_price=price,
        free=free,
    )


def _rank(model: ModelInfo) -> tuple:
    """Free first, then cheap, then roomy — and stable by id after that."""
    price = model.prompt_price if model.prompt_price >= 0 else float("inf")
    return (0 if model.free else 1, price, -model.context, model.id)
