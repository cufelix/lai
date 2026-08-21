"""Choosing and switching model backends at runtime.

Shared by every interface — the chat REPL, the TUI and the web UI all need the
same three things: what could answer, what is answering, and make that one
answer from now on. Keeping it here means a backend picked in the browser and a
backend picked in the terminal end up in exactly the same config file.
"""

from __future__ import annotations

from dataclasses import replace

from ..config import Config
from ..errors import ProviderError


def catalogue(runtime=None, *, probe: bool = True) -> list:
    """Every backend this machine could use, ready ones first.

    Given a runtime, backends known to be refusing right now are marked as
    such — a key that is out of quota until noon is not something to offer as
    a choice without saying so.
    """
    from ..models import discover  # noqa: PLC0415

    home = runtime.config.home if runtime is not None else None
    deny = runtime.config.provider.deny if runtime is not None else ()
    return discover(probe_local=probe, home=home, deny=deny)


def describe(runtime) -> dict:
    """What is answering right now, and what stands behind it."""
    provider = runtime.provider
    chain = list(getattr(provider, "chain", []) or ([provider.name] if provider else []))
    return {
        "name": provider.name if provider else "",
        "model": provider.model if provider else "",
        "chain": chain,
        "failures": dict(getattr(provider, "failures", {}) or {}),
        "error": runtime.provider_error,
        "configured": runtime.config.provider.name,
        "fallback": list(runtime.config.provider.fallback),
    }


def use(runtime, name: str, *, model: str = "", persist: bool = True) -> str:
    """Switch the live runtime to ``name``. Returns a one-line confirmation.

    The switch is verified by construction — an unusable backend raises here,
    while the runtime keeps the provider it already had. Nothing is written to
    disk until after that succeeds, so a typo can never lock the next start out
    of a working model.
    """
    from ..agent.providers.registry import build_provider  # noqa: PLC0415

    name = (name or "").strip()
    if not name:
        raise ProviderError("no backend named")

    config = replace(runtime.config.provider, name=name, model=model, api_key="", base_url="")
    provider = build_provider(config, home=runtime.config.home)

    previous = runtime.provider
    runtime.config = runtime.config.with_overrides(provider=config)
    runtime.provider = provider
    runtime.provider_error = ""
    if previous is not None and previous is not provider:
        try:
            previous.close()
        except Exception:
            pass

    if persist:
        save(runtime.config, {"provider": {"name": name, "model": model}})
    return f"{provider.name}/{provider.model}"


def set_fallback(runtime, chain: list[str], *, persist: bool = True) -> list[str]:
    """Replace the standby order. An empty list turns failover off."""
    cleaned = tuple(dict.fromkeys(c.strip() for c in chain if c.strip()))
    provider = replace(runtime.config.provider, fallback=cleaned)
    runtime.config = runtime.config.with_overrides(provider=provider)
    if persist:
        save(runtime.config, {"provider": {"fallback": list(cleaned)}})
    return list(cleaned)


def set_mode(runtime, mode: str, *, persist: bool = True) -> str:
    """Change the permission mode for this session (and, by default, for good)."""
    from ..config import PERMISSION_MODES  # noqa: PLC0415

    mode = (mode or "").strip().lower()
    if mode not in PERMISSION_MODES:
        raise ValueError(f"mode must be one of {', '.join(PERMISSION_MODES)}")
    safety = replace(runtime.config.safety, mode=mode)
    runtime.config = runtime.config.with_overrides(safety=safety)
    runtime.policy.config = safety
    if persist:
        save(runtime.config, {"safety": {"mode": mode}})
    return mode


def save(config: Config, updates: dict) -> None:
    """Merge ``updates`` into the on-disk config, keeping everything else."""
    from .. import config_file  # noqa: PLC0415

    existing = config_file.read(config.home)
    config_file.write(config.home, config_file.merge(existing, updates))


def set_key(runtime, name: str, key: str, *, model: str = "", persist: bool = True) -> str:
    """Save an API key for a vendor, after proving it works.

    Verified first, always: a key saved without being tried is a key you
    discover is wrong on your next real task, when you least want to.
    """
    from dataclasses import replace as _replace  # noqa: PLC0415

    from ..agent.providers.base import Message  # noqa: PLC0415
    from ..agent.providers.registry import build_provider  # noqa: PLC0415

    name, key = (name or "").strip(), (key or "").strip()
    if not name or not key:
        raise ProviderError("both a backend and a key are needed")

    candidate = _replace(
        runtime.config.provider, name=name, model=model, api_key=key,
        base_url="", max_tokens=16,
    )
    provider = build_provider(candidate)
    try:
        turn = provider.complete([Message.user("Say OK.")], system="Reply with one word.")
        answered = (turn.text or "").strip()[:40]
        resolved = provider.model
    finally:
        try:
            provider.close()
        except Exception:
            pass

    use(runtime, name, model=model or resolved, persist=False)
    runtime.config = runtime.config.with_overrides(
        provider=_replace(runtime.config.provider, api_key=key)
    )
    if persist:
        save(runtime.config, {"provider": {"name": name, "model": model or resolved, "api_key": key}})
    return f"{name}/{resolved}" + (f" replied {answered!r}" if answered else "")
