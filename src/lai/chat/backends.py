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


def catalogue(*, probe: bool = True) -> list:
    """Every backend this machine could use, ready ones first."""
    from ..models import discover  # noqa: PLC0415

    return discover(probe_local=probe)


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
    provider = build_provider(config)

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
