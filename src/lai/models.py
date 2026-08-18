"""What this machine could think with — and how to switch between them.

`lai models` answers three questions that were previously only answerable by
reading source: what is ready right now, what is installed but needs a login,
and what LAI knows how to reach if you had a key. Keeping the answer in one
place also gives the setup wizard its menu.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import httpx

READY = "ready"        # usable right now
NEEDS_AUTH = "auth"    # present, but not signed in / no key
KNOWN = "known"        # LAI can reach it, given a key

KIND_API = "api"
KIND_LOCAL = "local"
KIND_CLI = "cli"


@dataclass(frozen=True, slots=True)
class Backend:
    """One way to get a model, and whether it works today."""

    name: str
    label: str
    kind: str
    status: str
    detail: str = ""
    model: str = ""
    signup: str = ""
    hint: str = ""
    vision: bool = True

    @property
    def usable(self) -> bool:
        return self.status == READY

    def to_dict(self) -> dict:
        return {
            "name": self.name, "label": self.label, "kind": self.kind,
            "status": self.status, "detail": self.detail, "model": self.model,
            "vision": self.vision, "hint": self.hint, "signup": self.signup,
        }


def discover(*, probe_local: bool = True, timeout: float = 0.6) -> list[Backend]:
    """Everything LAI could use, best first.

    Local endpoints are probed by opening a connection rather than assumed from
    a config file: "LM Studio is installed" and "LM Studio is serving" are
    different facts, and only the second one is useful.
    """
    from .agent.providers.catalog import ALL_VENDORS, LOCAL_VENDORS, env_key_for, source_of
    from .agent.providers.registry import discover_credentials

    env = os.environ
    backends: list[Backend] = []
    seen: set[str] = set()

    # 1. Whatever the runtime would actually pick, in its own order.
    try:
        for credential in discover_credentials():
            if credential.provider in seen:
                continue
            seen.add(credential.provider)
            kind = KIND_CLI if credential.provider.startswith("cli:") else (
                KIND_LOCAL if not credential.api_key else KIND_API
            )
            # Being installed is not the same as being signed in, and finding
            # out costs a real invocation — so say so rather than promise it.
            unverified = kind == KIND_CLI
            backends.append(Backend(
                name=credential.provider,
                label=credential.provider,
                kind=kind,
                status=READY,
                detail=credential.source + (" (sign-in not verified)" if unverified else ""),
                model=credential.model,
                hint=f"lai models test {credential.provider}" if unverified else "",
                vision=_cli_sees_images(credential.provider) if unverified else True,
            ))
    except Exception as exc:  # discovery must never take the listing down
        backends.append(Backend("?", "credential discovery failed", KIND_API, KNOWN, str(exc)))

    # 2. Agent CLIs that are installed but were not picked up as ready.
    try:
        from .agent.providers.cli_agent import CLI_SPECS, LOGIN_HINTS

        for spec in CLI_SPECS.values():
            name = f"cli:{spec.name}"
            if name in seen:
                continue
            seen.add(name)
            installed = bool(shutil.which(spec.command))
            backends.append(Backend(
                name=name,
                label=spec.describe or spec.name,
                kind=KIND_CLI,
                status=NEEDS_AUTH if installed else KNOWN,
                detail="installed" if installed else f"{spec.command} is not on PATH",
                model=spec.name,
                hint=LOGIN_HINTS.get(spec.name, ""),
                vision=spec.sees_images,
            ))
    except Exception:
        pass

    # 3. Local servers: ask the socket, do not guess.
    if probe_local:
        for vendor in LOCAL_VENDORS:
            if vendor.name in seen:
                continue
            seen.add(vendor.name)
            alive, detail = _probe(vendor.base_url, timeout)
            backends.append(Backend(
                name=vendor.name,
                label=vendor.label,
                kind=KIND_LOCAL,
                status=READY if alive else KNOWN,
                detail=detail or (vendor.notes or vendor.base_url),
                model=(env.get(vendor.model_env) or "").strip() or vendor.default_model,
                signup=vendor.signup,
                hint=vendor.notes,
                vision=vendor.vision,
            ))

    # 4. Everything else LAI knows how to reach.
    for vendor in ALL_VENDORS:
        if vendor.name in seen:
            continue
        seen.add(vendor.name)
        key = env_key_for(vendor, env)
        backends.append(Backend(
            name=vendor.name,
            label=vendor.label,
            kind=KIND_LOCAL if vendor.local else KIND_API,
            status=READY if key else KNOWN,
            detail=source_of(vendor, env) or f"set {' or '.join(vendor.env_keys)}",
            model=(env.get(vendor.model_env) or "").strip() or vendor.default_model,
            signup=vendor.signup,
            vision=vendor.vision,
        ))

    order = {READY: 0, NEEDS_AUTH: 1, KNOWN: 2}
    backends.sort(key=lambda b: (order.get(b.status, 3), b.kind != KIND_API, b.name))
    return backends


def _cli_sees_images(name: str) -> bool:
    """Whether this agent CLI can be handed a screenshot to read."""
    try:
        from .agent.providers.cli_agent import CLI_SPECS

        spec = CLI_SPECS.get(name.split(":", 1)[-1])
        return bool(spec and spec.sees_images)
    except Exception:
        return False


def _probe(base_url: str, timeout: float) -> tuple[bool, str]:
    """Is something actually answering on this endpoint?"""
    url = base_url.rstrip("/") + "/models"
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError:
        return False, "not running"
    if response.status_code >= 500:
        return False, f"HTTP {response.status_code}"
    try:
        payload = response.json()
        names = [str(entry.get("id", "")) for entry in payload.get("data", []) if isinstance(entry, dict)]
    except Exception:
        names = []
    if names:
        return True, f"serving {len(names)} model(s): " + ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
    return True, "running"


def check(name: str, *, timeout: float = 120.0) -> tuple[bool, str]:
    """Actually talk to a backend. Returns (works, detail)."""
    from .agent.providers.base import Message
    from .agent.providers.registry import build_provider
    from .config import ProviderConfig

    provider = None
    try:
        provider = build_provider(ProviderConfig(name=name, max_tokens=16, timeout=timeout))
        turn = provider.complete([Message.user("Say OK.")], system="Reply with one word.")
        text = (turn.text or "").strip()[:60]
        return True, f"{provider.name}/{provider.model}" + (f" replied {text!r}" if text else "")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        if provider is not None:
            try:
                provider.close()
            except Exception:
                pass


__all__ = [
    "KIND_API",
    "KIND_CLI",
    "KIND_LOCAL",
    "KNOWN",
    "NEEDS_AUTH",
    "READY",
    "Backend",
    "check",
    "discover",
]
