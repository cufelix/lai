"""What this machine could think with — and how to switch between them.

`lai models` answers three questions that were previously only answerable by
reading source: what is ready right now, what is installed but needs a login,
and what LAI knows how to reach if you had a key. Keeping the answer in one
place also gives the setup wizard its menu.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, replace

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
    resting: str = ""
    """Why this backend refused recently, and roughly when it recovers.

    Credentials existing and a backend answering are different facts. A key
    that is out of quota until noon is not "ready now", and saying so is the
    difference between a listing and a guess.
    """

    @property
    def usable(self) -> bool:
        return self.status == READY and not self.resting

    def to_dict(self) -> dict:
        return {
            "resting": self.resting,
            "name": self.name, "label": self.label, "kind": self.kind,
            "status": self.status, "detail": self.detail, "model": self.model,
            "vision": self.vision, "hint": self.hint, "signup": self.signup,
        }


def discover(*, probe_local: bool = True, timeout: float = 0.6, home=None,
             deny: tuple = ()) -> list[Backend]:
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
    resting = _resting(home)
    denied = {name.strip().lower() for name in (deny or ()) if str(name).strip()}

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
            alive, detail = _probe(vendor.url(env), timeout)
            backends.append(Backend(
                name=vendor.name,
                label=vendor.label,
                kind=KIND_LOCAL,
                status=READY if alive else KNOWN,
                detail=detail or (vendor.notes or vendor.url(env)),
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

    if denied:
        # Saying "never use this" once has to hold in the listing too, or the
        # menu keeps offering what the user already ruled out.
        backends = [b for b in backends if b.name not in denied]
    if resting:
        backends = [
            replace(backend, resting=resting[backend.name])
            if backend.name in resting else backend
            for backend in backends
        ]

    order = {READY: 0, NEEDS_AUTH: 1, KNOWN: 2}
    # A backend that is refusing today sorts below one that is not, whatever
    # its credentials say — the point of the list is what to reach for now.
    backends.sort(key=lambda b: (order.get(b.status, 3), bool(b.resting), b.kind != KIND_API, b.name))
    return backends


def _resting(home) -> dict[str, str]:
    """Backends known to be refusing, and for how long."""
    if home is None:
        return {}
    try:
        from .agent.providers.health import cooling  # noqa: PLC0415

        return {name: entry.describe() for name, entry in cooling(home).items()}
    except Exception:
        return {}


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


def check(name: str, *, model: str = "", timeout: float = 120.0) -> tuple[bool, str]:
    """Actually talk to a backend. Returns (works, detail).

    A model id can be pinned, because "OpenRouter works" and "this model on
    OpenRouter works" are different claims — a vendor can be reachable while
    the model you picked is retired, gated or misspelled.
    """
    from .agent.providers.base import Message
    from .agent.providers.registry import build_provider
    from .config import ProviderConfig

    provider = None
    try:
        provider = build_provider(
            ProviderConfig(name=name, model=model, max_tokens=16, timeout=timeout)
        )
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


def endpoint_for(name: str) -> tuple[str, str]:
    """(base_url, api_key) for a backend, so its catalogue can be fetched.

    Uses whatever the machine already has — a discovered credential first,
    because that carries a key the environment actually holds, then the vendor
    table for the URL alone.
    """
    from .agent.providers.catalog import ALL_VENDORS, env_key_for
    from .agent.providers.registry import discover_credentials

    name = (name or "").strip().lower()
    for credential in discover_credentials():
        if credential.provider == name and credential.base_url:
            return credential.base_url, credential.api_key

    for vendor in ALL_VENDORS:
        if vendor.name == name:
            return vendor.url(), env_key_for(vendor, os.environ)

    raise LookupError(name)


def available_models(name: str, *, timeout: float = 8.0) -> list:
    """What a backend says it can serve, live."""
    from .agent.providers.listing import fetch

    base_url, api_key = endpoint_for(name)
    return fetch(base_url, api_key, timeout=timeout)
