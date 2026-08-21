"""Provider selection and credential discovery.

``auto`` picks the first backend that has usable credentials, in this order:

1. ``LAI_PROVIDER`` / config — an explicit choice always wins.
2. Anthropic (``ANTHROPIC_API_KEY``).
3. Z.ai / GLM (``ZAI_API_KEY``, ``GLM_API_KEY``, or an ``ANTHROPIC_AUTH_TOKEN``
   pointed at a non-Anthropic base URL).
4. OpenAI-compatible (``OPENAI_API_KEY``, ``OPENROUTER_API_KEY``).
5. Ollama, if the local daemon answers — so LAI still runs fully offline.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from ...config import ProviderConfig
from ...errors import ProviderError
from .anthropic_api import AnthropicProvider
from .base import Provider
from .openai_api import OpenAIProvider

ZAI_BASE_URL = "https://api.z.ai/api/anthropic"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "zai": "glm-4.6",
    "openai": "gpt-4o",
    "openrouter": "anthropic/claude-sonnet-4.5",
    "ollama": "qwen3-vl:2b",
}


@dataclass(frozen=True, slots=True)
class Credential:
    provider: str
    api_key: str
    base_url: str
    model: str
    source: str
    """Where the key came from — shown by `lai doctor`, never the key itself."""

    def describe(self) -> str:
        return f"{self.provider} ({self.model}) via {self.source}"


def discover_credentials() -> list[Credential]:
    """Find every backend this machine could use, best first."""
    env = os.environ
    found: list[Credential] = []

    key = env.get("ANTHROPIC_API_KEY", "").strip()
    base = env.get("ANTHROPIC_BASE_URL", "").strip()
    if key and (not base or "anthropic.com" in base):
        found.append(
            Credential("anthropic", key, base or "https://api.anthropic.com",
                       env.get("ANTHROPIC_MODEL") or DEFAULT_MODELS["anthropic"], "ANTHROPIC_API_KEY")
        )

    # An ANTHROPIC_* pair aimed elsewhere is a compatible gateway (z.ai, proxies).
    token = (env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY") or "").strip()
    if token and base and "anthropic.com" not in base:
        model = env.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or env.get("ANTHROPIC_MODEL") or DEFAULT_MODELS["zai"]
        found.append(Credential("zai", token, base, model, "ANTHROPIC_BASE_URL + AUTH_TOKEN"))

    for var in ("ZAI_API_KEY", "Z_AI_API_KEY", "GLM_API_KEY", "BIGMODEL_API_KEY"):
        value = env.get(var, "").strip()
        if value:
            found.append(Credential("zai", value, ZAI_BASE_URL, env.get("GLM_MODEL") or DEFAULT_MODELS["zai"], var))
            break

    if not any(c.provider == "zai" for c in found):
        wrapper = _sniff_wrapper_script()
        if wrapper:
            found.append(wrapper)

    value = env.get("OPENROUTER_API_KEY", "").strip()
    if value:
        found.append(
            Credential("openrouter", value, "https://openrouter.ai/api/v1",
                       env.get("OPENROUTER_MODEL") or DEFAULT_MODELS["openrouter"], "OPENROUTER_API_KEY")
        )

    value = env.get("OPENAI_API_KEY", "").strip()
    if value:
        found.append(
            Credential("openai", value, env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                       env.get("OPENAI_MODEL") or DEFAULT_MODELS["openai"], "OPENAI_API_KEY")
        )

    ollama_model = _ollama_model()
    if ollama_model:
        found.append(Credential("ollama", "", OLLAMA_BASE_URL, ollama_model, "local ollama daemon"))

    found.extend(_catalog_credentials(env, already={c.provider for c in found}))
    found.extend(_cli_credentials())
    return found


def _catalog_credentials(env, already: set[str]) -> list[Credential]:
    """Every catalogued vendor whose key is in the environment.

    The hand-written probes above come first because they encode preferences
    (an Anthropic-compatible gateway, a glm wrapper script) that a flat table
    cannot express. This fills in the long tail.
    """
    from .catalog import VENDORS, env_key_for, source_of  # noqa: PLC0415

    found: list[Credential] = []
    for vendor in VENDORS:
        if vendor.name in already:
            continue
        key = env_key_for(vendor, env)
        if not key:
            continue
        model = (env.get(vendor.model_env) or "").strip() if vendor.model_env else ""
        found.append(
            Credential(vendor.name, key, vendor.base_url, model or vendor.default_model,
                       source_of(vendor, env))
        )
    return found


def _cli_credentials() -> list[Credential]:
    """Agent CLIs installed on this machine, usable with no API key at all.

    Deliberately last: a real API is faster, streams, and can see screenshots.
    A CLI is the answer when somebody has a subscription and no key, which is
    common enough to be worth reaching for automatically.
    """
    try:
        from .cli_agent import available_clis  # noqa: PLC0415
    except Exception:
        return []
    return [
        Credential(f"cli:{spec.name}", "", "", spec.name, f"{spec.command} CLI on PATH")
        for spec in available_clis()
    ]


def _sniff_wrapper_script() -> Credential | None:
    """Read credentials out of a local ``glm``-style launcher script.

    Users commonly keep a one-line wrapper in ``~/.local/bin`` that exports
    ``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_AUTH_TOKEN``. Reusing it means LAI
    works out of the box on a machine already set up for GLM, with no key
    copied anywhere new.
    """
    for name in ("glm", "zai"):
        path = shutil.which(name)
        if not path:
            continue
        script = Path(path)
        try:
            if script.stat().st_size > 64_000:
                continue
            text = script.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "ANTHROPIC_AUTH_TOKEN" not in text:
            continue
        token = _extract(text, "ANTHROPIC_AUTH_TOKEN") or _extract(text, "ANTHROPIC_API_KEY")
        base = _extract(text, "ANTHROPIC_BASE_URL") or ZAI_BASE_URL
        model = _extract(text, "ANTHROPIC_DEFAULT_SONNET_MODEL") or DEFAULT_MODELS["zai"]
        if token and "anthropic.com" not in base:
            return Credential("zai", token, base, model, f"{name} wrapper script ({path})")
    return None


def _extract(text: str, variable: str) -> str:
    match = re.search(rf"^\s*(?:export\s+)?{variable}=([\"']?)([^\"'\s]+)\1", text, re.MULTILINE)
    return match.group(2) if match else ""


def _ollama_model() -> str:
    """Preferred local model — a vision model if one is pulled."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/models", timeout=1.5)
        if response.status_code != 200:
            return ""
        models = [m.get("id", "") for m in (response.json().get("data") or [])]
    except Exception:
        try:
            proc = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=3, check=False
            )
            models = [line.split()[0] for line in proc.stdout.splitlines()[1:] if line.strip()]
        except Exception:
            return ""
    if not models:
        return ""
    for model in models:
        if any(tag in model.lower() for tag in ("-vl", "vision", "llava", "moondream")):
            return model
    return models[0]


def build_provider(config: ProviderConfig, *, on_switch=None, home=None) -> Provider:
    """Instantiate the configured provider, with its fallback chain behind it.

    ``on_switch(from_name, to_name, reason)`` is called if a run has to move to
    a standby backend, so the interface can say so rather than silently
    answering as somebody else.
    """
    chain = build_chain(config, home=home)
    if len(chain) == 1:
        return chain[0].build()

    from .fallback import FallbackProvider  # noqa: PLC0415

    return FallbackProvider(chain, on_switch=on_switch, home=home)


def build_chain(config: ProviderConfig, *, home=None) -> list:
    """The ordered backends this configuration may use, best first.

    Every entry is lazy: building a provider can probe a socket or shell out to
    a CLI, and a standby that is never reached should cost nothing.
    """
    from dataclasses import replace  # noqa: PLC0415

    from .fallback import Candidate  # noqa: PLC0415

    denied = {n.strip().lower() for n in (config.deny or ()) if n.strip()}
    credentials = [c for c in discover_credentials() if c.provider not in denied]
    name = (config.name or "auto").lower()
    resting = _resting(home)

    if name in denied:
        raise ProviderError(
            f"{name} is on the deny list",
            detail="remove it from provider.deny, or configure a different backend",
        )

    if name == "auto":
        if not credentials:
            raise ProviderError(
                "no model backend available",
                detail=(
                    "set one of ANTHROPIC_API_KEY, ZAI_API_KEY/GLM_API_KEY, OPENAI_API_KEY, "
                    "OPENROUTER_API_KEY, or run a local ollama. See `lai doctor`."
                ),
            )
        # "auto" means "whichever works", so a backend known to be refusing is
        # not the one to start on. If every one of them is resting, start at
        # the top anyway — a stale cooldown must not leave the machine with no
        # agent at all.
        awake = [c for c in credentials if c.provider not in resting]
        name = (awake or credentials)[0].provider

    primary = next((c for c in credentials if c.provider == name), None)
    chain = [Candidate(name, lambda: _instantiate(name, config, primary))]

    wanted = tuple(config.fallback or ())
    if not wanted:
        return chain
    if len(wanted) == 1 and wanted[0].lower() == "auto":
        wanted = tuple(_by_capability(credentials))

    seen = {name}
    for candidate_name in wanted:
        standby = candidate_name.strip().lower()
        if not standby or standby in seen or standby == "auto":
            continue
        seen.add(standby)
        if standby in resting or standby in denied:
            continue
        credential = next((c for c in credentials if c.provider == standby), None)
        # A standby never inherits the primary's key, model or URL — a z.ai key
        # aimed at Anthropic is not a fallback, it is a confusing 401.
        neutral = replace(config, name=standby, model="", api_key="", base_url="")
        chain.append(
            Candidate(
                standby,
                lambda n=standby, cfg=neutral, cred=credential: _instantiate(n, cfg, cred),
            )
        )
    return chain


def _instantiate(name: str, config: ProviderConfig, credential: Credential | None) -> Provider:
    api_key = config.api_key or (credential.api_key if credential else "")
    base_url = config.base_url or (credential.base_url if credential else "")
    model = config.model or (credential.model if credential else "") or DEFAULT_MODELS.get(name, "")

    if name in ("anthropic", "zai"):
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            base_url=base_url or (ZAI_BASE_URL if name == "zai" else "https://api.anthropic.com"),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            thinking_budget=config.thinking_budget,
            timeout=config.timeout,
            name=name,
            prompt_cache=config.prompt_cache,
        )
    if name in ("openai", "openrouter", "ollama"):
        if name == "ollama":
            base_url = base_url or OLLAMA_BASE_URL
            api_key = api_key or "ollama"
        if not api_key:
            raise ProviderError(f"{name}: no API key configured")
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            name=name,
            supports_vision=name != "ollama" or bool(re.search(r"-vl|vision|llava", model, re.I)),
        )
    if name.startswith("cli:") or name in _cli_names():
        from .cli_agent import CLIAgentProvider  # noqa: PLC0415

        return CLIAgentProvider(
            name.split(":", 1)[-1],
            model=config.model,
            timeout=max(config.timeout, 300.0),
        )

    from .catalog import get as get_vendor  # noqa: PLC0415

    vendor = get_vendor(name)
    if vendor is not None:
        key = api_key or (os.environ.get(vendor.env_keys[0], "") if vendor.env_keys else "")
        if not key and not vendor.local:
            raise ProviderError(
                f"{name}: no API key configured",
                detail=f"set {' or '.join(vendor.env_keys)}"
                + (f" — get one at {vendor.signup}" if vendor.signup else ""),
            )
        return OpenAIProvider(
            api_key=key or "local",
            model=model or vendor.default_model,
            base_url=base_url or vendor.base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            name=name,
            supports_vision=vendor.vision,
        )

    raise ProviderError(
        f"unknown provider {name!r}",
        detail="run `lai models` to see everything this machine can use",
    )


def _resting(home) -> dict:
    """Backends known to be refusing right now, and until when.

    A backend that told us when its quota resets should not be asked again
    before then. An explicitly configured backend is exempt: the user asked for
    it, and being wrong about a recovery time must cost a retry, never an
    outage.
    """
    if home is None:
        return {}
    try:
        from .health import cooling  # noqa: PLC0415

        return cooling(home)
    except Exception:
        return {}


def _by_capability(credentials: list[Credential]) -> list[str]:
    """Order standbys by how well they can actually finish the job.

    Discovery order encodes "what should answer by default", which is not the
    same question. A 2B model running on the CPU is a fine last resort and a
    poor second choice: when a hosted key hits its quota, a signed-in coding
    CLI carries a frontier model and will finish the task, so it goes first.
    """
    hosted, cli, local = [], [], []
    for credential in credentials:
        if credential.provider.startswith("cli:"):
            cli.append(credential.provider)
        elif credential.provider in ("ollama",) or not credential.api_key:
            local.append(credential.provider)
        else:
            hosted.append(credential.provider)
    return hosted + cli + local


def _cli_names() -> set[str]:
    try:
        from .cli_agent import CLI_SPECS  # noqa: PLC0415

        return set(CLI_SPECS)
    except Exception:
        return set()
