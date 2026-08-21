"""Configuration: file, environment and defaults.

Precedence (highest first): explicit kwargs → environment (``LAI_*``) →
``~/.lai/config.toml`` → built-in defaults. Config is immutable once built, so
nothing can mutate policy mid-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError

DEFAULT_HOME = Path(os.environ.get("LAI_HOME", Path.home() / ".lai"))
CONFIG_FILENAME = "config.toml"

# Where skills are discovered, in priority order. Claude Code's directories are
# included on purpose: skills written for Claude Code work here unchanged.
DEFAULT_SKILL_PATHS: tuple[str, ...] = (
    "{home}/skills",
    "{cwd}/.lai/skills",
    "{cwd}/.claude/skills",
    "~/.claude/skills",
    "~/.openclaw/skills",
)

PERMISSION_MODES = ("readonly", "ask", "auto", "yolo")


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Which model answers, and how we reach it."""

    name: str = "auto"
    """anthropic | zai | openai | claude_cli | ollama | auto"""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    max_tokens: int = 8192
    temperature: float = 1.0
    thinking_budget: int = 0
    timeout: float = 180.0
    prompt_cache: bool = True
    """Ask the backend to cache the unchanging prefix of each request.

    An agent loop re-sends the tools, the system prompt and the whole
    conversation on every turn. Caching that prefix is the single largest
    saving available — the far end charges a fraction for it and answers
    sooner — and costs nothing on backends that ignore the marker.
    """
    deny: tuple[str, ...] = ()
    """Backends never to use, whatever else says otherwise.

    A machine can have a working login for a model its owner does not want
    used — on cost, on policy, or simply on preference. Saying so once has to
    hold everywhere: auto-detection, the failover chain, and the menus. A
    denied backend is not offered and not fallen back to.
    """
    fallback: tuple[str, ...] = ("auto",)
    """Backends to try when this one refuses — quota, auth or an outage.

    ``("auto",)`` means every other backend this machine can use, best first;
    an explicit list pins the order; ``()`` disables failover entirely.
    """

    def redacted(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": ("set" if self.api_key else "unset"),
            "max_tokens": self.max_tokens,
            "fallback": list(self.fallback),
            "deny": list(self.deny),
        }


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    mode: str = "ask"
    """readonly (no side effects) | ask (confirm risky) | auto (allow known-safe) | yolo"""
    allow_tools: tuple[str, ...] = ()
    deny_tools: tuple[str, ...] = ()
    protected_apps: tuple[str, ...] = (
        "keepassxc", "bitwarden", "1password", "keepass", "seahorse", "gnome-keyring",
        "polkit", "gnome-disks", "gparted",
    )
    protected_titles: tuple[str, ...] = (
        "password", "authentication required", "unlock", "sudo", "polkit", "private browsing",
    )
    deny_shell_patterns: tuple[str, ...] = (
        r"\brm\s+-[a-z]*[rf]", r"\bmkfs\b", r"\bdd\s+if=", r":\(\)\s*\{.*\};:",
        r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b", r"\bhalt\b",
        r">\s*/dev/sd[a-z]", r"\bchmod\s+-R\s+777\s+/", r"\buserdel\b", r"\bpasswd\b",
        r"curl[^|]*\|\s*(ba)?sh", r"wget[^|]*\|\s*(ba)?sh", r"\bgit\s+push\s+--force",
        r"\bfdisk\b", r"\bparted\b", r"\bcryptsetup\b",
    )
    confirm_shell_patterns: tuple[str, ...] = (
        r"\bsudo\b", r"\bapt(-get)?\s+(install|remove|purge)", r"\bpip\s+install",
        r"\bnpm\s+(install|publish)", r"\bsystemctl\b", r"\bkill(all)?\b",
        r"\bgit\s+(push|reset\s+--hard)", r"\bdocker\b", r"\bsnap\s+(install|remove)",
    )
    max_actions_per_minute: int = 240
    dry_run: bool = False
    redact_secrets: bool = True
    yield_to_user: bool = True
    """Stand aside while the human is using the machine.

    A desktop agent shares one mouse with its owner. Two hands on it at once
    does not split the work — it produces clicks landing in whatever window the
    other one just switched to. So the agent waits, and the human never has to
    fight it for control.
    """
    user_idle_seconds: float = 4.0
    """How long the human must be still before the agent moves again."""
    max_yield_seconds: float = 300.0
    """Give up waiting after this and say so, rather than hanging forever."""

    def __post_init__(self) -> None:
        if self.mode not in PERMISSION_MODES:
            raise ConfigError(
                f"safety.mode must be one of {PERMISSION_MODES}, got {self.mode!r}"
            )


@dataclass(frozen=True, slots=True)
class DesktopConfig:
    max_edge: int = 1400
    a11y_timeout_ms: int = 800
    max_elements: int = 220
    settle_timeout: float = 3.0
    annotate_screenshots: bool = False
    display: str = ""
    virtual_width: int = 1920
    virtual_height: int = 1080
    """Size of the agent's own screen, when it is given one."""


@dataclass(frozen=True, slots=True)
class ChannelsConfig:
    """Remote connectors — Telegram, webhooks and friends.

    Off by default: a channel is a remote control for the desktop, so it has to
    be turned on deliberately.
    """

    enabled: tuple[str, ...] = ()
    open_access: bool = False
    """Accept anyone who messages the bot. Only ever for a throwaway sandbox."""
    telegram_token: str = ""
    discord_token: str = ""
    webhook_url: str = ""
    webhook_secret: str = ""
    webhook_style: str = "json"
    approval_timeout: float = 180.0

    def redacted(self) -> dict:
        return {
            "enabled": list(self.enabled),
            "open_access": self.open_access,
            "telegram_token": "set" if self.telegram_token else "unset",
            "discord_token": "set" if self.discord_token else "unset",
            "webhook_url": self.webhook_url,
            "webhook_secret": "set" if self.webhook_secret else "unset",
        }


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Whether the agent keeps notes on what it discovers here.

    On by default: an agent that rediscovers the same desktop every run is
    the single biggest avoidable cost in a long session. It costs one extra
    model call at the end of a run that actually did something.
    """

    enabled: bool = True
    reflect: bool = True
    """Write new notes after a run. Off means notes are still read, never written."""
    max_notes_in_prompt: int = 6


@dataclass(frozen=True, slots=True)
class WebConfig:
    """The browser view that runs alongside the chat.

    On by default: it costs one thread and a loopback port, and it shows the
    things a terminal cannot — the live screen, the notes, the settings — while
    you are talking to the agent in the same session.
    """

    autostart: bool = True
    open_browser: bool = False
    """Starting a browser window unasked is a step too far; the URL is printed."""
    host: str = "127.0.0.1"
    port: int = 8788
    """Deliberately not the daemon's 8787, so `lai` and `lai serve` coexist."""


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    max_steps: int = 60
    max_seconds: float = 1800.0
    max_tokens: int = 600_000
    max_tool_output_chars: int = 20_000
    max_consecutive_errors: int = 6


@dataclass(frozen=True, slots=True)
class Config:
    home: Path = DEFAULT_HOME
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    desktop: DesktopConfig = field(default_factory=DesktopConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    web: WebConfig = field(default_factory=WebConfig)
    skill_paths: tuple[str, ...] = DEFAULT_SKILL_PATHS
    mcp_config_paths: tuple[str, ...] = (
        "{home}/mcp.json", "{cwd}/.mcp.json", "~/.claude/mcp-configs/mcp-servers.json",
    )
    log_level: str = "info"

    # -- derived paths ---------------------------------------------------

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def artifacts_dir(self) -> Path:
        return self.home / "artifacts"

    @property
    def channels_file(self) -> Path:
        return self.home / "channels.json"

    @property
    def memory_file(self) -> Path:
        return self.home / "memory.db"

    @property
    def schedule_file(self) -> Path:
        return self.home / "schedule.json"

    @property
    def notes_dir(self) -> Path:
        return self.home / "notes"

    def ensure_dirs(self) -> None:
        for path in (self.home, self.sessions_dir, self.logs_dir, self.skills_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)

    def resolved_skill_paths(self, cwd: Path | None = None) -> list[Path]:
        return _resolve_paths(self.skill_paths, self.home, cwd)

    def resolved_mcp_paths(self, cwd: Path | None = None) -> list[Path]:
        return _resolve_paths(self.mcp_config_paths, self.home, cwd)

    def with_overrides(self, **kwargs: Any) -> Config:
        return replace(self, **kwargs)

    def redacted(self) -> dict:
        return {
            "home": str(self.home),
            "provider": self.provider.redacted(),
            "safety": {"mode": self.safety.mode, "dry_run": self.safety.dry_run},
            "channels": self.channels.redacted(),
            "desktop": {"max_edge": self.desktop.max_edge, "max_elements": self.desktop.max_elements},
            "limits": {"max_steps": self.limits.max_steps, "max_seconds": self.limits.max_seconds},
            "learning": {"enabled": self.learning.enabled, "reflect": self.learning.reflect},
        }


def _resolve_paths(patterns: tuple[str, ...], home: Path, cwd: Path | None) -> list[Path]:
    base = cwd or Path.cwd()
    out: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        raw = pattern.format(home=str(home), cwd=str(base))
        path = Path(raw).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _load_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import tomllib  # noqa: PLC0415 - py3.11+
    except ImportError:  # pragma: no cover - py3.10
        try:
            import tomli as tomllib  # type: ignore  # noqa: PLC0415
        except ImportError:
            raise ConfigError(
                "reading config.toml needs Python 3.11+ or the 'tomli' package"
            ) from None
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except Exception as exc:
        raise ConfigError(f"cannot parse {path}", detail=str(exc)) from exc


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"config section [{name}] must be a table")
    return value


def _fallback_chain(env, provider_data: dict) -> tuple[str, ...]:
    """Which backends stand in when the configured one refuses.

    Defaults to ``auto`` — the machine's other working backends — because a run
    dying at "usage limit reached" when a second key is sitting right there is
    a failure nobody chose. ``LAI_FALLBACK=off`` (or an empty list) turns it off.
    """
    raw = env.get("LAI_FALLBACK", provider_data.get("fallback", "auto"))
    if raw in (None, False):
        return ()
    chain = _tuple(raw, ("auto",))
    if len(chain) == 1 and chain[0].lower() in ("off", "none", "false", "no", ""):
        return ()
    return chain


def _tuple(value: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return fallback
    if isinstance(value, str):
        return tuple(v.strip() for v in value.split(",") if v.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise ConfigError(f"expected a list or comma-separated string, got {type(value).__name__}")


def load_config(
    path: Path | None = None, *, cwd: Path | None = None, overrides: dict | None = None
) -> Config:
    """Build the effective config from file + environment + overrides."""
    home = Path(os.environ.get("LAI_HOME", str(DEFAULT_HOME))).expanduser()
    config_path = path or (home / CONFIG_FILENAME)
    data = _load_toml(config_path)

    provider_data = _section(data, "provider")
    safety_data = _section(data, "safety")
    desktop_data = _section(data, "desktop")
    limits_data = _section(data, "limits")
    channels_data = _section(data, "channels")

    env = os.environ
    provider = ProviderConfig(
        name=env.get("LAI_PROVIDER", provider_data.get("name", "auto")),
        model=env.get("LAI_MODEL", provider_data.get("model", "")),
        base_url=env.get("LAI_BASE_URL", provider_data.get("base_url", "")),
        api_key=env.get("LAI_API_KEY", provider_data.get("api_key", "")),
        max_tokens=int(env.get("LAI_MAX_TOKENS", provider_data.get("max_tokens", 8192))),
        temperature=float(provider_data.get("temperature", 1.0)),
        thinking_budget=int(env.get("LAI_THINKING", provider_data.get("thinking_budget", 0))),
        timeout=float(provider_data.get("timeout", 180.0)),
        prompt_cache=_bool(env.get("LAI_PROMPT_CACHE"), provider_data.get("prompt_cache", True)),
        deny=_tuple(env.get("LAI_DENY") or provider_data.get("deny"), ()),
        fallback=_fallback_chain(env, provider_data),
    )

    safety = SafetyConfig(
        mode=env.get("LAI_MODE", safety_data.get("mode", "ask")),
        allow_tools=_tuple(safety_data.get("allow_tools"), ()),
        deny_tools=_tuple(safety_data.get("deny_tools"), ()),
        protected_apps=_tuple(
            safety_data.get("protected_apps"), _SAFETY_DEFAULTS.protected_apps
        ),
        protected_titles=_tuple(
            safety_data.get("protected_titles"), _SAFETY_DEFAULTS.protected_titles
        ),
        deny_shell_patterns=_tuple(
            safety_data.get("deny_shell_patterns"), _SAFETY_DEFAULTS.deny_shell_patterns
        ),
        confirm_shell_patterns=_tuple(
            safety_data.get("confirm_shell_patterns"), _SAFETY_DEFAULTS.confirm_shell_patterns
        ),
        max_actions_per_minute=int(safety_data.get("max_actions_per_minute", 240)),
        yield_to_user=_bool(env.get("LAI_YIELD"), safety_data.get("yield_to_user", True)),
        user_idle_seconds=float(safety_data.get("user_idle_seconds", 4.0)),
        max_yield_seconds=float(safety_data.get("max_yield_seconds", 300.0)),
        dry_run=_bool(env.get("LAI_DRY_RUN"), safety_data.get("dry_run", False)),
        redact_secrets=bool(safety_data.get("redact_secrets", True)),
    )

    desktop = DesktopConfig(
        max_edge=int(env.get("LAI_MAX_EDGE", desktop_data.get("max_edge", 1400))),
        a11y_timeout_ms=int(desktop_data.get("a11y_timeout_ms", 800)),
        max_elements=int(desktop_data.get("max_elements", 220)),
        settle_timeout=float(desktop_data.get("settle_timeout", 3.0)),
        annotate_screenshots=bool(desktop_data.get("annotate_screenshots", False)),
        display=env.get("LAI_DISPLAY", desktop_data.get("display", "")),
        virtual_width=int(desktop_data.get("virtual_width", 1920)),
        virtual_height=int(desktop_data.get("virtual_height", 1080)),
    )

    limits = LimitsConfig(
        max_steps=int(env.get("LAI_MAX_STEPS", limits_data.get("max_steps", 60))),
        max_seconds=float(env.get("LAI_MAX_SECONDS", limits_data.get("max_seconds", 1800.0))),
        max_tokens=int(limits_data.get("max_tokens", 600_000)),
        max_tool_output_chars=int(limits_data.get("max_tool_output_chars", 20_000)),
        max_consecutive_errors=int(limits_data.get("max_consecutive_errors", 6)),
    )

    learning_data = _section(data, "learning")
    learning = LearningConfig(
        enabled=_bool(env.get("LAI_LEARNING"), learning_data.get("enabled", True)),
        reflect=_bool(env.get("LAI_REFLECT"), learning_data.get("reflect", True)),
        max_notes_in_prompt=int(learning_data.get("max_notes_in_prompt", 6)),
    )

    web_data = _section(data, "web")
    web = WebConfig(
        autostart=_bool(env.get("LAI_WEB"), web_data.get("autostart", True)),
        open_browser=_bool(env.get("LAI_WEB_OPEN"), web_data.get("open_browser", False)),
        host=str(web_data.get("host", "127.0.0.1")),
        port=int(env.get("LAI_WEB_PORT", web_data.get("port", 8788))),
    )

    telegram_data = _section(channels_data, "telegram")
    discord_data = _section(channels_data, "discord")
    webhook_data = _section(channels_data, "webhook")
    channels = ChannelsConfig(
        enabled=_tuple(env.get("LAI_CHANNELS") or channels_data.get("enabled"), ()),
        open_access=_bool(env.get("LAI_CHANNELS_OPEN"), channels_data.get("open_access", False)),
        telegram_token=env.get("LAI_TELEGRAM_TOKEN", telegram_data.get("token", "")),
        discord_token=env.get("LAI_DISCORD_TOKEN", discord_data.get("token", "")),
        webhook_url=env.get("LAI_WEBHOOK_URL", webhook_data.get("url", "")),
        webhook_secret=env.get("LAI_WEBHOOK_SECRET", webhook_data.get("secret", "")),
        webhook_style=str(webhook_data.get("style", "json")),
        approval_timeout=float(channels_data.get("approval_timeout", 180.0)),
    )

    config = Config(
        home=home,
        provider=provider,
        safety=safety,
        desktop=desktop,
        limits=limits,
        channels=channels,
        learning=learning,
        web=web,
        skill_paths=_tuple(data.get("skill_paths"), DEFAULT_SKILL_PATHS),
        mcp_config_paths=_tuple(data.get("mcp_config_paths"), _CONFIG_DEFAULTS.mcp_config_paths),
        log_level=env.get("LAI_LOG_LEVEL", data.get("log_level", "info")),
    )
    if overrides:
        config = config.with_overrides(**overrides)
    return config


# With slots=True, class attributes are member descriptors rather than the
# default values, so defaults must be read from a real instance.
_SAFETY_DEFAULTS = SafetyConfig()
_CONFIG_DEFAULTS = Config()


def _bool(env_value: str | None, fallback: bool) -> bool:
    if env_value is None:
        return bool(fallback)
    return env_value.strip().lower() in ("1", "true", "yes", "on")
