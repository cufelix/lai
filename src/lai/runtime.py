"""Runtime assembly — wires config, OS layer, safety, tools, skills and provider.

One place builds the whole object graph so the CLI, the daemon and the MCP
server all get identical behaviour. Optional pieces (MCP servers, a model
provider) degrade to absent rather than failing the whole runtime: `lai doctor`
must still work on a machine with no API key.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .agent.loop import Agent
from .agent.providers.base import Provider
from .agent.providers.registry import build_provider
from .agent.session import Session
from .config import Config, load_config
from .errors import BackendUnavailable, LaiError, ProviderError
from .osl.desktop import Desktop
from .safety.audit import AuditLog
from .safety.policy import PolicyEngine
from .skills.registry import SkillRegistry
from .skills.tools import register as register_skill_tools
from .tools import build_registry
from .tools.base import ToolRegistry
from .tools.control import register as register_control_tools
from .tools.discovery import register as register_discovery_tools


@dataclass(slots=True)
class Runtime:
    """Everything an agent needs, assembled and ready."""

    config: Config
    desktop: Desktop
    policy: PolicyEngine
    audit: AuditLog
    registry: ToolRegistry
    skills: SkillRegistry
    provider: Provider | None = None
    provider_error: str = ""
    mcp_pool: object | None = None
    mcp_tools: list[str] = field(default_factory=list)
    mcp_errors: dict = field(default_factory=dict)
    cwd: Path = field(default_factory=Path.cwd)
    memory: object | None = None
    journal: object | None = None
    desktop_lock: object | None = None
    virtual_display: object | None = None
    """A second X server this runtime started, and must therefore shut down."""
    display_note: str = ""
    """Why the agent ended up on the screen it is on, in one sentence."""
    scheduler: object | None = None
    task_store: object | None = None
    extra: dict = field(default_factory=dict)

    def agent(
        self,
        *,
        session: Session | None = None,
        approver: Callable | None = None,
        on_event: Callable | None = None,
        system_extra: str = "",
    ) -> Agent:
        if self.provider is None:
            raise ProviderError(
                "no model provider available",
                detail=self.provider_error or "run `lai setup` to add one",
            )
        session = session or Session()
        if self.config.sessions_dir:
            session.bind(self.config.sessions_dir)
        agent = Agent(
            config=self.config,
            provider=self.provider,
            registry=self.registry,
            desktop=self.desktop,
            policy=self.policy,
            audit=self.audit,
            skills=self.skills,
            session=session,
            approver=approver,
            on_event=on_event,
            cwd=self.cwd,
            system_extra=system_extra,
            journal=self.journal,
            desktop_lock=self.desktop_lock,
            memory=self.memory,
            on_own_screen=self.virtual_display is not None,
            screen_note=self.display_note,
        )
        # Shared, long-lived services the tools reach through ToolContext.extra.
        # `agent` is set last so `delegate` can spawn a child of this very run.
        agent.tool_extra = {
            **self.extra,
            "memory_store": self.memory,
            "task_store": self.task_store,
            "scheduler": self.scheduler,
            "agent": agent,
        }
        return agent

    def hand_over(self, artifacts=()) -> tuple[list, str]:
        """Reopen what the agent left, on the human's desktop.

        Only meaningful when the agent had a screen of its own — otherwise the
        windows are already in front of the person, and reopening them would be
        a second copy of something they can see.
        """
        screen = self.virtual_display
        if screen is None or not getattr(self.config.desktop, "handover", True):
            return [], ""

        from .osl.handover import collect, deliver  # noqa: PLC0415

        try:
            found = collect(self.desktop, artifacts=artifacts)
        except Exception as exc:
            return [], f"could not read the agent's screen: {exc}"

        # A browser left open stays open, so every later task would find the
        # same page and reopen it — a tab a run in this session already handed
        # over is not new work.
        already = self.extra.setdefault("handed_over", set())
        fresh = [handoff for handoff in found if handoff.target not in already]
        opened, problem = deliver(fresh, display=getattr(screen, "host_display", ""))
        already.update(handoff.target for handoff in opened)
        return opened, problem

    def close(self) -> None:
        for closer in (
            getattr(self.scheduler, "stop", None),
            getattr(self.virtual_display, "stop", None),
            getattr(self.memory, "close", None),
            getattr(self.mcp_pool, "close_all", None),
            getattr(self.provider, "close", None),
            self.desktop.close,
        ):
            if closer is None:
                continue
            try:
                closer()
            except Exception:
                continue


def build_runtime(
    config: Config | None = None,
    *,
    cwd: Path | None = None,
    virtual: bool | None = None,
    with_provider: bool = True,
    with_mcp: bool = True,
    groups: set[str] | None = None,
    session_id: str = "",
    audit_echo: Callable | None = None,
) -> Runtime:
    """Assemble a Runtime. Never raises for a missing optional dependency."""
    config = config or load_config(cwd=cwd)
    config.ensure_dirs()
    work_dir = Path(cwd or Path.cwd())

    # A display of the agent's own: its own pointer, its own focus, its own
    # window stack. The human keeps typing in theirs. This is the default,
    # because sharing one desktop means clicking into whatever window its owner
    # just switched to — and no amount of taking turns makes that pleasant.
    screen, display_note = _own_screen(config, virtual)
    display = screen.display if screen is not None else (config.desktop.display or None)

    desktop = Desktop(
        max_edge=config.desktop.max_edge,
        a11y_timeout_ms=config.desktop.a11y_timeout_ms,
        display=display,
        # On its own screen a browser must be given its own profile, or it
        # simply hands the request to the copy already running on yours and
        # exits — which reads as "the application would not start".
        # Keyed by display: a browser left running on a screen that has since
        # gone takes the singleton lock on a shared profile with it, and every
        # later launch is quietly handed to a window nobody can see.
        browser_profile=(
            Path(config.home) / "browser" / screen.display.lstrip(":")
            if screen is not None else None
        ),
    )

    policy = PolicyEngine(
        config.safety,
        focus_provider=lambda: _safe_active_window(desktop),
    )
    audit = AuditLog.for_session(
        config.logs_dir,
        session_id or "cli",
        redact=config.safety.redact_secrets,
        echo=audit_echo,
    )

    registry = build_registry(policy=policy, groups=groups)
    register_control_tools(registry)
    register_skill_tools(registry)
    register_discovery_tools(registry)

    skills = SkillRegistry(config.resolved_skill_paths(work_dir))

    provider: Provider | None = None
    provider_error = ""
    if with_provider:
        try:
            provider = build_provider(config.provider, home=config.home)
        except LaiError as exc:
            provider_error = str(exc)

    memory = _open_memory(config)
    journal = _open_journal(config)
    desktop_lock = _open_desktop_lock(config, display=desktop.display)
    task_store = _open_task_store(config)

    mcp_pool = None
    mcp_tools: list[str] = []
    mcp_errors: dict = {}
    if with_mcp:
        mcp_pool, mcp_tools, mcp_errors = _attach_mcp(config, registry, work_dir)

    return Runtime(
        config=config,
        desktop=desktop,
        policy=policy,
        audit=audit,
        registry=registry,
        skills=skills,
        provider=provider,
        provider_error=provider_error,
        mcp_pool=mcp_pool,
        mcp_tools=mcp_tools,
        mcp_errors=mcp_errors,
        cwd=work_dir,
        memory=memory,
        journal=journal,
        desktop_lock=desktop_lock,
        virtual_display=screen,
        display_note=display_note,
        task_store=task_store,
    )


def _attach_mcp(config: Config, registry: ToolRegistry, cwd: Path):
    """Connect configured MCP servers and register their tools.

    Entirely optional: a broken or missing MCP server must never stop LAI from
    controlling the desktop.
    """
    try:
        from .mcp.client import MCPPool, load_mcp_configs, register_mcp_tools  # noqa: PLC0415
    except Exception:
        return None, [], {}

    try:
        servers = [s for s in load_mcp_configs(config, cwd) if getattr(s, "enabled", True)]
    except Exception as exc:
        return None, [], {"_load": str(exc)}
    if not servers:
        return None, [], {}

    try:
        pool = MCPPool(servers)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pool.connect_all()
        names = register_mcp_tools(registry, pool)
        return pool, list(names), dict(getattr(pool, "errors", {}) or {})
    except Exception as exc:
        return None, [], {"_connect": str(exc)}


def _open_desktop_lock(config: Config, *, display: str = ""):
    """The cross-process claim on *this* desktop.

    Keyed by display, because that is what is actually being contended: an
    agent working on its own virtual screen takes nothing away from the person
    using the real one, and making it queue behind them would defeat the point.
    """
    try:
        from .osl.lock import DesktopLock  # noqa: PLC0415

        suffix = (display or "").replace(":", "").replace(".", "-")
        name = f"desktop{'-' + suffix if suffix not in ('', '0') else ''}.lock"
        return DesktopLock(Path(config.home) / name)
    except Exception:
        return None


def _own_screen(config: Config, wanted: bool | None):
    """(the agent's own X server or None, one sentence saying why).

    ``wanted`` is the caller's override — True forces a screen, False forbids
    one, None leaves it to ``desktop.own_display``.
    """
    mode = config.desktop.own_display
    if wanted is False or (wanted is None and mode == "never"):
        return None, "sharing your desktop — it will wait while you are using the mouse"

    from .osl.virtual import VirtualDisplay  # noqa: PLC0415

    screen = VirtualDisplay(
        size=(config.desktop.virtual_width, config.desktop.virtual_height),
        # Watching means a nested server: Xephyr draws the agent's whole screen
        # into a window on yours, so you can see it working without it being
        # able to touch anything of yours.
        server="Xephyr" if config.desktop.watch else "",
    )
    try:
        screen.start()
    except Exception as exc:
        insist = wanted is True or mode == "always"
        if insist:
            raise BackendUnavailable(
                "cannot give the agent a screen of its own",
                detail=f"{exc} — install Xvfb (`lai doctor --fix`), or set "
                       "desktop.own_display = \"auto\" to fall back to your desktop",
            ) from exc
        return None, (
            f"no screen of its own ({exc}) — working on your desktop instead, "
            "and waiting while you use the mouse"
        )
    where = "in a window on your desktop" if config.desktop.watch else "off-screen"
    return screen, (
        f"working on its own screen {screen.display}, {where} — "
        "nothing it does reaches your windows"
    )


def _open_journal(config: Config):
    """The learned-notes journal. Unreadable notes must not stop a run."""
    try:
        from .knowledge import Journal  # noqa: PLC0415

        return Journal.open(config.home)
    except Exception:
        return None


def _open_memory(config: Config):
    """Long-term memory is optional: a broken database must not stop the agent."""
    try:
        from .agent.memory import MemoryStore  # noqa: PLC0415

        return MemoryStore(config.memory_file)
    except Exception:
        return None


def _open_task_store(config: Config):
    try:
        from .scheduler import TaskStore  # noqa: PLC0415

        return TaskStore(config.schedule_file)
    except Exception:
        return None


def _safe_active_window(desktop: Desktop):
    try:
        return desktop.windows.active_window()
    except Exception:
        return None
