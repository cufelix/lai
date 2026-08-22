"""Environment checks that know how to repair themselves.

``lai doctor`` and ``lai setup`` are the same knowledge shown two ways: doctor
reports, setup fixes. Keeping the checks here — rather than inline in the CLI —
means a diagnosis can never drift from the repair for it, and every failure
carries the exact command that resolves it.

A check never raises. A probe that blows up is a failed check with the
exception as its detail, because "the accessibility bus is unreachable" is a
finding, not a crash.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Package names differ across distributions; these are the Debian/Ubuntu/Mint
# ones, which is what LAI targets today. `_install_command` adapts the manager.
APT_PACKAGES = {
    "xdotool": "xdotool",
    "a11y": "python3-gi gir1.2-atspi-2.0 gir1.2-gtk-3.0",
    "tesseract": "tesseract-ocr",
    "ffmpeg": "ffmpeg",
    "xvfb": "xvfb",
}


@dataclass(frozen=True, slots=True)
class Fix:
    """How to repair one failed check.

    Exactly one of ``command`` or ``apply`` does the work; ``manual`` covers the
    cases a program has no business automating (logging into a different
    session, obtaining an API key).
    """

    description: str
    command: tuple[str, ...] = ()
    apply: Callable[[], str] | None = None
    needs_sudo: bool = False
    manual: str = ""

    @property
    def automatic(self) -> bool:
        return bool(self.command) or self.apply is not None

    def run(self, *, timeout: float = 300.0) -> tuple[bool, str]:
        """Apply the fix. Returns (succeeded, output)."""
        if self.apply is not None:
            try:
                return True, self.apply() or "done"
            except Exception as exc:
                return False, f"{type(exc).__name__}: {exc}"
        if not self.command:
            return False, "nothing to run"
        try:
            result = subprocess.run(
                list(self.command), capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output

    def shell(self) -> str:
        """The command as a user would type it."""
        if not self.command:
            return ""
        parts = list(self.command)
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnosis, with the repair for it attached."""

    key: str
    label: str
    status: str
    detail: str
    fix: Fix | None = None
    required: bool = True
    """False for things LAI works without (OCR, recording, connectors)."""

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def blocking(self) -> bool:
        """Would this stop LAI from doing its job at all?"""
        return self.required and self.status == FAIL

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "status": self.status,
            "detail": self.detail, "required": self.required,
            "fix": self.fix.description if self.fix else "",
            "fix_command": self.fix.shell() if self.fix else "",
        }


@dataclass(slots=True)
class Report:
    checks: list[Check] = field(default_factory=list)

    def __iter__(self):
        return iter(self.checks)

    def __len__(self) -> int:
        return len(self.checks)

    @property
    def ready(self) -> bool:
        """Can LAI actually run a task right now?"""
        return not any(c.blocking for c in self.checks)

    @property
    def blockers(self) -> list[Check]:
        return [c for c in self.checks if c.blocking]

    @property
    def fixable(self) -> list[Check]:
        return [c for c in self.checks if c.status != OK and c.fix is not None and c.fix.automatic]

    def get(self, key: str) -> Check | None:
        return next((c for c in self.checks if c.key == key), None)

    def to_dict(self) -> dict:
        return {"ready": self.ready, "checks": [c.to_dict() for c in self.checks]}


# -- package management --------------------------------------------------


def package_manager() -> str:
    """Which installer this machine uses, or '' if none is recognised."""
    for name in ("apt-get", "dnf", "pacman", "zypper"):
        if shutil.which(name):
            return name
    return ""


def _install_command(packages: str) -> tuple[str, ...]:
    """Build a non-interactive install command for the local package manager."""
    names = packages.split()
    manager = package_manager()
    if manager == "apt-get":
        return ("sudo", "apt-get", "install", "-y", *names)
    if manager == "dnf":
        return ("sudo", "dnf", "install", "-y", *names)
    if manager == "pacman":
        return ("sudo", "pacman", "-S", "--noconfirm", *names)
    if manager == "zypper":
        return ("sudo", "zypper", "install", "-y", *names)
    return ()


def _install_fix(what: str, packages: str) -> Fix:
    command = _install_command(packages)
    if not command:
        return Fix(
            description=f"install {what}",
            manual=f"install these packages with your distribution's package manager: {packages}",
        )
    return Fix(description=f"install {what}", command=command, needs_sudo=True)


# -- individual probes ---------------------------------------------------


def check_platform() -> Check:
    system = platform.system()
    detail = f"{system} {platform.release()} / {_distro()}"
    if system != "Linux":
        return Check(
            "platform", "platform", FAIL, detail + " — LAI drives X11, which needs Linux",
            fix=Fix("run LAI on Linux", manual="LAI controls a Linux desktop; macOS and Windows are not supported."),
        )
    return Check("platform", "platform", OK, detail)


def check_display() -> Check:
    session = os.environ.get("XDG_SESSION_TYPE", "")
    display = os.environ.get("DISPLAY", "")
    detail = f"{session or 'unknown'} (DISPLAY={display or 'unset'})"

    if session == "x11" and display:
        return Check("display", "display server", OK, detail)
    if session == "wayland":
        return Check(
            "display", "display server", FAIL,
            detail + " — LAI needs X11",
            fix=Fix(
                "log in to an Xorg session",
                manual="Log out, then on the login screen pick the gear icon and choose "
                       "the Xorg / X11 session. Wayland support is not implemented yet.",
            ),
        )
    if not display:
        return Check(
            "display", "display server", FAIL, detail + " — no DISPLAY",
            fix=Fix(
                "run inside a graphical session",
                manual="LAI drives a real desktop, so it must run in your graphical session "
                       "(not over a plain SSH connection). Over SSH, use `ssh -X` or run it "
                       "in a terminal on the machine itself.",
            ),
        )
    return Check("display", "display server", WARN, detail + " — expected x11")


def check_xdotool() -> Check:
    if shutil.which("xdotool"):
        return Check("xdotool", "input (xdotool)", OK, "available")
    return Check(
        "xdotool", "input (xdotool)", FAIL, "missing — the agent cannot click or type",
        fix=_install_fix("xdotool", APT_PACKAGES["xdotool"]),
    )


def check_a11y(desktop=None) -> Check:
    """The accessibility tree is what makes LAI more than a pixel-guesser."""
    try:
        import gi  # noqa: F401
    except ImportError:
        return Check(
            "a11y", "accessibility (AT-SPI)", FAIL,
            "python3-gi is missing — semantic control unavailable",
            fix=_install_fix("the GTK/AT-SPI bindings", APT_PACKAGES["a11y"]),
        )

    if desktop is None:
        return Check("a11y", "accessibility (AT-SPI)", OK, "bindings installed")

    try:
        available = desktop.a11y.available
    except Exception as exc:
        return Check("a11y", "accessibility (AT-SPI)", FAIL, f"error: {exc}")

    if not available:
        return Check(
            "a11y", "accessibility (AT-SPI)", FAIL, "the accessibility bus is not reachable",
            fix=_toolkit_accessibility_fix(),
        )

    try:
        apps = [name for name, pid, _ in desktop.a11y.applications() if pid]
        elements = desktop.snapshot(max_elements=40)
    except Exception as exc:
        return Check("a11y", "accessibility (AT-SPI)", WARN, f"reachable, but reading failed: {exc}")

    detail = f"{len(apps)} app(s) registered, {len(elements)} element(s) in the focused window"
    if not elements:
        # The bus answers but nothing publishes a tree: the toolkit flag is off.
        return Check(
            "a11y", "accessibility (AT-SPI)", WARN,
            detail + " — applications are not publishing their interfaces",
            fix=_toolkit_accessibility_fix(),
        )
    return Check("a11y", "accessibility (AT-SPI)", OK, detail)


def _toolkit_accessibility_fix() -> Fix:
    return Fix(
        "turn on toolkit accessibility",
        command=("gsettings", "set", "org.gnome.desktop.interface", "toolkit-accessibility", "true"),
        manual="Some applications only publish their interface after a restart.",
    )


def check_screen(desktop=None) -> Check:
    if desktop is None:
        return Check("screen", "screen capture", WARN, "not probed")
    try:
        monitors = desktop.screen.monitors()
    except Exception as exc:
        return Check("screen", "screen capture", FAIL, str(exc))
    if not monitors:
        return Check("screen", "screen capture", FAIL, "no monitors detected")
    detail = ", ".join(f"{m.name} {m.bounds.width}x{m.bounds.height}" for m in monitors)
    return Check("screen", "screen capture", OK, detail)


def check_windows(desktop=None) -> Check:
    if desktop is None:
        return Check("windows", "window manager", WARN, "not probed")
    try:
        windows = desktop.windows.list_windows()
    except Exception as exc:
        return Check("windows", "window manager", FAIL, str(exc))
    return Check("windows", "window manager", OK, f"{len(windows)} window(s) visible")


def check_clipboard(desktop=None) -> Check:
    if desktop is None:
        return Check("clipboard", "clipboard", WARN, "not probed", required=False)
    try:
        available = desktop.clipboard.available
    except Exception as exc:
        return Check("clipboard", "clipboard", WARN, str(exc), required=False)
    return Check(
        "clipboard", "clipboard",
        OK if available else WARN,
        "available" if available else "unavailable — copy/paste tools will not work",
        required=False,
    )


def check_apps(desktop=None) -> Check:
    if desktop is None:
        return Check("apps", "applications", WARN, "not probed")
    try:
        apps = desktop.apps.apps()
    except Exception as exc:
        return Check("apps", "applications", FAIL, str(exc))
    return Check("apps", "applications", OK if apps else WARN, f"{len(apps)} installed .desktop entries")


def check_provider(runtime=None) -> Check:
    """A model backend is the one thing LAI cannot supply for itself."""
    from .agent.providers.registry import discover_credentials  # noqa: PLC0415

    try:
        credentials = discover_credentials()
    except Exception as exc:
        credentials = []
        detail = f"credential discovery failed: {exc}"
    else:
        detail = ""

    if runtime is not None and getattr(runtime, "provider", None) is not None:
        provider = runtime.provider
        extra = f" (+{len(credentials) - 1} more available)" if len(credentials) > 1 else ""
        return Check("provider", "model provider", OK, f"{provider.name} / {provider.model}{extra}")

    if credentials:
        return Check(
            "provider", "model provider", WARN,
            detail or f"{credentials[0].describe()} found but not usable yet",
        )

    return Check(
        "provider", "model provider", FAIL,
        detail or "no API key found — LAI needs a model to think with",
        fix=Fix(
            "add a model backend",
            manual="Run `lai setup` and paste an API key, or export one of:\n"
                   "  ANTHROPIC_API_KEY   (console.anthropic.com)\n"
                   "  ZAI_API_KEY         (z.ai — GLM)\n"
                   "  OPENAI_API_KEY      (platform.openai.com)\n"
                   "  OPENROUTER_API_KEY  (openrouter.ai)\n"
                   "Or run `ollama serve` for a local model with no key at all.",
        ),
    )


def check_ocr() -> Check:
    """OCR needs two halves, and reporting only one is how it fails at runtime.

    The binary and the Python binding are installed by different package
    managers, so a machine can easily have one without the other — and then
    `lai doctor` says OCR is fine right up until `ocr_read` raises.
    """
    binary = shutil.which("tesseract") is not None
    try:
        import pytesseract  # noqa: F401, PLC0415

        binding = True
    except ImportError:
        binding = False

    if binary and binding:
        return Check("ocr", "OCR (tesseract)", OK, "available", required=False)

    missing = []
    if not binary:
        missing.append("the tesseract binary")
    if not binding:
        missing.append("the pytesseract binding")
    return Check(
        "ocr", "OCR (tesseract)", WARN,
        f"missing {' and '.join(missing)} — apps without an accessibility tree are harder to read",
        fix=_ocr_fix(binary=binary, binding=binding),
        required=False,
    )


def _ocr_fix(*, binary: bool, binding: bool) -> Fix:
    """Install whichever half is absent — pip for the binding, the distro for the binary."""
    if binary and not binding:
        return Fix(
            description="install the pytesseract binding",
            command=(sys.executable, "-m", "pip", "install", "--quiet", "pytesseract"),
        )
    install = _install_fix("tesseract", APT_PACKAGES["tesseract"])
    if binding or not install.command:
        return install
    return Fix(
        description="install tesseract and its Python binding",
        command=install.command,
        needs_sudo=True,
        manual=f"then: {sys.executable} -m pip install pytesseract",
    )


def check_coders() -> Check:
    """Which coding agents are here for `code_agent` to delegate to.

    Optional by design: LAI writes files perfectly well itself. Having one is
    the difference between building software one `file_write` at a time and
    handing the job to a specialist, so it is worth naming either way.
    """
    try:
        from .tools.coding import PREFERENCE, available_coders  # noqa: PLC0415

        found = available_coders()
    except Exception as exc:
        return Check("coders", "coding agents", WARN, f"unavailable: {exc}", required=False)

    if found:
        return Check("coders", "coding agents", OK, ", ".join(found) + " — `code_agent` can delegate",
                     required=False)
    return Check(
        "coders", "coding agents", WARN,
        "none installed — LAI will write code itself, one file at a time",
        fix=Fix(
            description="install a coding agent",
            manual="any of: " + ", ".join(PREFERENCE)
            + ". They also work as model backends (`lai models`), so one install covers both.",
        ),
        required=False,
    )


def check_virtual_display() -> Check:
    """Whether the agent can be given a screen of its own.

    This is how the agent works by default, so its absence is not a missing
    nicety — it is the difference between an agent that works alongside you and
    one that clicks into whatever window you just switched to.
    """
    from .osl.virtual import available  # noqa: PLC0415

    found = available()
    if "Xvfb" in found:
        return Check("virtual", "own screen (Xvfb)", OK,
                     "the agent works off-screen; your desktop is its own", required=False)
    if found:
        return Check(
            "virtual", "own screen", WARN,
            f"only {', '.join(found)} — nested, so the agent's screen appears "
            "in a window on your desktop",
            fix=_install_fix("Xvfb", APT_PACKAGES["xvfb"]),
            required=False,
        )
    return Check(
        "virtual", "own screen", WARN,
        "not installed — so the agent has to share your mouse, keyboard and "
        "window stack, and will click into whatever you switch to",
        fix=_install_fix("Xvfb", APT_PACKAGES["xvfb"]),
        required=False,
    )


def check_recorder() -> Check:
    if shutil.which("ffmpeg"):
        return Check("recorder", "screen recording (ffmpeg)", OK, "available", required=False)
    return Check(
        "recorder", "screen recording (ffmpeg)", WARN, "missing — record_start is unavailable",
        fix=_install_fix("ffmpeg", APT_PACKAGES["ffmpeg"]),
        required=False,
    )


def check_config(config) -> Check:
    path = Path(getattr(config, "home", Path.home() / ".lai")) / "config.toml"
    if path.is_file():
        return Check("config", "configuration", OK, str(path), required=False)
    return Check(
        "config", "configuration", WARN, f"{path} does not exist yet — defaults are in use",
        fix=Fix("write a starter config", manual="Run `lai setup`."),
        required=False,
    )


# -- the whole picture ---------------------------------------------------


def run_checks(runtime=None, config=None, *, include_optional: bool = True) -> Report:
    """Probe everything. Never raises; a broken probe is a failed check."""
    desktop = getattr(runtime, "desktop", None)
    probes: list[Callable[[], Check]] = [
        check_platform,
        check_display,
        lambda: check_screen(desktop),
        check_xdotool,
        lambda: check_windows(desktop),
        lambda: check_a11y(desktop),
        lambda: check_clipboard(desktop),
        lambda: check_apps(desktop),
        lambda: check_provider(runtime),
    ]
    if include_optional:
        probes.extend([check_ocr, check_recorder, check_coders, check_virtual_display])
    if config is not None:
        probes.append(lambda: check_config(config))

    checks: list[Check] = []
    for probe in probes:
        try:
            checks.append(probe())
        except Exception as exc:  # a probe must never take the report down
            checks.append(Check("unknown", getattr(probe, "__name__", "check"), FAIL, str(exc)))
    return Report(checks)


def _distro() -> str:
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return "unknown"


__all__ = [
    "FAIL",
    "OK",
    "WARN",
    "Check",
    "Fix",
    "Report",
    "check_a11y",
    "check_apps",
    "check_clipboard",
    "check_config",
    "check_display",
    "check_ocr",
    "check_platform",
    "check_provider",
    "check_recorder",
    "check_screen",
    "check_windows",
    "check_xdotool",
    "package_manager",
    "run_checks",
]
