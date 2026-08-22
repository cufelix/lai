"""Application discovery and launch.

Indexes freedesktop ``.desktop`` entries (system, user and flatpak) so the agent
can "open GIMP" without knowing binaries or paths, then — the part that actually
matters — waits for the app's window to appear so the next action isn't fired at
a desktop that hasn't drawn yet.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..errors import AppNotFound, LaiError
from .windows import WindowInfo, WindowManager

XDG_DATA_DIRS = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))

# %f %F %u %U %i %c %k etc. — Exec field codes we must strip before running.
_FIELD_CODES = re.compile(r"%[fFuUdDnNickvm]")


@dataclass(frozen=True, slots=True)
class AppEntry:
    """One installed application."""

    id: str
    name: str
    exec_line: str
    path: str
    comment: str = ""
    categories: tuple[str, ...] = ()
    terminal: bool = False
    wm_class: str = ""
    no_display: bool = False

    @property
    def command(self) -> list[str]:
        cleaned = _FIELD_CODES.sub("", self.exec_line).strip()
        try:
            return shlex.split(cleaned)
        except ValueError:
            return cleaned.split()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "comment": self.comment,
            "exec": self.exec_line,
            "categories": list(self.categories),
            "wm_class": self.wm_class,
        }

    def score(self, query: str) -> float:
        """Fuzzy relevance for ``query``; 0 means no match."""
        q = query.lower().strip()
        if not q:
            return 0.0
        name, app_id, wm = self.name.lower(), self.id.lower(), self.wm_class.lower()
        if q == name or q == app_id or (wm and q == wm):
            return 100.0
        if name.startswith(q) or app_id.startswith(q):
            return 80.0
        if q in name:
            return 60.0
        if wm and q in wm:
            return 50.0
        if q in app_id:
            return 40.0
        if q in self.comment.lower():
            return 20.0
        # Sub-word match: "text editor" -> "Text Editor"
        words = set(re.split(r"\W+", name))
        if all(any(w.startswith(part) for w in words) for part in q.split() if part):
            return 30.0
        return 0.0


@dataclass(frozen=True, slots=True)
class LaunchResult:
    app: AppEntry | None
    command: list[str]
    pid: int | None
    window: WindowInfo | None
    waited: float
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "launched": " ".join(self.command),
            "app": self.app.name if self.app else None,
            "pid": self.pid,
            "window": self.window.to_dict() if self.window else None,
            "waited_seconds": round(self.waited, 2),
            "note": self.note,
        }


# Browsers are single-instance by design: start one while another is already
# running and it hands your request to that one and exits. On the agent's own
# display that is fatal — the process starts, the window appears on *your*
# desktop or nowhere at all, and the launcher reports "no window appeared".
# These are the flags that make a second, independent instance.
_CHROMIUM = (
    "--user-data-dir={profile}",
    "--no-first-run",
    "--no-default-browser-check",
    # Without this a Chromium window is a single opaque rectangle to AT-SPI —
    # not the page, not even the address bar. The agent is reduced to reading
    # pixels, and the handover at the end of a run cannot tell what page was
    # open. It costs some memory and buys the whole accessibility tree.
    "--force-renderer-accessibility",
)

BROWSER_ISOLATION: dict[str, tuple[str, ...]] = {
    "firefox": ("--no-remote", "--profile", "{profile}"),
    "firefox-esr": ("--no-remote", "--profile", "{profile}"),
    "librewolf": ("--no-remote", "--profile", "{profile}"),
    "waterfox": ("--no-remote", "--profile", "{profile}"),
    "google-chrome": _CHROMIUM,
    "google-chrome-stable": _CHROMIUM,
    "chromium": _CHROMIUM,
    "chromium-browser": _CHROMIUM,
    "brave-browser": _CHROMIUM,
    "microsoft-edge": _CHROMIUM,
    "vivaldi": _CHROMIUM,
    "opera": _CHROMIUM,
}


def isolate_browser(
    command: list[str], profile_root: Path, *, already_running: bool = False
) -> list[str]:
    """Command that starts its *own* browser, not a message to somebody else's.

    ``already_running`` matters for the Firefox family: ``--no-remote`` is what
    stops it handing the URL to the copy on your desktop, but it also refuses
    to start a *second* copy on this profile — so once one is up on this
    screen, the URL should go to that one. Chromium needs no such care: the
    same ``--user-data-dir`` reaches the same instance either way.

    Left alone if it is not a browser, or if the caller already said which
    profile to use — an explicit choice outranks this one.
    """
    if not command:
        return command
    binary = Path(command[0]).name.lower()
    flags = BROWSER_ISOLATION.get(binary)
    if flags is None:
        return command
    already = {"--profile", "-p", "--user-data-dir", "--no-remote"}
    if any(part.split("=")[0] in already for part in command[1:]):
        return command

    profile = profile_root / binary
    profile.mkdir(parents=True, exist_ok=True)
    if already_running:
        flags = tuple(flag for flag in flags if flag != "--no-remote")
    extra = [flag.format(profile=str(profile)) for flag in flags]
    # Flags go before the URL: Chrome ignores switches that follow a positional.
    return [command[0], *extra, *command[1:]]


class AppLauncher:
    """Finds and starts desktop applications."""

    def __init__(
        self,
        window_manager: WindowManager | None = None,
        *,
        browser_profile: Path | None = None,
    ) -> None:
        self._wm = window_manager or WindowManager()
        self.browser_profile = browser_profile
        """Where isolated browser profiles live, when browsers must be isolated."""
        self._cache: list[AppEntry] | None = None
        self._cache_at: float = 0.0
        self._cache_ttl: float = 60.0

    # -- discovery -------------------------------------------------------

    def search_dirs(self) -> list[Path]:
        dirs = [Path(XDG_DATA_HOME) / "applications"]
        dirs += [Path(d) / "applications" for d in XDG_DATA_DIRS.split(":") if d]
        dirs += [
            Path("/var/lib/flatpak/exports/share/applications"),
            Path(XDG_DATA_HOME) / "flatpak" / "exports" / "share" / "applications",
            Path("/var/lib/snapd/desktop/applications"),
        ]
        seen, out = set(), []
        for d in dirs:
            resolved = str(d)
            if resolved not in seen and d.is_dir():
                seen.add(resolved)
                out.append(d)
        return out

    def apps(self, *, refresh: bool = False, include_hidden: bool = False) -> list[AppEntry]:
        now = time.time()
        if not refresh and self._cache is not None and now - self._cache_at < self._cache_ttl:
            entries = self._cache
        else:
            found: dict[str, AppEntry] = {}
            for directory in self.search_dirs():
                for path in sorted(directory.glob("*.desktop")):
                    entry = _parse_desktop_file(path)
                    if entry and entry.id not in found:
                        found[entry.id] = entry
            entries = sorted(found.values(), key=lambda e: e.name.lower())
            self._cache = entries
            self._cache_at = now
        return entries if include_hidden else [e for e in entries if not e.no_display]

    def find(self, query: str, *, limit: int = 8) -> list[AppEntry]:
        scored = [(app.score(query), app) for app in self.apps()]
        ranked = sorted(((s, a) for s, a in scored if s > 0), key=lambda p: (-p[0], p[1].name))
        return [app for _, app in ranked[:limit]]

    def find_one(self, query: str) -> AppEntry:
        matches = self.find(query, limit=1)
        if not matches:
            raise AppNotFound(
                f"no installed application matches {query!r}",
                detail="use app_list to see what is installed",
            )
        return matches[0]

    # -- launching -------------------------------------------------------

    def _has_window_for(self, binary: str) -> bool:
        """Whether this program already has a window on this display."""
        name = Path(binary).name.lower()
        stem = name.split("-")[0]
        try:
            return any(
                stem in (window.wm_class or "").lower()
                for window in self._wm.list_windows()
            )
        except Exception:
            return False

    def launch(
        self,
        query: str,
        *,
        args: list[str] | None = None,
        wait_for_window: bool = True,
        timeout: float = 25.0,
        raise_window: bool = True,
    ) -> LaunchResult:
        """Start an app (or a raw command) and wait until its window exists."""
        started = time.monotonic()
        app: AppEntry | None = None
        try:
            app = self.find_one(query)
            command = app.command + (args or [])
        except AppNotFound:
            # Fall back to a bare executable name, e.g. launch("xterm").
            binary = shutil.which(query.split()[0]) if query.strip() else None
            if not binary:
                raise
            command = [binary, *query.split()[1:], *(args or [])]

        if not command:
            raise LaiError(f"application {query!r} has no runnable command")
        if self.browser_profile is not None:
            command = isolate_browser(
                command, self.browser_profile,
                already_running=self._has_window_for(command[0]),
            )

        before = {w.id for w in self._safe_window_ids()}
        env = {**os.environ, "GDK_BACKEND": os.environ.get("GDK_BACKEND", "x11")}
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        except FileNotFoundError as exc:
            raise AppNotFound(f"executable not found: {command[0]}", detail=str(exc)) from exc
        except PermissionError as exc:
            raise LaiError(f"not permitted to execute {command[0]}", detail=str(exc)) from exc

        window: WindowInfo | None = None
        note = ""
        if wait_for_window:
            window = self._await_window(app, proc.pid, before, timeout)
            if window is None:
                note = (
                    f"process started (pid {proc.pid}) but no window appeared within {timeout}s; "
                    "it may be a background service, or still loading"
                )
            elif raise_window:
                try:
                    window = self._wm.focus(window.id)
                except Exception:
                    pass

        return LaunchResult(
            app=app,
            command=command,
            pid=proc.pid,
            window=window,
            waited=time.monotonic() - started,
            note=note,
        )

    def _await_window(
        self, app: AppEntry | None, pid: int, before: set[int], timeout: float
    ) -> WindowInfo | None:
        """Match a new window to the app we just started.

        PID matching is the reliable signal, but many apps re-parent to a
        pre-existing process (single-instance browsers, gnome-terminal-server),
        so name/class matching is the fallback.
        """
        descendants = _descendant_pids(pid)
        expected = {app.wm_class.lower(), app.name.lower()} if app else set()
        expected.discard("")

        def is_ours(window: WindowInfo) -> bool:
            if window.id in before:
                return False
            if window.pid and (window.pid == pid or window.pid in descendants):
                return True
            if expected:
                haystack = f"{window.wm_class} {window.instance} {window.title}".lower()
                return any(token in haystack for token in expected)
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            descendants = _descendant_pids(pid)
            match = self._wm.wait_for(is_ours, timeout=1.0, poll=0.2, ignore_ids=before)
            if match is not None:
                return match
        return None

    def _safe_window_ids(self) -> list[WindowInfo]:
        try:
            return self._wm.list_windows(include_invisible=True)
        except Exception:
            return []

    # -- stopping --------------------------------------------------------

    def terminate(self, pid: int, *, force_after: float = 4.0) -> str:
        """SIGTERM, then SIGKILL if it refuses to die."""
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return "not_running"
        except PermissionError as exc:
            raise LaiError(f"not permitted to signal pid {pid}", detail=str(exc)) from exc

        deadline = time.monotonic() + force_after
        while time.monotonic() < deadline:
            time.sleep(0.15)
            if not _pid_alive(pid):
                return "terminated"
        try:
            os.kill(pid, signal.SIGKILL)
            return "killed"
        except ProcessLookupError:
            return "terminated"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _descendant_pids(pid: int, *, max_depth: int = 4) -> set[int]:
    """Direct and indirect children — apps commonly fork a real UI process."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return {pid}
    try:
        proc = psutil.Process(pid)
        return {pid} | {c.pid for c in proc.children(recursive=True)}
    except Exception:
        return {pid}


def _parse_desktop_file(path: Path) -> AppEntry | None:
    """Minimal, tolerant .desktop parser (configparser chokes on real-world files)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    in_entry = False
    data: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            if in_entry:
                break  # only the first [Desktop Entry] group matters
            in_entry = line == "[Desktop Entry]"
            continue
        if not in_entry or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if "[" in key:  # skip localized keys like Name[cs]
            continue
        data.setdefault(key, value.strip())

    if data.get("Type", "Application") != "Application":
        return None
    exec_line = data.get("Exec", "").strip()
    name = data.get("Name", "").strip() or path.stem
    if not exec_line:
        return None

    return AppEntry(
        id=path.stem,
        name=name,
        exec_line=exec_line,
        path=str(path),
        comment=data.get("Comment", "").strip(),
        categories=tuple(c for c in data.get("Categories", "").split(";") if c),
        terminal=data.get("Terminal", "false").lower() == "true",
        wm_class=data.get("StartupWMClass", "").strip(),
        no_display=(
            data.get("NoDisplay", "false").lower() == "true"
            or data.get("Hidden", "false").lower() == "true"
        ),
    )
