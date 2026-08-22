"""Giving the work back at the end.

The agent works on its own X server, which is the point — but an X window
belongs to the display its client connected to, and no amount of wishing moves
one across. So the run cannot hand you the window it was using.

It can hand you what was *in* it. A browser knows the page it is on; a run
knows the files it wrote. Reopening those on your desktop is the part that
actually mattered: you wanted the finished page and the finished file, not that
particular instance of Firefox.

What is deliberately not attempted: reconstructing unsaved state. If the agent
left something typed and unsaved in an editor on its screen, that is gone when
the server stops, and pretending otherwise would be worse than saying so.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

BROWSERS = frozenset({
    "firefox", "firefox-esr", "librewolf", "waterfox", "navigator",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "brave-browser", "microsoft-edge", "vivaldi", "opera",
})

URL_LIKE = re.compile(r"^(?:https?://|[\w-]+(?:\.[\w-]+)+(?:[/:?#]|$))", re.IGNORECASE)
"""A location, not a search. Matches `example.com/x` as well as a full URL."""

NOT_WORTH_REOPENING = ("about:", "chrome://", "moz-extension:", "file:///tmp/.")


@dataclass(frozen=True, slots=True)
class Handoff:
    """One thing worth reopening on the human's desktop."""

    kind: str
    """url | file"""
    target: str
    source: str = ""
    """Which application it came from, for the sentence describing it."""

    def describe(self) -> str:
        where = f" (from {self.source})" if self.source else ""
        return f"{self.target}{where}"


def collect(desktop, *, artifacts=()) -> list[Handoff]:
    """What the run left behind that can be opened again somewhere else."""
    found: list[Handoff] = []
    seen: set[str] = set()

    def add(kind: str, target: str, source: str = "") -> None:
        target = target.strip()
        if not target or target in seen:
            return
        seen.add(target)
        found.append(Handoff(kind, target, source))

    for window in _windows(desktop):
        wm_class = (window.wm_class or "").lower()
        if wm_class not in BROWSERS:
            continue
        for url in _urls_in(desktop, window):
            add("url", url, wm_class)

    for path in artifacts:
        text = str(path).strip()
        if text and Path(text).exists():
            add("file", str(Path(text).resolve()))

    return found


def deliver(handoffs: list[Handoff], *, display: str = "") -> tuple[list[Handoff], str]:
    """Open each on the given display. Returns (what opened, what went wrong)."""
    if not handoffs:
        return [], ""
    opener = _opener()
    if opener is None:
        return [], "xdg-open is not installed, so nothing could be reopened"

    environment = dict(os.environ)
    if display:
        environment["DISPLAY"] = display
    opened: list[Handoff] = []
    failures: list[str] = []
    for handoff in handoffs:
        try:
            subprocess.Popen(  # noqa: S603
                [opener, handoff.target],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, start_new_session=True, env=environment,
            )
            opened.append(handoff)
        except Exception as exc:
            failures.append(f"{handoff.target}: {exc}")
    return opened, "; ".join(failures)


def _opener() -> str | None:
    import shutil  # noqa: PLC0415

    return shutil.which("xdg-open")


def _windows(desktop) -> list:
    try:
        return desktop.windows.list_windows()
    except Exception:
        return []


def _urls_in(desktop, window) -> list[str]:
    """The page a browser window is showing, read out of its address bar.

    Matched on the *value* rather than the label, because the label is
    translated and the value is a URL in every language.
    """
    try:
        desktop.windows.focus(window.id)
        snapshot = desktop.snapshot()
    except Exception:
        return []

    out = []
    for element in snapshot.elements:
        if element.role not in ("entry", "text", "combo box"):
            continue
        value = (element.value or "").strip()
        if not value or not URL_LIKE.match(value):
            continue
        if value.lower().startswith(NOT_WORTH_REOPENING):
            continue
        out.append(value if "://" in value else f"https://{value}")
    return out[:4]
