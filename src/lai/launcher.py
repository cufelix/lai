"""Putting LAI in the applications menu.

Everything else here assumes a terminal. That is a reasonable assumption for
the person who installed it and a fatal one for everybody they might want to
hand it to: an agent you can only reach by typing a command is an agent most
people cannot reach at all.

A `.desktop` entry costs nothing and removes the whole barrier. LAI appears in
the menu with an icon, clicking it opens the browser interface, and the
terminal never comes into it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

APP_ID = "lai"
ENTRY_NAME = f"{APP_ID}.desktop"

ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <rect x="4" y="10" width="56" height="38" rx="5" fill="#1d4ed8"/>
  <rect x="10" y="16" width="44" height="26" rx="2" fill="#eff6ff"/>
  <circle cx="24" cy="27" r="3.2" fill="#1d4ed8"/>
  <circle cx="40" cy="27" r="3.2" fill="#1d4ed8"/>
  <path d="M24 34c4 3.2 12 3.2 16 0" stroke="#1d4ed8" stroke-width="2.6"
        stroke-linecap="round" fill="none"/>
  <rect x="22" y="50" width="20" height="4" rx="2" fill="#1d4ed8"/>
</svg>
"""

ENTRY = """[Desktop Entry]
Type=Application
Name=LAI
GenericName=Desktop assistant
Comment=Tell your computer what to do, in plain words
Exec={command}
Icon={icon}
Terminal=false
Categories=Utility;
Keywords=agent;assistant;automation;
StartupNotify=true
"""


def entry_path(home: Path | None = None) -> Path:
    base = Path(home) if home else Path(
        os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    )
    return base / "applications" / ENTRY_NAME


def icon_path(home: Path | None = None) -> Path:
    base = Path(home) if home else Path(
        os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    )
    return base / "icons" / "hicolor" / "scalable" / "apps" / f"{APP_ID}.svg"


def command() -> str:
    """How the menu entry should start LAI.

    An absolute path, because a desktop session's PATH is whatever the display
    manager decided at login and frequently does not include ~/.local/bin.
    """
    found = shutil.which("lai")
    return f"{found} open" if found else "lai open"


def install(*, home: Path | None = None) -> Path:
    """Write the menu entry and its icon. Returns the entry's path."""
    icon = icon_path(home)
    icon.parent.mkdir(parents=True, exist_ok=True)
    icon.write_text(ICON, encoding="utf-8")

    entry = entry_path(home)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(ENTRY.format(command=command(), icon=icon), encoding="utf-8")
    entry.chmod(0o755)
    _refresh(entry.parent)
    return entry


def uninstall(*, home: Path | None = None) -> bool:
    """Remove it again. True if there was something to remove."""
    removed = False
    for path in (entry_path(home), icon_path(home)):
        if path.exists():
            path.unlink()
            removed = True
    if removed:
        _refresh(entry_path(home).parent)
    return removed


def installed(*, home: Path | None = None) -> bool:
    return entry_path(home).is_file()


def _refresh(directory: Path) -> None:
    """Ask the desktop to notice. Harmless where the tool does not exist."""
    tool = shutil.which("update-desktop-database")
    if not tool:
        return
    try:
        subprocess.run(  # noqa: S603
            [tool, str(directory)], check=False, timeout=15,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
