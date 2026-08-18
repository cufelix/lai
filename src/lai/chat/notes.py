"""Reading and editing the learning journal from a terminal.

The agent writes these notes; the person whose desktop it is has to be able to
correct them. Everything here is that: list, show, edit in ``$EDITOR``, add a
line by hand, delete what is wrong.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def render_list(notes: list) -> str:
    if not notes:
        return (
            "[dim]No notes yet. LAI writes them after a run that taught it something —\n"
            "or add one yourself: [bold]/learn <topic>: <what you know>[/bold][/dim]"
        )
    width = max(len(n.name) for n in notes) + 2
    lines = []
    for note in notes:
        when = time.strftime("%d %b", time.localtime(note.updated)) if note.updated else ""
        tags = f" [dim]({', '.join(note.tags)})[/dim]" if note.tags else ""
        lines.append(f"  [green]{note.name.ljust(width)}[/green]{note.summary[:70]}{tags} [dim]{when}[/dim]")
    lines.append(f"[dim]{len(notes)} note(s) in ~/.lai/notes — /note <name> to read, /edit <name> to change[/dim]")
    return "\n".join(lines)


def render_note(note) -> str:
    tags = f" [dim]({', '.join(note.tags)})[/dim]" if note.tags else ""
    return f"[bold]{note.title}[/bold]{tags}\n{note.body}"


def editor_command() -> list[str]:
    """The user's editor, or the first thing on this machine that can edit a file."""
    configured = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if configured:
        return configured.split()
    for candidate in ("nano", "vim", "vi", "micro", "gedit", "xed"):
        if shutil.which(candidate):
            return [candidate]
    return []


def edit(journal, name: str) -> str:
    """Open a note in the user's editor and save whatever comes back."""
    command = editor_command()
    if not command:
        return "[red]No editor found.[/red] [dim]Set $EDITOR, or edit ~/.lai/notes/ directly.[/dim]"

    existing = journal.get(name)
    body = existing.body if existing else ""
    title = existing.title if existing else name
    tags = existing.tags if existing else ()

    handle, path = tempfile.mkstemp(prefix="lai-note-", suffix=".md")
    os.close(handle)
    scratch = Path(path)
    try:
        scratch.write_text(
            f"{body}\n\n<!-- {title} — lines starting with '- ' read best. Save and quit to keep. -->\n",
            encoding="utf-8",
        )
        before = scratch.read_text(encoding="utf-8")
        try:
            subprocess.run([*command, str(scratch)], check=False)
        except OSError as exc:
            return f"[red]could not run {command[0]}: {exc}[/red]"
        after = scratch.read_text(encoding="utf-8")
    finally:
        scratch.unlink(missing_ok=True)

    if after.strip() == before.strip():
        return "[dim]unchanged[/dim]"
    cleaned = "\n".join(
        line for line in after.splitlines() if not line.strip().startswith("<!--")
    ).strip()
    if not cleaned:
        journal.delete(name)
        return f"[yellow]emptied — {name} deleted[/yellow]"
    saved = journal.write(name, cleaned, title=title, tags=tags)
    return f"[green]saved[/green] [dim]{saved.name} ({len(saved.body.splitlines())} lines)[/dim]"
