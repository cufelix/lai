"""Continuing where a conversation left off.

An agent you talk to over days is worth more than one that forgets between
invocations, and the alternative — re-explaining the machine, the task and
what already failed — is the most tedious thing about tools like this. So a
session can be picked up by id, or simply "the last one".

What is restored is the transcript: text, tool calls and their results.
Screenshots are not, because they are not stored — the desktop has moved on
anyway, and a stale image is worse than none.
"""

from __future__ import annotations

from pathlib import Path

from ..agent.session import Session


def resume(sessions_dir: str | Path, wanted: str = "") -> tuple[Session | None, str]:
    """Load a past session. Returns (session, human-readable explanation).

    ``wanted`` is a session id, a unique prefix of one, or empty for the most
    recent. A session that cannot be found is not an error worth stopping for:
    the caller gets None and a sentence saying why.
    """
    directory = Path(sessions_dir)
    listing = Session.list_sessions(directory, limit=200)
    if not listing:
        return None, "no past sessions to continue"

    wanted = (wanted or "").strip()
    if wanted == "last":
        wanted = ""
    if not wanted:
        chosen = listing[0]
    else:
        matches = [entry for entry in listing if entry["id"].startswith(wanted)]
        if not matches:
            return None, f"no session starting with {wanted!r} — `lai sessions` lists them"
        if len(matches) > 1:
            ids = ", ".join(entry["id"] for entry in matches[:5])
            return None, f"{wanted!r} matches several sessions: {ids}"
        chosen = matches[0]

    session = Session.load(Path(chosen["path"]))
    if not session.messages:
        return None, f"session {chosen['id']} has nothing in it"

    # Rebind so new turns append to the same file rather than starting a
    # second transcript for one conversation.
    session.bind(directory)
    return session, describe(session, chosen)


def describe(session: Session, entry: dict) -> str:
    task = (entry.get("task") or session.task or "").strip()
    headline = f"continuing session {session.id} — {len(session.messages)} message(s)"
    return f"{headline}\n  [dim]{task[:100]}[/dim]" if task else headline
