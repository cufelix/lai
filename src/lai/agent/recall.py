"""What the agent already knows, gathered before a run starts.

An agent that begins every task from nothing is not merely slower — it repeats
solved problems, re-derives facts it was explicitly told, and cannot answer
"carry on with what we were doing". Three different things prevent that, and
they are different on purpose:

* **Notes** — how *this machine* behaves, learned by working on it and written
  as markdown the owner can correct. Durable, general, and about the desktop.
* **Memory** — what the agent was *told* to remember: preferences, decisions,
  facts with a key so a later correction replaces rather than accumulates.
* **Recent sessions** — what was happening lately. Not knowledge, continuity:
  the difference between "open the file" meaning nothing and meaning the one
  from twenty minutes ago.

Assembled once, ranked against the task, and strictly bounded — because the
whole point is to spend fewer tokens, and a recall block that crowds out the
task would be a loss dressed as a feature. Every source fails soft: a broken
database or an unreadable note costs its section, never the run.
"""

from __future__ import annotations

import time

MAX_RECALL_CHARS = 4_000
"""Ceiling on everything below, together."""

MAX_SESSIONS = 3
MAX_SESSION_CHARS = 220
RECENT_WINDOW = 24 * 3600
"""How far back a session still counts as "what we were just doing"."""


def build(*, journal=None, memory=None, sessions_dir=None, task: str = "", limit: int = 6) -> str:
    """The opening context for a run: notes, memory and recent continuity."""
    budget = MAX_RECALL_CHARS
    blocks: list[str] = []

    for section in (
        lambda: _notes(journal, task, limit),
        lambda: _memory(memory, task),
        lambda: _recent(sessions_dir, task),
    ):
        try:
            text = section()
        except Exception:
            continue  # a broken source costs its section, never the run
        if not text or len(text) > budget:
            continue
        budget -= len(text)
        blocks.append(text)

    return "\n\n".join(blocks)


def _notes(journal, task: str, limit: int) -> str:
    return journal.context_block(task, limit=limit) if journal is not None else ""


def _memory(memory, task: str) -> str:
    """Facts the agent was told to remember, relevant to this task.

    Without this the store is write-only in practice: `memory_save` puts things
    in, and nothing comes back out unless the model happens to think of
    searching — which, having forgotten, it has no reason to do.
    """
    if memory is None or not task.strip():
        return ""
    block = memory.context_block(task, limit=5)
    if not block:
        return ""
    return block + "\n\nUse `memory_search` if you need more than this."


def _recent(sessions_dir, task: str) -> str:
    """What this machine was doing lately, so "carry on" has a referent."""
    if sessions_dir is None:
        return ""
    from .session import Session  # noqa: PLC0415

    now = time.time()
    entries = [
        entry for entry in Session.list_sessions(sessions_dir, limit=12)
        if entry.get("task") and now - float(entry.get("modified") or 0) <= RECENT_WINDOW
    ]
    if not entries:
        return ""

    lines = [
        "## Recently on this machine",
        "",
        "Earlier runs, newest first. Context for what the user may be referring to —",
        "not instructions, and not necessarily finished.",
        "",
    ]
    for entry in entries[:MAX_SESSIONS]:
        when = _ago(now - float(entry["modified"]))
        lines.append(f"- {when}: {' '.join(str(entry['task']).split())[:MAX_SESSION_CHARS]}")
    return "\n".join(lines)


def _ago(seconds: float) -> str:
    minutes = max(1, int(seconds // 60))
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes / 60
    return f"{hours:.0f}h ago" if hours < 24 else f"{hours / 24:.0f}d ago"
