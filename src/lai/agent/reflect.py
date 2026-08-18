"""Turning a finished run into something the next run can use.

Without this, an agent rediscovers the same desktop every time: which launcher
entry opens the editor, that a save dialog labels its field "Name:", that one
app publishes no accessibility tree and needs pixels. The run knows all of
that by the end and then throws it away.

So after a run ends, the model is shown a compressed trace of what it just did
and asked one question: what is worth writing down about *this machine*? The
answer is merged into the journal as markdown the human can read and correct.

Three rules keep it from becoming a liability:

* **Never fabricate.** Only what actually happened in the trace. A guessed
  coordinate written down as fact is worse than no note.
* **Never block.** Reflection happens after the result exists; a failure here
  is logged and dropped, never surfaced as a failed task.
* **Never grow without bound.** One model call, a capped trace, a handful of
  lessons, and duplicates merged rather than appended.
"""

from __future__ import annotations

import json
import re

MIN_STEPS = 2
"""Below this there is nothing to learn: one tool call is not a lesson."""

MAX_TRACE_CHARS = 6_000
MAX_LESSONS = 4

PROMPT = """\
You have just finished a task on a Linux desktop. Write down what is worth \
remembering about THIS MACHINE for next time.

Good notes are durable and specific to this desktop:
  - which application a vague name actually opens ("Text Editor" is Xed)
  - where a window's usable area really starts, if a toolbar took part of it
  - the exact label of a control that was hard to find
  - an approach that failed, and what worked instead

Bad notes are anything already obvious, anything about this one task's content, \
anything you did not actually observe, and anything that will be false tomorrow.

Reply with ONE JSON object and nothing else:

  {"notes": [{"topic": "short-kebab-topic", "title": "Human readable title",
              "tags": ["app", "editor"], "lesson": "one sentence, specific"}]}

Use an existing topic name when the lesson belongs with one. Return \
{"notes": []} if nothing durable was learned — that is a perfectly good answer \
and much better than inventing something.

## Existing topics
%(topics)s

## What just happened
Task: %(task)s
Outcome: %(status)s

%(trace)s
"""


def reflect(*, provider, journal, task: str, result, trace: str, audit=None) -> list:
    """Ask the model what it learned, and file it. Returns the notes written."""
    if result is None or int(getattr(result, "steps", 0)) < MIN_STEPS:
        return []

    topics = "\n".join(f"- {n.name}: {n.title}" for n in journal.list()[:20]) or "(none yet)"
    prompt = PROMPT % {
        "topics": topics,
        "task": task[:500],
        "status": getattr(result, "status", "unknown"),
        "trace": trace[-MAX_TRACE_CHARS:],
    }

    try:
        raw = _ask(provider, prompt)
        lessons = _parse(raw)
    except Exception as exc:
        if audit is not None:
            audit.write("reflect_failed", error=str(exc)[:200])
        return []

    written = []
    for lesson in lessons[:MAX_LESSONS]:
        topic = str(lesson.get("topic") or lesson.get("title") or "").strip()
        text = str(lesson.get("lesson") or "").strip()
        if not topic or not text:
            continue
        tags = tuple(str(t) for t in (lesson.get("tags") or []) if str(t).strip())
        try:
            note = journal.append(topic, text, title=str(lesson.get("title") or topic), tags=tags)
        except OSError:
            continue
        if note is not None:
            written.append(note)
    if written and audit is not None:
        audit.write("learned", notes=[n.name for n in written])
    return written


def _ask(provider, prompt: str) -> str:
    from .providers.base import Message  # noqa: PLC0415

    turn = provider.complete([Message.user(prompt)], system="", tools=[])
    return (getattr(turn, "text", "") or "").strip()


def _parse(raw: str) -> list[dict]:
    """Pull the notes list out of whatever the model actually printed."""
    for candidate in _candidates(raw):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("notes"), list):
            return [item for item in parsed["notes"] if isinstance(item, dict)]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _candidates(raw: str):
    text = (raw or "").strip()
    if not text:
        return
    yield text
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        yield match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        yield text[start:end + 1]


def build_trace(session, *, limit: int = 40) -> str:
    """A compressed account of the run: what was tried, and what came back."""
    lines: list[str] = []
    for message in getattr(session, "messages", [])[-limit:]:
        for block in getattr(message, "content", []):
            kind = getattr(block, "type", "")
            if kind == "tool_use":
                arguments = json.dumps(getattr(block, "input", {}), ensure_ascii=False)[:200]
                lines.append(f"→ {block.name} {arguments}")
            elif kind == "tool_result":
                mark = "✗" if getattr(block, "is_error", False) else "✓"
                content = str(getattr(block, "content", ""))[:200].replace("\n", " ")
                lines.append(f"  {mark} {content}")
            elif kind == "text" and getattr(block, "text", "").strip():
                lines.append(f"· {block.text.strip()[:200]}")
    return "\n".join(lines)
