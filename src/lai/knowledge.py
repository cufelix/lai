"""What the agent has learned about *this* machine, as editable markdown.

An agent that drives a desktop meets the same obstacles over and over: the save
dialog whose field is called "Name:", the app whose canvas starts below a
toolbar, the launcher entry whose name is nothing like the program. Rediscovering
those costs steps every single run, and the model has no way to keep them.

So it keeps a journal: one markdown file per topic in ``~/.lai/notes``, written
by the agent after a run and editable by the human at any time — in an editor,
in the chat, or in the browser. Markdown rather than a database because the
person whose desktop this is must be able to read what their agent believes,
correct it, and delete what is wrong. A note that cannot be audited is a note
that will quietly mislead every future run.

The frontmatter is the same shape skills use, so the two are learnable as one
idea: a fenced ``---`` block of ``key: value`` lines, then the body.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

NOTES_DIRNAME = "notes"
MAX_NOTE_CHARS = 8_000
"""A note is a summary, not a transcript; beyond this it stops being readable."""

MAX_CONTEXT_CHARS = 6_000
"""Ceiling on what is injected into a prompt, so the journal cannot crowd out the task."""

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_SLUG_SAFE = re.compile(r"[^a-z0-9._-]+")


def slugify(text: str) -> str:
    """A filename that is safe, stable and still recognisable."""
    slug = _SLUG_SAFE.sub("-", str(text).strip().lower()).strip("-.")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:64] or "note"


@dataclass(frozen=True, slots=True)
class Note:
    """One thing the agent learned, and where it lives."""

    name: str
    title: str
    body: str
    tags: tuple[str, ...] = ()
    updated: float = 0.0
    path: Path | None = None

    @property
    def summary(self) -> str:
        for line in self.body.splitlines():
            stripped = line.strip().lstrip("-*# ").strip()
            if stripped:
                return stripped[:120]
        return ""

    def render(self) -> str:
        """The file as it is written to disk."""
        stamp = time.strftime("%Y-%m-%d", time.localtime(self.updated or time.time()))
        lines = ["---", f"title: {self.title}"]
        if self.tags:
            lines.append(f"tags: {', '.join(self.tags)}")
        lines += [f"updated: {stamp}", "---", "", self.body.strip(), ""]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "title": self.title, "tags": list(self.tags),
            "updated": self.updated, "summary": self.summary, "body": self.body,
        }


def parse(text: str, *, name: str = "", path: Path | None = None) -> Note:
    """Read a note from markdown. Frontmatter is optional — a bare file still works."""
    metadata: dict[str, str] = {}
    body = text
    match = _FRONTMATTER.match(text)
    if match:
        body = text[match.end():]
        for line in match.group(1).splitlines():
            key, sep, value = line.partition(":")
            if sep:
                metadata[key.strip().lower()] = value.strip()

    tags = tuple(t.strip() for t in metadata.get("tags", "").replace(",", " ").split() if t.strip())
    updated = 0.0
    if path is not None:
        try:
            updated = path.stat().st_mtime
        except OSError:
            updated = 0.0
    return Note(
        name=name or (path.stem if path else "note"),
        title=metadata.get("title") or (name or (path.stem if path else "note")).replace("-", " "),
        body=body.strip(),
        tags=tags,
        updated=updated,
        path=path,
    )


def _age(note) -> str:
    """How old a lesson is, in words.

    Advice learned once, months ago, from a single unlucky run should not read
    like something established. This machine had a note saying the text editor
    could not be opened at all — untrue, written after one bad run, and it
    steered every run afterwards.
    """
    when = getattr(note, "updated", 0.0) or 0.0
    if not when:
        return "age unknown"
    days = max(0, int((time.time() - when) // 86400))
    if days == 0:
        return "learned today"
    if days == 1:
        return "learned yesterday"
    if days < 14:
        return f"learned {days} days ago"
    if days < 60:
        return f"learned {days // 7} weeks ago"
    return f"learned {days // 30} months ago — worth re-checking"


@dataclass(slots=True)
class Journal:
    """The notes directory, read and written.

    Every method tolerates a missing or unreadable directory: knowledge is an
    improvement, never a prerequisite, and a permissions problem in ``~/.lai``
    must not stop the agent from working.
    """

    directory: Path
    _cache: dict = field(default_factory=dict)

    @classmethod
    def open(cls, home: str | Path) -> Journal:
        return cls(Path(home) / NOTES_DIRNAME)

    # -- reading -----------------------------------------------------------

    def list(self) -> list[Note]:
        """Every note, most recently updated first."""
        notes: list[Note] = []
        try:
            paths = sorted(self.directory.glob("*.md"))
        except OSError:
            return []
        for path in paths:
            note = self._read(path)
            if note is not None:
                notes.append(note)
        return sorted(notes, key=lambda n: n.updated, reverse=True)

    def get(self, name: str) -> Note | None:
        path = self._path_for(name)
        return self._read(path) if path is not None and path.is_file() else None

    def search(self, query: str, *, limit: int = 6) -> list[Note]:
        """Notes worth showing for this task, best first.

        Deliberately a keyword score rather than embeddings: the corpus is a
        few dozen short notes about one machine, and a dependency-free match on
        title, tags and body is both good enough and explainable.
        """
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        if not terms:
            return self.list()[:limit]

        scored: list[tuple[int, Note]] = []
        for note in self.list():
            haystack = f"{note.title} {' '.join(note.tags)}".lower()
            body = note.body.lower()
            score = sum(4 for term in terms if term in haystack)
            score += sum(1 for term in terms if term in body)
            if score:
                scored.append((score, note))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].updated))
        return [note for _, note in scored[:limit]]

    def context_block(self, task: str = "", *, limit: int = 6) -> str:
        """The section injected into the system prompt."""
        notes = self.search(task, limit=limit) if task else self.list()[:limit]
        if not notes:
            return ""
        lines = [
            "## What you have learned on this machine",
            "",
            "Notes from your own earlier runs here. Trust them as a starting point,",
            "but verify — the desktop may have changed since, and a note written after",
            "one bad run can be wrong. If the screen contradicts one, correct the note",
            "yourself with `note_write` rather than working around it: a wrong note",
            "left standing will steer every run after this one.",
            "",
        ]
        budget = MAX_CONTEXT_CHARS
        for note in notes:
            chunk = f"### {note.title} [{_age(note)}]\n{note.body.strip()}\n"
            if len(chunk) > budget:
                break
            budget -= len(chunk)
            lines.append(chunk)
        return "\n".join(lines).strip()

    # -- writing -----------------------------------------------------------

    def write(self, name: str, body: str, *, title: str = "", tags=()) -> Note:
        """Create or replace a note. Returns what was stored."""
        slug = slugify(name)
        note = Note(
            name=slug,
            title=title or name.strip() or slug.replace("-", " "),
            body=body.strip()[:MAX_NOTE_CHARS],
            tags=tuple(tags),
            updated=time.time(),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{slug}.md"
        path.write_text(note.render(), encoding="utf-8")
        self._cache.pop(path, None)
        return self._read(path) or note

    def append(self, name: str, lesson: str, *, title: str = "", tags=()) -> Note:
        """Add a line to a note, creating it if needed, without repeating one.

        The agent re-learns the same thing constantly — the same dialog, the
        same layout — and a journal that accumulates ten copies of one fact is
        worse than no journal at all.
        """
        lesson = lesson.strip().lstrip("-").strip()
        if not lesson:
            existing = self.get(slugify(name))
            return existing or self.write(name, "", title=title, tags=tags)

        existing = self.get(slugify(name))
        lines = existing.body.splitlines() if existing else []
        if any(_similar(lesson, line) for line in lines):
            return existing  # type: ignore[return-value]
        lines.append(f"- {lesson}")
        merged_tags = tuple(dict.fromkeys([*(existing.tags if existing else ()), *tags]))
        return self.write(
            name, "\n".join(lines),
            title=title or (existing.title if existing else name), tags=merged_tags,
        )

    def delete(self, name: str) -> bool:
        path = self._path_for(name)
        if path is None or not path.is_file():
            return False
        try:
            path.unlink()
        except OSError:
            return False
        self._cache.pop(path, None)
        return True

    # -- internals ---------------------------------------------------------

    def _path_for(self, name: str) -> Path | None:
        """Resolve a note name to a path inside the journal, or None.

        A note name arrives from an HTTP request and from the model, so it is
        untrusted: anything that escapes the directory is refused rather than
        sanitised into something surprising.
        """
        slug = slugify(name.removesuffix(".md"))
        if not slug:
            return None
        path = (self.directory / f"{slug}.md").resolve()
        try:
            root = self.directory.resolve()
        except OSError:
            return None
        return path if path.is_relative_to(root) else None

    def _read(self, path: Path) -> Note | None:
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return None
        cached = self._cache.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        note = parse(text, name=path.stem, path=path)
        self._cache[path] = (stamp, note)
        return note


def _similar(candidate: str, existing: str) -> bool:
    """Close enough that adding it again would be noise."""
    left = re.sub(r"\W+", " ", candidate.lower()).strip()
    right = re.sub(r"\W+", " ", existing.lower().lstrip("-").strip()).strip()
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    a, b = set(left.split()), set(right.split())
    overlap = len(a & b) / max(len(a | b), 1)
    return overlap > 0.8
