"""Long-term memory: what the agent has learned, kept across sessions.

A desktop agent without memory relearns the same things every run — that a
particular app's save dialog names its field "Name:" instead of "Filename:",
that the user prefers dark themes, that a given site's login form needs a
scroll before the button is visible. None of that is task state; it belongs
outside any one :class:`~lai.agent.session.Session` and needs to survive the
process exiting. This module is a small SQLite-backed key/value-ish store for
exactly that: durable, searchable notes indexed by *kind* (what sort of thing
it is) and an optional *key* (so a fact can be looked up and overwritten
rather than accumulating duplicates forever).

Non-obvious design decision — search without a hard FTS5 dependency: SQLite's
FTS5 extension gives proper ranked full-text search, but it is a *compile-time*
option and some distributions (notably musl/Alpine builds, some minimal
container images) ship a ``sqlite3`` without it. Rather than making memory
search fail outright on those systems, :class:`MemoryStore` probes for FTS5 at
open time with a throwaway ``CREATE VIRTUAL TABLE`` and falls back to a
LIKE-based scorer when it is unavailable. The LIKE fallback is cruder than
BM25 (it counts substring hits, weighting the key and tags above the body) but
it keeps memory usable everywhere LAI runs, which matters more here than
search quality.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DB_FILENAME = "memory.db"

# Guidance kinds — see src/lai/tools/agentic.py's memory_save description for
# how the model is steered toward these.
KINDS: tuple[str, ...] = ("fact", "app", "preference", "task")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    key TEXT,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    source_session TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind);
CREATE INDEX IF NOT EXISTS idx_memory_key ON memory(key);
"""


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One remembered thing."""

    id: int
    kind: str
    content: str
    key: str | None = None
    tags: tuple[str, ...] = ()
    source_session: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    hits: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "key": self.key,
            "content": self.content,
            "tags": list(self.tags),
            "source_session": self.source_session,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hits": self.hits,
        }


class MemoryStore:
    """SQLite-backed long-term memory, safe to share across threads.

    One connection is opened with ``check_same_thread=False`` and every public
    method takes an internal :class:`threading.RLock` (re-entrant, since some
    methods call each other while already holding it) before touching the
    database. That is enough for LAI's usage pattern — occasional reads and
    writes from a handful of threads (the agent loop, a scheduler thread) —
    without pulling in a connection pool.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._fts_enabled = self._try_enable_fts()
        self._conn.commit()

    @classmethod
    def open(cls, home: str | Path) -> MemoryStore:
        """Open (creating if needed) the store at ``<home>/memory.db``."""
        return cls(Path(home) / DB_FILENAME)

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- fts setup ---------------------------------------------------------

    def _try_enable_fts(self) -> bool:
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
                "USING fts5(content, tags, tokenize='unicode61')"
            )
        except sqlite3.OperationalError:
            return False
        return True

    def _sync_fts(self, entry_id: int, content: str, tags_str: str) -> None:
        if not self._fts_enabled:
            return
        self._conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (entry_id,))
        self._conn.execute(
            "INSERT INTO memory_fts(rowid, content, tags) VALUES (?, ?, ?)",
            (entry_id, content, tags_str),
        )

    # -- writes --------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        kind: str,
        key: str | None = None,
        tags: tuple[str, ...] | list[str] = (),
        source_session: str = "",
    ) -> MemoryEntry:
        """Save a fact. Upserts on ``(kind, key)`` when ``key`` is given.

        Without a key, every call inserts a new row — appropriate for
        one-off notes that are not expected to be revised in place.
        """
        if not content or not content.strip():
            raise ValueError("memory content must not be empty")
        if not kind or not kind.strip():
            raise ValueError("memory kind must not be empty")

        now = time.time()
        tags_str = ",".join(t.strip() for t in tags if t and t.strip())

        with self._lock:
            if key:
                row = self._conn.execute(
                    "SELECT id, created_at, hits FROM memory WHERE kind = ? AND key = ?",
                    (kind, key),
                ).fetchone()
                if row is not None:
                    entry_id, created_at, hits = row
                    self._conn.execute(
                        "UPDATE memory SET content = ?, tags = ?, updated_at = ?, "
                        "source_session = ? WHERE id = ?",
                        (content, tags_str, now, source_session, entry_id),
                    )
                    self._sync_fts(entry_id, content, tags_str)
                    self._conn.commit()
                    return MemoryEntry(
                        id=entry_id, kind=kind, key=key, content=content,
                        tags=tuple(t for t in tags_str.split(",") if t),
                        source_session=source_session, created_at=created_at,
                        updated_at=now, hits=hits,
                    )

            cur = self._conn.execute(
                "INSERT INTO memory(kind, key, content, tags, source_session, "
                "created_at, updated_at, hits) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (kind, key, content, tags_str, source_session, now, now),
            )
            entry_id = cur.lastrowid
            self._sync_fts(entry_id, content, tags_str)
            self._conn.commit()
            return MemoryEntry(
                id=entry_id, kind=kind, key=key, content=content,
                tags=tuple(t for t in tags_str.split(",") if t),
                source_session=source_session, created_at=now, updated_at=now, hits=0,
            )

    def forget(self, id_or_key: int | str) -> int:
        """Delete by numeric id, or by exact ``key`` match across kinds.

        Returns the number of rows removed (0 or 1 for an id, 0+ for a key
        that was reused — upsert only dedupes within one kind).
        """
        with self._lock:
            rid = _coerce_id(id_or_key)
            if rid is not None:
                ids = [rid] if self._conn.execute(
                    "SELECT 1 FROM memory WHERE id = ?", (rid,)
                ).fetchone() else []
            else:
                ids = [
                    r[0]
                    for r in self._conn.execute(
                        "SELECT id FROM memory WHERE key = ?", (str(id_or_key),)
                    ).fetchall()
                ]
            for entry_id in ids:
                self._conn.execute("DELETE FROM memory WHERE id = ?", (entry_id,))
                if self._fts_enabled:
                    self._conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (entry_id,))
            self._conn.commit()
            return len(ids)

    # -- reads -----------------------------------------------------------

    def recall(self, query: str, *, kind: str | None = None, limit: int = 10) -> list[MemoryEntry]:
        """Search memory. Empty query returns the most recently touched entries."""
        with self._lock:
            if not query or not query.strip():
                entries = self.all(kind=kind, limit=limit)
            else:
                entries = self._fts_search(query, kind, limit) if self._fts_enabled else None
                if entries is None:
                    entries = self._like_search(query, kind, limit)
            if entries:
                ids = [e.id for e in entries]
                placeholders = ",".join("?" * len(ids))
                # Interpolated only in shape: `placeholders` is a run of "?" sized
                # from len(ids); every value is still bound, never interpolated.
                self._conn.execute(
                    f"UPDATE memory SET hits = hits + 1 WHERE id IN ({placeholders})",  # noqa: S608
                    ids,
                )
                self._conn.commit()
            return entries

    def _fts_search(self, query: str, kind: str | None, limit: int) -> list[MemoryEntry] | None:
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens:
            return []
        match_expr = " OR ".join(f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens)
        sql = (
            "SELECT m.id, m.kind, m.key, m.content, m.tags, m.source_session, "
            "m.created_at, m.updated_at, m.hits "
            "FROM memory_fts f JOIN memory m ON m.id = f.rowid WHERE f MATCH ?"
        )
        params: list = [match_expr]
        if kind:
            sql += " AND m.kind = ?"
            params.append(kind)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # A malformed MATCH expression (shouldn't happen given the quoting
            # above, but FTS5 syntax is picky) — degrade to LIKE rather than
            # surfacing a search failure to the model.
            return None
        return [_row_to_entry(r) for r in rows]

    def _like_search(self, query: str, kind: str | None, limit: int) -> list[MemoryEntry]:
        """Score rows by counting term occurrences. See module docstring."""
        terms = [t.lower() for t in re.findall(r"\w+", query, flags=re.UNICODE) if t]
        if not terms:
            return []
        sql = (
            "SELECT id, kind, key, content, tags, source_session, "
            "created_at, updated_at, hits FROM memory"
        )
        params: list = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        rows = self._conn.execute(sql, params).fetchall()

        scored: list[tuple[int, tuple]] = []
        for row in rows:
            content_l = (row[3] or "").lower()
            tags_l = (row[4] or "").lower()
            key_l = (row[2] or "").lower()
            score = 0
            for term in terms:
                if term in content_l:
                    score += content_l.count(term)
                if term in tags_l:
                    score += 2
                if term in key_l:
                    score += 3
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda pair: (-pair[0], -pair[1][7]))
        return [_row_to_entry(row) for _, row in scored[:limit]]

    def all(self, kind: str | None = None, limit: int = 100) -> list[MemoryEntry]:
        with self._lock:
            sql = (
                "SELECT id, kind, key, content, tags, source_session, "
                "created_at, updated_at, hits FROM memory"
            )
            params: list = []
            if kind:
                sql += " WHERE kind = ?"
                params.append(kind)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
            return [_row_to_entry(r) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            by_kind = dict(
                self._conn.execute("SELECT kind, COUNT(*) FROM memory GROUP BY kind").fetchall()
            )
            return {
                "total": total,
                "by_kind": by_kind,
                "fts5": self._fts_enabled,
                "path": str(self.path),
            }

    def context_block(self, query: str, limit: int = 5) -> str:
        """Short markdown block for a system prompt; "" when nothing is relevant."""
        entries = self.recall(query, limit=limit)
        if not entries:
            return ""
        lines = ["## Memory", ""]
        for entry in entries:
            suffix = f" ({', '.join(entry.tags)})" if entry.tags else ""
            lines.append(f"- [{entry.kind}] {entry.content}{suffix}")
        return "\n".join(lines)


def _row_to_entry(row: tuple) -> MemoryEntry:
    id_, kind, key, content, tags, source_session, created_at, updated_at, hits = row
    tag_tuple = tuple(t for t in (tags or "").split(",") if t)
    return MemoryEntry(
        id=id_, kind=kind, key=key, content=content, tags=tag_tuple,
        source_session=source_session or "", created_at=created_at,
        updated_at=updated_at, hits=hits,
    )


def _coerce_id(value: int | str) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return None


__all__ = ["DB_FILENAME", "KINDS", "MemoryEntry", "MemoryStore"]
