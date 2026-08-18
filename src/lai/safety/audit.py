"""Append-only audit log.

Every tool call, permission verdict and model turn lands here as one JSON line,
redacted. This is the record you read when you want to know what the agent
actually did to your machine.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .redact import redact_obj


@dataclass(slots=True)
class AuditEvent:
    kind: str
    data: dict = field(default_factory=dict)
    at: float = field(default_factory=time.time)
    session: str = ""

    def to_dict(self) -> dict:
        return {
            "ts": round(self.at, 3),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.at)),
            "session": self.session,
            "kind": self.kind,
            **self.data,
        }


class AuditLog:
    """Thread-safe JSONL writer with optional live subscribers."""

    def __init__(
        self,
        path: Path | None,
        *,
        session: str = "",
        redact: bool = True,
        echo: Callable[[AuditEvent], None] | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.session = session
        self.redact = redact
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[AuditEvent], None]] = []
        if echo:
            self._subscribers.append(echo)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, callback: Callable[[AuditEvent], None]) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def write(self, kind: str, **data: Any) -> AuditEvent:
        payload = redact_obj(data, enabled=self.redact)
        event = AuditEvent(kind=kind, data=payload, session=self.session)
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=_fallback)
        if self.path:
            with self._lock:
                try:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                except OSError:
                    pass  # never let logging break the run
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception:
                continue
        return event

    def read(self, *, limit: int = 200) -> list[dict]:
        if not self.path or not self.path.is_file():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        out: list[dict] = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    @classmethod
    def for_session(cls, logs_dir: Path, session_id: str, **kwargs) -> AuditLog:
        day = time.strftime("%Y-%m-%d")
        return cls(Path(logs_dir) / f"{day}.jsonl", session=session_id, **kwargs)

    @classmethod
    def disabled(cls) -> AuditLog:
        return cls(None)


def _fallback(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return repr(value)[:400]
