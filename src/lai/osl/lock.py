"""One desktop, one agent — across processes.

The daemon already refuses to run two tasks at once, but that gate lives in
one process's memory. Nothing stopped `lai do` in one terminal, `lai web` in
another and a scheduled run from all driving the same mouse at the same time,
which does not produce three half-finished tasks: it produces one mangled
desktop and three agents confidently reporting on a screen that somebody else
just changed.

So the claim is a file lock. ``flock`` specifically, because the kernel drops
it when the holding process dies — a crashed agent must never leave the
desktop permanently claimed, and no amount of stale-PID heuristics is as
reliable as that. The lock file records who holds it, so the refusal can name
the process and the task rather than saying "busy".

Observation is deliberately outside this: reading the screen while another
agent works is useful, not dangerous. Only acting takes the lock.
"""

from __future__ import annotations

import errno
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

LOCK_FILENAME = "desktop.lock"
DEFAULT_WAIT = 0.0
POLL_INTERVAL = 0.25

# Holding a flock means holding the file *open*. If the caller lets its
# DesktopLock go out of scope, the handle is closed by the garbage collector
# and the desktop is silently released while an agent is still driving it —
# the exact failure this module exists to prevent. So an acquired lock is kept
# alive here until it is released.
_ACTIVE: set = set()


@dataclass(frozen=True, slots=True)
class Holder:
    """Who currently has the desktop."""

    pid: int = 0
    task: str = ""
    since: float = 0.0
    host: str = ""

    def describe(self) -> str:
        parts = [f"pid {self.pid}"] if self.pid else []
        if self.task:
            parts.append(f'"{self.task[:60]}"')
        if self.since:
            parts.append(f"for {max(0.0, time.time() - self.since):.0f}s")
        return ", ".join(parts) or "another process"


class DesktopBusy(RuntimeError):
    """Somebody else is driving. Carries who, so the message can say so."""

    def __init__(self, holder: Holder) -> None:
        super().__init__(f"the desktop is being driven by {holder.describe()}")
        self.holder = holder


class DesktopLock:
    """An advisory, cross-process claim on the desktop.

    Re-entrant within one process: the daemon may already hold it when a tool
    starts a subagent, and that must not deadlock against itself.
    """

    __slots__ = ("__weakref__", "_depth", "_handle", "path", "task")

    def __init__(self, path: str | Path, *, task: str = "") -> None:
        self.path = Path(path)
        self.task = task
        self._handle = None
        self._depth = 0

    @classmethod
    def for_home(cls, home: str | Path, *, task: str = "") -> DesktopLock:
        return cls(Path(home) / LOCK_FILENAME, task=task)

    @property
    def held(self) -> bool:
        return self._depth > 0

    def holder(self) -> Holder:
        """Who the lock file says is holding it. Best effort — it may be stale."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            return Holder()
        return Holder(
            pid=int(data.get("pid") or 0),
            task=str(data.get("task") or ""),
            since=float(data.get("since") or 0.0),
            host=str(data.get("host") or ""),
        )

    def acquire(self, *, wait: float = DEFAULT_WAIT) -> bool:
        """Take the desktop. Raises :class:`DesktopBusy` if someone else has it."""
        if self._depth:
            self._depth += 1
            return True

        try:
            import fcntl  # noqa: PLC0415
        except ImportError:  # pragma: no cover - not Linux
            self._depth = 1
            return True

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the lock's lifetime
        deadline = time.monotonic() + max(0.0, wait)
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    handle.close()
                    raise
                if time.monotonic() >= deadline:
                    holder = self.holder()
                    handle.close()
                    raise DesktopBusy(holder) from None
                time.sleep(POLL_INTERVAL)

        self._handle = handle
        self._depth = 1
        _ACTIVE.add(self)
        self._stamp()
        return True

    def release(self) -> None:
        if self._depth > 1:
            self._depth -= 1
            return
        self._depth = 0
        _ACTIVE.discard(self)
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            import fcntl  # noqa: PLC0415

            handle.truncate(0)
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, ImportError):
            pass
        finally:
            handle.close()

    def __enter__(self) -> DesktopLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    # -- internals ---------------------------------------------------------

    def _stamp(self) -> None:
        """Record who we are, so a refusal elsewhere can name this process."""
        if self._handle is None:
            return
        import socket  # noqa: PLC0415

        payload = {
            "pid": os.getpid(),
            "task": self.task[:200],
            "since": time.time(),
            "host": socket.gethostname(),
        }
        try:
            self._handle.seek(0)
            self._handle.truncate(0)
            self._handle.write(json.dumps(payload))
            self._handle.flush()
        except OSError:
            pass
