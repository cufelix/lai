"""Run tasks on a schedule: a small cron engine plus a background daemon.

Two schedule syntaxes are supported: standard 5-field cron (``*``, lists,
ranges, ``*/n`` steps, and the ``@hourly``/``@daily``/``@weekly``/``@monthly``
shorthands) and a plain interval, ``every:<seconds>``. No third-party cron
library is used — the format is small but easy to get subtly wrong at the
boundaries (a day/month/year rollover, ``*/15`` landing only on :00/:15/:30/:45,
day-of-month vs day-of-week using OR semantics when both are restricted), so
it is worth owning outright and testing hard rather than trusting silently.

Non-obvious design decision: due-ness is decided entirely from a persisted
``next_run`` timestamp, never by re-evaluating a cron expression against wall
clock time on each tick. ``CronMatcher.matches(dt)`` exists and is tested
because a schedule's *spec* needs to be checkable in isolation, but
:class:`Scheduler` never calls it in normal operation — only ``next_after``,
once per firing. Checking ``matches()`` every tick would silently skip a task
whose exact due minute the process slept through (laptop suspend, a starved
thread); timestamp comparison instead fires it as soon as a tick next runs.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

STORE_FILENAME = "schedule.json"
DEFAULT_POLL_SECONDS = 5.0

_CRON_SHORTHANDS: dict[str, str] = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
}

# Bound on next_after()'s field-jump loop, so an unschedulable expression
# (e.g. "0 0 30 2 *" — 30 February never exists) raises instead of hanging.
# The common case resolves in well under 100 iterations.
_MAX_NEXT_AFTER_ITERATIONS = 20_000


# -- cron parsing ----------------------------------------------------------


def _parse_field(raw: str, low: int, high: int) -> set[int]:
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty field component in {raw!r}")
        values |= _parse_component(part, low, high)
    return values


def _parse_component(part: str, low: int, high: int) -> set[int]:
    if "/" in part:
        base, step_text = part.split("/", 1)
        try:
            step = int(step_text)
        except ValueError:
            raise ValueError(f"invalid step {step_text!r} in {part!r}") from None
        if step <= 0:
            raise ValueError(f"step must be positive, got {step} in {part!r}")
        if base == "*":
            start = low
            span = range(low, high + 1)
        elif "-" in base:
            start, span = _parse_range(base, low, high)
        else:
            start = _parse_int(base, low, high)
            span = range(start, high + 1)
        return {v for v in span if (v - start) % step == 0}

    if part == "*":
        return set(range(low, high + 1))
    if "-" in part:
        _, span = _parse_range(part, low, high)
        return set(span)
    return {_parse_int(part, low, high)}


def _parse_range(part: str, low: int, high: int) -> tuple[int, range]:
    start_text, _, end_text = part.partition("-")
    start = _parse_int(start_text, low, high)
    end = _parse_int(end_text, low, high)
    if start > end:
        raise ValueError(f"range start {start} is after end {end} in {part!r}")
    return start, range(start, end + 1)


def _parse_int(text: str, low: int, high: int) -> int:
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"expected an integer, got {text!r}") from None
    if value < low or value > high:
        raise ValueError(f"value {value} out of range [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class CronMatcher:
    """A parsed 5-field cron expression."""

    minutes: frozenset[int]
    hours: frozenset[int]
    doms: frozenset[int]
    months: frozenset[int]
    dows: frozenset[int]
    dom_is_star: bool
    dow_is_star: bool

    def matches(self, dt: datetime) -> bool:
        if dt.minute not in self.minutes:
            return False
        if dt.hour not in self.hours:
            return False
        if dt.month not in self.months:
            return False
        return self._day_matches(dt)

    def _day_matches(self, dt: datetime) -> bool:
        dom_match = dt.day in self.doms
        dow_match = (dt.isoweekday() % 7) in self.dows
        if self.dom_is_star and self.dow_is_star:
            return True
        if self.dom_is_star:
            return dow_match
        if self.dow_is_star:
            return dom_match
        # Standard cron quirk: when *both* fields are restricted, a match on
        # either one is enough (they are OR'd, not AND'd).
        return dom_match or dow_match

    def next_after(self, after: datetime) -> datetime:
        """The next minute strictly after ``after`` that this schedule matches."""
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(_MAX_NEXT_AFTER_ITERATIONS):
            if candidate.month not in self.months:
                candidate = _advance_month(candidate, self.months)
                continue
            if not self._day_matches(candidate):
                candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if candidate.hour not in self.hours:
                nxt = _next_or_none(self.hours, candidate.hour)
                candidate = (
                    candidate.replace(hour=nxt, minute=0)
                    if nxt is not None
                    else (candidate + timedelta(days=1)).replace(hour=0, minute=0)
                )
                continue
            if candidate.minute not in self.minutes:
                nxt = _next_or_none(self.minutes, candidate.minute)
                candidate = (
                    candidate.replace(minute=nxt)
                    if nxt is not None
                    else (candidate + timedelta(hours=1)).replace(minute=0)
                )
                continue
            return candidate
        raise ValueError("cron schedule has no matching time within the lookahead window")


def _next_or_none(values: frozenset[int], current: int) -> int | None:
    higher = [v for v in values if v > current]
    return min(higher) if higher else None


def _advance_month(candidate: datetime, months: frozenset[int]) -> datetime:
    higher = sorted(m for m in months if m > candidate.month)
    if higher:
        return candidate.replace(month=higher[0], day=1, hour=0, minute=0)
    return candidate.replace(year=candidate.year + 1, month=min(months), day=1, hour=0, minute=0)


def parse_cron(expr: str) -> CronMatcher:
    """Parse a 5-field cron expression or one of the ``@`` shorthands."""
    expr = expr.strip()
    expr = _CRON_SHORTHANDS.get(expr, expr)
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"cron expression must have 5 fields, got {len(parts)}: {expr!r}")
    minute, hour, dom, month, dow = parts
    minutes = _parse_field(minute, 0, 59)
    hours = _parse_field(hour, 0, 23)
    doms = _parse_field(dom, 1, 31)
    months = _parse_field(month, 1, 12)
    dows_raw = _parse_field(dow, 0, 7)
    dows = {0 if d == 7 else d for d in dows_raw}
    return CronMatcher(
        minutes=frozenset(minutes),
        hours=frozenset(hours),
        doms=frozenset(doms),
        months=frozenset(months),
        dows=frozenset(dows),
        dom_is_star=(dom.strip() == "*"),
        dow_is_star=(dow.strip() == "*"),
    )


def next_run_after(schedule: str, after: datetime) -> datetime:
    """Compute the next fire time for either a cron string or ``every:<seconds>``."""
    schedule = schedule.strip()
    if schedule.startswith("every:"):
        return after + timedelta(seconds=_parse_interval(schedule))
    return parse_cron(schedule).next_after(after)


def describe_schedule(schedule: str) -> str:
    """Render a schedule the way a person would say it.

    Best-effort by design: an expression this cannot phrase is echoed back
    verbatim rather than described wrongly.
    """
    schedule = schedule.strip()
    if schedule.startswith("every:"):
        try:
            seconds = _parse_interval(schedule)
        except ValueError:
            return schedule
        for size, unit in ((86400.0, "day"), (3600.0, "hour"), (60.0, "minute")):
            if seconds >= size and seconds % size == 0:
                count = int(seconds // size)
                return f"every {unit}" if count == 1 else f"every {count} {unit}s"
        return f"every {seconds:g}s"
    return _CRON_PHRASES.get(schedule, schedule)


_CRON_PHRASES: dict[str, str] = {
    "@hourly": "every hour",
    "@daily": "every day at midnight",
    "@weekly": "every Sunday at midnight",
    "@monthly": "on the 1st of each month",
    "0 * * * *": "every hour",
    "0 0 * * *": "every day at midnight",
    "0 0 * * 0": "every Sunday at midnight",
    "0 0 1 * *": "on the 1st of each month",
    "* * * * *": "every minute",
}


def _parse_interval(schedule: str) -> float:
    _, _, raw = schedule.partition(":")
    try:
        seconds = float(raw)
    except ValueError:
        raise ValueError(f"invalid interval schedule {schedule!r}; expected 'every:<seconds>'") from None
    if seconds <= 0:
        raise ValueError(f"interval must be > 0 seconds, got {seconds}")
    return seconds


# -- task model --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    """One entry in the schedule."""

    id: str
    name: str
    task: str
    """The prompt to run when this task fires."""
    schedule: str
    """A 5-field cron expression, an ``@`` shorthand, or ``every:<seconds>``."""
    mode: str = "auto"
    enabled: bool = True
    last_run: float | None = None
    next_run: float | None = None
    runs: int = 0
    failures: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "task": self.task, "schedule": self.schedule,
            "mode": self.mode, "enabled": self.enabled, "last_run": self.last_run,
            "next_run": self.next_run, "runs": self.runs, "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScheduledTask:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            task=str(data["task"]),
            schedule=str(data["schedule"]),
            mode=str(data.get("mode", "auto")),
            enabled=bool(data.get("enabled", True)),
            last_run=_optional_float(data.get("last_run")),
            next_run=_optional_float(data.get("next_run")),
            runs=int(data.get("runs", 0)),
            failures=int(data.get("failures", 0)),
        )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def make_task(
    *, name: str, task: str, schedule: str, mode: str = "auto", at: datetime | None = None
) -> ScheduledTask:
    """Build a new :class:`ScheduledTask`, validating ``schedule`` eagerly.

    Raises ``ValueError`` on a bad cron expression immediately, rather than
    the first time the scheduler tries to use it.
    """
    if not name or not name.strip():
        raise ValueError("scheduled task name must not be empty")
    if not task or not task.strip():
        raise ValueError("scheduled task prompt must not be empty")
    reference = at or datetime.now()
    next_run = next_run_after(schedule, reference)
    return ScheduledTask(
        id=uuid.uuid4().hex[:12],
        name=name,
        task=task,
        schedule=schedule,
        mode=mode,
        enabled=True,
        last_run=None,
        next_run=next_run.timestamp(),
        runs=0,
        failures=0,
    )


# -- persistence -------------------------------------------------------


class TaskStore:
    """JSON-file persistence for scheduled tasks, tolerant of a bad file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read_all(self) -> dict[str, ScheduledTask]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        entries = raw.get("tasks")
        if not isinstance(entries, dict):
            return {}
        tasks: dict[str, ScheduledTask] = {}
        for task_id, data in entries.items():
            if not isinstance(data, dict):
                continue
            try:
                tasks[task_id] = ScheduledTask.from_dict(data)
            except (KeyError, TypeError, ValueError):
                continue  # one corrupt entry must not take down the whole file
        return tasks

    def _write_all(self, tasks: dict[str, ScheduledTask]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tasks": {task_id: task.to_dict() for task_id, task in tasks.items()}}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def add(self, task: ScheduledTask) -> ScheduledTask:
        with self._lock:
            tasks = self._read_all()
            tasks[task.id] = task
            self._write_all(tasks)
            return task

    def remove(self, task_id: str) -> bool:
        with self._lock:
            tasks = self._read_all()
            if task_id not in tasks:
                return False
            del tasks[task_id]
            self._write_all(tasks)
            return True

    def get(self, task_id: str) -> ScheduledTask | None:
        with self._lock:
            return self._read_all().get(task_id)

    def list(self, *, enabled_only: bool = False) -> list[ScheduledTask]:
        with self._lock:
            tasks = list(self._read_all().values())
        if enabled_only:
            tasks = [t for t in tasks if t.enabled]
        return sorted(tasks, key=lambda t: t.name)

    def update(self, task: ScheduledTask) -> ScheduledTask:
        with self._lock:
            tasks = self._read_all()
            if task.id not in tasks:
                raise KeyError(f"no scheduled task with id {task.id!r}")
            tasks[task.id] = task
            self._write_all(tasks)
            return task

    @classmethod
    def for_config(cls, config) -> TaskStore:
        return cls(Path(config.home) / STORE_FILENAME)


# -- daemon --------------------------------------------------------------


class Scheduler:
    """Background thread that fires due tasks through a callback.

    The callback runs synchronously on the scheduler's thread, so a slow one
    delays later ticks — an accepted trade-off for a handful of desktop-scale
    tasks; dispatching each firing to its own thread is not needed here.
    """

    def __init__(
        self,
        store: TaskStore,
        callback: Callable[[ScheduledTask], None],
        *,
        interval: float = DEFAULT_POLL_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.callback = callback
        self.interval = interval
        self._clock = clock or datetime.now
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lai-scheduler", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def due_now(self, at: datetime | None = None) -> list[ScheduledTask]:
        at = at or self._clock()
        cutoff = at.timestamp()
        return [
            task for task in self.store.list()
            if task.enabled and task.next_run is not None and task.next_run <= cutoff
        ]

    def tick(self, at: datetime | None = None) -> list[ScheduledTask]:
        """Fire every currently-due task once. Returns the tasks fired."""
        at = at or self._clock()
        due = self.due_now(at)
        for task in due:
            self._fire(task, at)
        return due

    def _fire(self, task: ScheduledTask, at: datetime) -> None:
        ok = True
        try:
            self.callback(task)
        except Exception:
            ok = False  # recorded on the task below, never raised further

        try:
            next_run_ts: float | None = next_run_after(task.schedule, at).timestamp()
        except ValueError:
            next_run_ts = None  # schedule became unschedulable; don't crash the loop over it

        updated = replace(
            task,
            last_run=at.timestamp(),
            next_run=next_run_ts,
            runs=task.runs + 1,
            failures=task.failures + (0 if ok else 1),
        )
        try:
            self.store.update(updated)
        except KeyError:
            pass  # the task was removed while it was firing; nothing to update

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                pass  # the loop itself must survive any unexpected failure
            self._stop.wait(self.interval)


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "STORE_FILENAME",
    "CronMatcher",
    "ScheduledTask",
    "Scheduler",
    "TaskStore",
    "describe_schedule",
    "make_task",
    "next_run_after",
    "parse_cron",
]
