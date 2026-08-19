"""What each backend did last time, so the next run does not repeat it.

A quota lasts hours. Without a memory, every run rediscovers it the same way:
send a request, wait, get a 429, fail over — paying the same toll each time,
and telling the user "ready now" about a backend that has been refusing all
morning.

So failures are written down. A backend that reported a quota is put on
cooldown until the moment it said it would reset (vendors usually say, and the
text is parsed rather than guessed); one that failed transiently gets a short
rest; one that refused authentication is remembered for longer, because a login
does not fix itself. `lai models` shows it, and the fallback chain skips
standbys still cooling — a backend that told us the time it recovers should not
be asked again before then.

The one thing this must never do is get in the way: an unreadable or absurd
health file is ignored, and a backend the user explicitly configured is always
tried, whatever this file says. Being wrong about a recovery time must cost a
retry, never an outage.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

HEALTH_FILENAME = "backends.json"

QUOTA_COOLDOWN = 3600.0
"""Used when a vendor says "out of quota" without saying until when."""

AUTH_COOLDOWN = 900.0
"""Not signed in. Long enough to stop retrying, short enough to notice a fix."""

TRANSIENT_COOLDOWN = 120.0
MAX_COOLDOWN = 6 * 3600.0

# "Your limit will reset at 2026-08-19 12:01:48" and friends.
_RESET_AT = re.compile(
    r"reset(?:s|ting)?\s+(?:at|on)?\s*[:\s]\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)",
    re.IGNORECASE,
)
_RETRY_IN = re.compile(r"(?:retry|try again)\s+(?:in|after)\s+(\d+)\s*(second|minute|hour)", re.IGNORECASE)

QUOTA_MARKERS = ("quota", "usage limit", "rate limit", "rate_limit", "429",
                 "insufficient", "credit", "billing", "session limit")
AUTH_MARKERS = ("401", "403", "unauthorized", "forbidden", "invalid api key",
                "authentication", "not signed in", "please run /login", "no api key")


@dataclass(frozen=True, slots=True)
class Health:
    """One backend's last known state."""

    name: str
    reason: str = ""
    kind: str = "error"
    """quota | auth | error"""
    at: float = 0.0
    until: float = 0.0

    @property
    def cooling(self) -> bool:
        return self.until > time.time()

    @property
    def recovers_in(self) -> float:
        return max(0.0, self.until - time.time())

    def describe(self) -> str:
        if not self.cooling:
            return ""
        minutes = self.recovers_in / 60
        when = f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.1f} h"
        label = {"quota": "out of quota", "auth": "not signed in"}.get(self.kind, "failing")
        return f"{label}, retry in {when}"

    def to_dict(self) -> dict:
        return {"reason": self.reason[:300], "kind": self.kind, "at": self.at, "until": self.until}


def classify(reason: str) -> tuple[str, float]:
    """(kind, cooldown seconds) for a failure message."""
    lowered = (reason or "").lower()
    if any(marker in lowered for marker in QUOTA_MARKERS):
        return "quota", QUOTA_COOLDOWN
    if any(marker in lowered for marker in AUTH_MARKERS):
        return "auth", AUTH_COOLDOWN
    return "error", TRANSIENT_COOLDOWN


def _stated_recovery(reason: str) -> float:
    """When the vendor itself said it would recover, as an absolute time."""
    match = _RESET_AT.search(reason or "")
    if match:
        stamp = match.group(1).replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                # Vendors state these in local time; being an hour out costs a
                # retry, which is the cheap direction to be wrong in.
                return time.mktime(time.strptime(stamp, fmt))
            except ValueError:
                continue
    match = _RETRY_IN.search(reason or "")
    if match:
        amount = int(match.group(1))
        unit = {"second": 1, "minute": 60, "hour": 3600}[match.group(2).lower()]
        return time.time() + amount * unit
    return 0.0


def note_failure(home, name: str, reason: str) -> Health:
    """Record why a backend stepped aside, and when it is worth asking again."""
    kind, cooldown = classify(reason)
    stated = _stated_recovery(reason)
    now = time.time()
    until = stated if stated > now else now + cooldown
    entry = Health(
        name=name, reason=" ".join(str(reason).split())[:300], kind=kind,
        at=now, until=min(until, now + MAX_COOLDOWN),
    )
    _update(home, lambda data: data.__setitem__(name, entry.to_dict()))
    return entry


def note_success(home, name: str) -> None:
    """A backend that just answered is healthy, whatever it did before."""
    _update(home, lambda data: data.pop(name, None))


def read(home) -> dict[str, Health]:
    path = Path(home) / HEALTH_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    found: dict[str, Health] = {}
    for name, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        try:
            found[str(name)] = Health(
                name=str(name),
                reason=str(payload.get("reason") or ""),
                kind=str(payload.get("kind") or "error"),
                at=float(payload.get("at") or 0.0),
                until=float(payload.get("until") or 0.0),
            )
        except (TypeError, ValueError):
            continue
    return found


def cooling(home) -> dict[str, Health]:
    """Backends not worth asking yet."""
    return {name: entry for name, entry in read(home).items() if entry.cooling}


def _update(home, mutate) -> None:
    path = Path(home) / HEALTH_FILENAME
    data = {name: entry.to_dict() for name, entry in read(home).items()}
    mutate(data)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".backends-")
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(temporary, path)
    except OSError:
        # Health is an optimisation. Failing to record it must never fail a run.
        pass
