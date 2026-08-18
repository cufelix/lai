"""Who is allowed to drive this desktop remotely.

A Telegram bot token is a bearer credential: anyone who finds the bot can send
it messages. So the token is not the security boundary — this file is. Nothing
runs until a sender is on the allowlist, and the only way onto the allowlist
from outside is a one-time pairing code that the operator has to read off their
own terminal.

Denials are deliberately quiet by default: an unknown sender gets a flat refusal
with no hint that a pairing mechanism exists.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .base import IncomingMessage

PAIRING_CODE_TTL = 900.0
PAIRING_CODE_ATTEMPTS = 5


class Access(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    PAIRING = "pairing"
    """The message was a pairing attempt and should be answered, not executed."""


@dataclass(frozen=True, slots=True)
class AccessDecision:
    access: Access
    reason: str = ""
    admin: bool = False

    @property
    def allowed(self) -> bool:
        return self.access is Access.ALLOWED


@dataclass(slots=True)
class Principal:
    channel: str
    sender: str
    name: str = ""
    admin: bool = False
    added_at: float = field(default_factory=time.time)
    mode: str = ""
    """Optional per-principal permission mode override."""

    @property
    def key(self) -> str:
        return f"{self.channel}:{self.sender}"

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "sender": self.sender,
            "name": self.name,
            "admin": self.admin,
            "added_at": self.added_at,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Principal:
        return cls(
            channel=str(data.get("channel", "")),
            sender=str(data.get("sender", "")),
            name=str(data.get("name", "")),
            admin=bool(data.get("admin", False)),
            added_at=float(data.get("added_at", time.time())),
            mode=str(data.get("mode", "")),
        )


class AccessPolicy:
    """Allowlist plus a time-limited pairing code, persisted to disk."""

    def __init__(self, path: Path | None = None, *, open_access: bool = False) -> None:
        self.path = Path(path) if path else None
        self.open_access = open_access
        """Dangerous: accept anyone. Only ever for a throwaway sandbox."""
        self._principals: dict[str, Principal] = {}
        self._lock = threading.RLock()
        self._code: str = ""
        self._code_expires: float = 0.0
        self._code_attempts: int = 0
        self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        if not self.path or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except (json.JSONDecodeError, OSError):
            return  # a corrupt allowlist must fail closed, not crash
        for entry in data.get("principals", []):
            if not isinstance(entry, dict):
                continue
            principal = Principal.from_dict(entry)
            if principal.channel and principal.sender:
                self._principals[principal.key] = principal
        self.open_access = bool(data.get("open_access", self.open_access))

    def _save(self) -> None:
        if not self.path:
            return
        payload = {
            "open_access": self.open_access,
            "principals": [p.to_dict() for p in self._principals.values()],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.path.chmod(0o600)
        except OSError:
            pass

    # -- allowlist -------------------------------------------------------

    def allow(self, channel: str, sender: str, *, name: str = "", admin: bool = False) -> Principal:
        with self._lock:
            principal = Principal(channel=channel, sender=str(sender), name=name, admin=admin)
            self._principals[principal.key] = principal
            self._save()
            return principal

    def revoke(self, channel: str, sender: str) -> bool:
        with self._lock:
            removed = self._principals.pop(f"{channel}:{sender}", None) is not None
            if removed:
                self._save()
            return removed

    def principals(self) -> list[Principal]:
        with self._lock:
            return sorted(self._principals.values(), key=lambda p: (p.channel, p.sender))

    def get(self, channel: str, sender: str) -> Principal | None:
        with self._lock:
            return self._principals.get(f"{channel}:{sender!s}")

    def set_open_access(self, value: bool) -> None:
        with self._lock:
            self.open_access = bool(value)
            self._save()

    # -- pairing ---------------------------------------------------------

    def new_pairing_code(self, *, ttl: float = PAIRING_CODE_TTL) -> str:
        """Mint a code the operator reads off their terminal and sends to the bot."""
        with self._lock:
            self._code = f"{secrets.randbelow(10**6):06d}"
            self._code_expires = time.monotonic() + ttl
            self._code_attempts = 0
            return self._code

    @property
    def pairing_active(self) -> bool:
        return bool(self._code) and time.monotonic() < self._code_expires

    def clear_pairing_code(self) -> None:
        with self._lock:
            self._code = ""
            self._code_expires = 0.0
            self._code_attempts = 0

    def redeem(self, code: str, channel: str, sender: str, *, name: str = "") -> AccessDecision:
        """Consume the pairing code. Single use, rate limited, constant-time compare."""
        with self._lock:
            if not self.pairing_active:
                return AccessDecision(Access.DENIED, "no pairing is in progress")
            self._code_attempts += 1
            if self._code_attempts > PAIRING_CODE_ATTEMPTS:
                self.clear_pairing_code()
                return AccessDecision(Access.DENIED, "too many attempts; the code was cancelled")
            if not secrets.compare_digest(code.strip(), self._code):
                remaining = PAIRING_CODE_ATTEMPTS - self._code_attempts
                return AccessDecision(Access.DENIED, f"wrong code ({remaining} attempts left)")

            # First principal on an empty allowlist becomes the admin.
            admin = not self._principals
            self.clear_pairing_code()
            principal = self.allow(channel, sender, name=name, admin=admin)
            return AccessDecision(
                Access.ALLOWED,
                f"paired as {'admin' if principal.admin else 'user'}",
                admin=principal.admin,
            )

    # -- the gate --------------------------------------------------------

    def check(self, message: IncomingMessage) -> AccessDecision:
        principal = self.get(message.channel, message.sender)
        if principal is not None:
            return AccessDecision(Access.ALLOWED, "allowlisted", admin=principal.admin)

        command, argument = message.command
        if command == "pair" and argument:
            return AccessDecision(Access.PAIRING, "pairing attempt")

        if self.open_access:
            return AccessDecision(Access.ALLOWED, "open access is enabled")

        return AccessDecision(
            Access.DENIED,
            f"sender {message.sender} is not authorised on {message.channel}",
        )

    def summary(self) -> dict:
        return {
            "open_access": self.open_access,
            "principals": [p.to_dict() for p in self.principals()],
            "pairing_active": self.pairing_active,
            "path": str(self.path) if self.path else None,
        }
