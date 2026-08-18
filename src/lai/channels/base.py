"""Channel contract — how the outside world reaches the agent.

A channel is anything that can deliver a message to LAI and carry a reply back:
Telegram, a webhook, a chat app, an inbox. Channels know nothing about the
agent; the manager owns that. They deal only in :class:`IncomingMessage` and
:class:`OutgoingMessage`.

The design constraint that shapes everything here: a channel is a **remote
control for someone's desktop**. So identity is not optional — every incoming
message carries a stable sender id, and nothing runs until that sender has been
explicitly allowed (see :mod:`lai.channels.access`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..errors import LaiError


class ChannelError(LaiError):
    code = "channel_error"


@dataclass(frozen=True, slots=True)
class Attachment:
    """A binary payload travelling with a message."""

    kind: str
    """image | file | audio"""
    data: bytes = b""
    filename: str = ""
    mime: str = "application/octet-stream"
    caption: str = ""

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """Something a human sent us."""

    channel: str
    sender: str
    """Stable per-channel identity — the thing the allowlist is keyed on."""
    chat: str
    """Conversation id. One agent session is kept per chat."""
    text: str = ""
    sender_name: str = ""
    attachments: tuple[Attachment, ...] = ()
    message_id: str = ""
    at: float = field(default_factory=time.time)
    raw: dict = field(default_factory=dict)

    @property
    def is_command(self) -> bool:
        return self.text.strip().startswith("/")

    @property
    def command(self) -> tuple[str, str]:
        """``("/mode auto")`` -> ``("mode", "auto")``; ``("", "")`` if not a command."""
        if not self.is_command:
            return ("", "")
        body = self.text.strip()[1:]
        name, _, rest = body.partition(" ")
        # Telegram sends "/status@mybotname" in groups.
        return (name.split("@", 1)[0].lower(), rest.strip())

    @property
    def route(self) -> str:
        """Key identifying the conversation this belongs to."""
        return f"{self.channel}:{self.chat}"

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "sender": self.sender,
            "sender_name": self.sender_name,
            "chat": self.chat,
            "text": self.text,
            "attachments": len(self.attachments),
            "at": self.at,
        }


@dataclass(frozen=True, slots=True)
class OutgoingMessage:
    """Something we are sending back."""

    chat: str
    text: str = ""
    attachments: tuple[Attachment, ...] = ()
    reply_to: str = ""
    edit_id: str = ""
    """When set, replace a previously sent message instead of adding one."""
    silent: bool = False
    markdown: bool = True

    @classmethod
    def image(cls, chat: str, png: bytes, *, caption: str = "", **kwargs) -> OutgoingMessage:
        return cls(
            chat=chat,
            attachments=(Attachment("image", png, "screenshot.png", "image/png", caption),),
            **kwargs,
        )


MessageHandler = Callable[[IncomingMessage], None]


class Channel(Protocol):
    """A transport. Implementations live alongside this module."""

    name: str

    @property
    def available(self) -> bool:
        """False when the channel is not configured (no token, etc.)."""
        ...

    def start(self, on_message: MessageHandler) -> None:
        """Begin delivering messages. Must not block the caller."""
        ...

    def send(self, message: OutgoingMessage) -> str:
        """Deliver a message; return the transport's message id (or "")."""
        ...

    def stop(self) -> None:
        """Stop delivering. Must be safe to call more than once."""
        ...


class BaseChannel:
    """Shared bookkeeping for channel implementations."""

    name = "base"

    def __init__(self) -> None:
        self._handler: MessageHandler | None = None
        self._running = False
        self.errors: list[str] = []

    @property
    def running(self) -> bool:
        return self._running

    @property
    def available(self) -> bool:  # pragma: no cover - overridden
        return True

    def start(self, on_message: MessageHandler) -> None:
        self._handler = on_message
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._handler = None

    def _deliver(self, message: IncomingMessage) -> None:
        """Hand a message to the manager, never letting a handler crash the poller."""
        handler = self._handler
        if handler is None:
            return
        try:
            handler(message)
        except Exception as exc:
            self._note(f"handler failed for {message.route}: {type(exc).__name__}: {exc}")

    def _note(self, problem: str) -> None:
        self.errors.append(problem)
        del self.errors[:-50]

    def send(self, message: OutgoingMessage) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def status(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
            "running": self.running,
            "recent_errors": self.errors[-3:],
        }
