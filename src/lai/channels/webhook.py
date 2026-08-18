"""Generic webhook channel — glue for everything that is not Telegram.

Outbound posts JSON to a URL, optionally HMAC-signed so the receiver can verify
it. Slack- and Discord-compatible payload shapes are emitted too, so a plain
"incoming webhook" URL from either service works with no extra code.

Inbound arrives through the daemon (``POST /channels/webhook``), which calls
:meth:`WebhookChannel.deliver`. That keeps LAI to a single listening socket
instead of one per channel.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import httpx

from .base import Attachment, BaseChannel, IncomingMessage, MessageHandler, OutgoingMessage

DEFAULT_TIMEOUT = 20.0


class WebhookChannel(BaseChannel):
    """HTTP in, HTTP out."""

    name = "webhook"

    def __init__(
        self,
        url: str = "",
        *,
        secret: str = "",
        style: str = "json",
        timeout: float = DEFAULT_TIMEOUT,
        headers: dict | None = None,
    ) -> None:
        super().__init__()
        self.url = (url or "").strip()
        self.secret = secret or ""
        self.style = style if style in ("json", "slack", "discord") else "json"
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.sent = 0

    @property
    def available(self) -> bool:
        # Inbound-only operation is legitimate: no URL still accepts deliveries.
        return True

    def start(self, on_message: MessageHandler) -> None:
        super().start(on_message)

    # -- inbound ---------------------------------------------------------

    def deliver(self, payload: dict, *, signature: str = "") -> IncomingMessage | None:
        """Called by the daemon for an inbound POST. Returns the accepted message."""
        if self.secret and not self.verify(payload, signature):
            self._note("rejected an inbound payload with a bad signature")
            return None
        text = str(payload.get("text") or payload.get("task") or "").strip()
        if not text:
            return None
        message = IncomingMessage(
            channel=self.name,
            sender=str(payload.get("sender") or payload.get("user") or "webhook"),
            sender_name=str(payload.get("sender_name") or ""),
            chat=str(payload.get("chat") or payload.get("channel") or "default"),
            text=text,
            message_id=str(payload.get("id") or int(time.time() * 1000)),
            raw=payload,
        )
        self._deliver(message)
        return message

    def sign(self, payload: dict) -> str:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()

    def verify(self, payload: dict, signature: str) -> bool:
        if not self.secret:
            return True
        return hmac.compare_digest(self.sign(payload), (signature or "").strip())

    # -- outbound --------------------------------------------------------

    def send(self, message: OutgoingMessage) -> str:
        if not self.url or not message.text:
            return ""
        payload = self._payload(message)
        headers = {"content-type": "application/json", **self.headers}
        if self.secret:
            headers["x-lai-signature"] = self.sign(payload)
        try:
            response = httpx.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            if response.status_code >= 400:
                self._note(f"POST {self.url} -> {response.status_code}")
                return ""
            self.sent += 1
            return str(self.sent)
        except httpx.HTTPError as exc:
            self._note(f"POST failed: {exc}")
            return ""

    def _payload(self, message: OutgoingMessage) -> dict:
        if self.style == "slack":
            return {"text": message.text, "mrkdwn": True}
        if self.style == "discord":
            return {"content": message.text[:2000]}
        return {
            "chat": message.chat,
            "text": message.text,
            "attachments": [
                {"kind": a.kind, "filename": a.filename, "mime": a.mime, "size": a.size}
                for a in message.attachments
            ],
            "at": time.time(),
        }


class LocalChannel(BaseChannel):
    """In-process channel: scripting, embedding and tests.

    ``inject()`` pushes a message in as if it arrived from outside; everything
    sent back lands in :attr:`outbox`. No network, no credentials.
    """

    name = "local"

    def __init__(self, *, sender: str = "local", chat: str = "local") -> None:
        super().__init__()
        self.sender = sender
        self.chat = chat
        self.outbox: list[OutgoingMessage] = []

    @property
    def available(self) -> bool:
        return True

    def inject(self, text: str, *, sender: str | None = None, chat: str | None = None,
               attachments: tuple[Attachment, ...] = ()) -> IncomingMessage:
        message = IncomingMessage(
            channel=self.name,
            sender=sender or self.sender,
            sender_name=sender or self.sender,
            chat=chat or self.chat,
            text=text,
            attachments=attachments,
            message_id=str(len(self.outbox) + 1),
        )
        self._deliver(message)
        return message

    def send(self, message: OutgoingMessage) -> str:
        if message.edit_id:
            for index in range(len(self.outbox)):
                if str(index + 1) == message.edit_id:
                    self.outbox[index] = message
                    return message.edit_id
        self.outbox.append(message)
        return str(len(self.outbox))

    @property
    def texts(self) -> list[str]:
        return [m.text for m in self.outbox if m.text]

    def last(self) -> OutgoingMessage | None:
        return self.outbox[-1] if self.outbox else None

    def clear(self) -> None:
        self.outbox.clear()
