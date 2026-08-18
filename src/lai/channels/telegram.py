"""Telegram channel — drive the desktop from your phone.

Long-polling against the Bot API in a daemon thread. No inbound port, no
webhook, no TLS certificate: it works from behind NAT on a laptop, which is
exactly where a personal desktop agent lives.

Two Telegram facts shape this module:

* Messages cap at 4096 characters, so long output is split on line boundaries.
* Editing is rate limited far more tightly than sending, so live progress goes
  through a single throttled edit rather than a stream of new messages.
"""

from __future__ import annotations

import html
import threading
import time

import httpx

from .base import Attachment, BaseChannel, IncomingMessage, MessageHandler, OutgoingMessage

API_ROOT = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 4096
POLL_TIMEOUT = 30
EDIT_MIN_INTERVAL = 1.2
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


class TelegramChannel(BaseChannel):
    """A Telegram bot as a front end for the agent."""

    name = "telegram"

    def __init__(self, token: str, *, timeout: float = 45.0, allowed_updates: tuple[str, ...] = ("message",)) -> None:
        super().__init__()
        self.token = (token or "").strip()
        self.allowed_updates = allowed_updates
        self._client = httpx.Client(
            base_url=f"{API_ROOT}/bot{self.token}", timeout=timeout
        ) if self.token else None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._offset = 0
        self._last_edit: dict[str, float] = {}
        self.me: dict = {}

    @property
    def available(self) -> bool:
        return bool(self.token) and self._client is not None

    # -- lifecycle -------------------------------------------------------

    def verify(self) -> dict:
        """Confirm the token works; returns the bot's own profile."""
        data = self._call("getMe")
        self.me = data or {}
        return self.me

    def start(self, on_message: MessageHandler) -> None:
        if not self.available:
            raise RuntimeError("telegram channel has no bot token")
        super().start(on_message)
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="lai-telegram", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        super().stop()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            # The poller can be parked in a 30 s long-poll; don't wait it out.
            thread.join(timeout=2.0)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    # -- receiving -------------------------------------------------------

    def _poll_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                updates = self._call(
                    "getUpdates",
                    {
                        "offset": self._offset,
                        "timeout": POLL_TIMEOUT,
                        "allowed_updates": list(self.allowed_updates),
                    },
                    timeout=POLL_TIMEOUT + 15,
                )
                backoff = 1.0
            except Exception as exc:
                self._note(f"poll failed: {exc}")
                self._stop.wait(min(backoff, 60.0))
                backoff *= 2
                continue

            for update in updates or []:
                try:
                    self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                    message = self._decode(update)
                    if message is not None:
                        self._deliver(message)
                except Exception as exc:
                    self._note(f"bad update: {exc}")

    def _decode(self, update: dict) -> IncomingMessage | None:
        raw = update.get("message") or update.get("edited_message")
        if not isinstance(raw, dict):
            return None
        chat = raw.get("chat") or {}
        sender = raw.get("from") or {}
        text = raw.get("text") or raw.get("caption") or ""

        attachments: list[Attachment] = []
        photos = raw.get("photo") or []
        if photos:
            largest = max(photos, key=lambda p: p.get("file_size", 0))
            data = self._download(largest.get("file_id", ""))
            if data:
                attachments.append(Attachment("image", data, "photo.jpg", "image/jpeg"))
        document = raw.get("document")
        if isinstance(document, dict) and document.get("file_size", 0) <= MAX_DOWNLOAD_BYTES:
            data = self._download(document.get("file_id", ""))
            if data:
                attachments.append(
                    Attachment(
                        "file",
                        data,
                        document.get("file_name", "file"),
                        document.get("mime_type", "application/octet-stream"),
                    )
                )

        name = " ".join(
            part for part in (sender.get("first_name"), sender.get("last_name")) if part
        ) or sender.get("username", "")

        return IncomingMessage(
            channel=self.name,
            sender=str(sender.get("id", "")),
            sender_name=name,
            chat=str(chat.get("id", "")),
            text=text,
            attachments=tuple(attachments),
            message_id=str(raw.get("message_id", "")),
            at=float(raw.get("date", time.time())),
            raw=raw,
        )

    def _download(self, file_id: str) -> bytes:
        if not file_id or self._client is None:
            return b""
        try:
            info = self._call("getFile", {"file_id": file_id})
            path = (info or {}).get("file_path")
            if not path:
                return b""
            response = httpx.get(f"{API_ROOT}/file/bot{self.token}/{path}", timeout=60.0)
            response.raise_for_status()
            return response.content if len(response.content) <= MAX_DOWNLOAD_BYTES else b""
        except Exception as exc:
            self._note(f"download failed: {exc}")
            return b""

    # -- sending ---------------------------------------------------------

    def send(self, message: OutgoingMessage) -> str:
        if not self.available:
            return ""
        last_id = ""

        if message.text:
            chunks = _split(message.text, MAX_MESSAGE_CHARS)
            if message.edit_id and chunks:
                last_id = self._edit(message, chunks[0])
                chunks = chunks[1:]
            for chunk in chunks:
                last_id = self._send_text(message, chunk) or last_id

        for attachment in message.attachments:
            sent = self._send_attachment(message, attachment)
            last_id = sent or last_id
        return last_id

    def _send_text(self, message: OutgoingMessage, text: str) -> str:
        payload = {
            "chat_id": message.chat,
            "text": _format(text, message.markdown),
            "disable_notification": message.silent,
            "link_preview_options": {"is_disabled": True},
        }
        if message.markdown:
            payload["parse_mode"] = "HTML"
        if message.reply_to:
            payload["reply_parameters"] = {"message_id": int(message.reply_to)}
        result = self._call("sendMessage", payload)
        return str((result or {}).get("message_id", ""))

    def _edit(self, message: OutgoingMessage, text: str) -> str:
        """Throttled: Telegram rate-limits edits much harder than sends."""
        now = time.monotonic()
        key = f"{message.chat}:{message.edit_id}"
        if now - self._last_edit.get(key, 0.0) < EDIT_MIN_INTERVAL:
            return message.edit_id
        self._last_edit[key] = now
        payload = {
            "chat_id": message.chat,
            "message_id": int(message.edit_id),
            "text": _format(text, message.markdown),
        }
        if message.markdown:
            payload["parse_mode"] = "HTML"
        result = self._call("editMessageText", payload, quiet_errors=("message is not modified",))
        return str((result or {}).get("message_id", message.edit_id))

    def _send_attachment(self, message: OutgoingMessage, attachment: Attachment) -> str:
        if self._client is None or not attachment.data:
            return ""
        method, field = ("sendPhoto", "photo") if attachment.kind == "image" else ("sendDocument", "document")
        data: dict = {"chat_id": message.chat}
        if attachment.caption:
            data["caption"] = attachment.caption[:1000]
        try:
            response = self._client.post(
                f"/{method}",
                data=data,
                files={field: (attachment.filename or "file", attachment.data, attachment.mime)},
                timeout=120.0,
            )
            body = response.json()
            if not body.get("ok"):
                self._note(f"{method} failed: {body.get('description', '')}")
                return ""
            return str((body.get("result") or {}).get("message_id", ""))
        except Exception as exc:
            self._note(f"{method} failed: {exc}")
            return ""

    # -- transport -------------------------------------------------------

    def _call(
        self,
        method: str,
        payload: dict | None = None,
        *,
        timeout: float | None = None,
        quiet_errors: tuple[str, ...] = (),
    ):
        if self._client is None:
            raise RuntimeError("telegram channel is not configured")
        response = self._client.post(f"/{method}", json=payload or {}, timeout=timeout)
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{method}: malformed response ({response.status_code})") from exc
        if not body.get("ok"):
            description = str(body.get("description", ""))
            if any(fragment in description for fragment in quiet_errors):
                return None
            raise RuntimeError(f"{method}: {description or response.status_code}")
        return body.get("result")


def _format(text: str, markdown: bool) -> str:
    """Escape for Telegram's HTML mode.

    HTML rather than MarkdownV2 on purpose: MarkdownV2 requires escaping 18
    different characters and rejects the whole message if one is missed, which
    is a poor fit for arbitrary agent output.
    """
    if not markdown:
        return text[:MAX_MESSAGE_CHARS]
    return html.escape(text, quote=False)[:MAX_MESSAGE_CHARS]


def _split(text: str, limit: int) -> list[str]:
    """Split on line boundaries where possible, hard-split only when forced."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks
