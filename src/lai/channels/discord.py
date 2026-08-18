"""Discord channel — the bot gateway, not just an incoming webhook.

Discord has no long-poll endpoint: a bot receives events over a WebSocket
gateway and sends replies over the REST API. So this module runs two halves.

The part that is easy to get wrong is the heartbeat. The gateway sends a
``heartbeat_interval`` in HELLO and closes the connection if it stops hearing
from you, so heartbeats run on their own thread independent of message
handling. If the socket drops we reconnect with backoff and RESUME where
possible, because a fresh IDENTIFY is rate limited far more aggressively.
"""

from __future__ import annotations

import json
import threading
import time

import httpx

from .base import Attachment, BaseChannel, IncomingMessage, MessageHandler, OutgoingMessage

API_ROOT = "https://discord.com/api/v10"
GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
MAX_MESSAGE_CHARS = 2000

# GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT
DEFAULT_INTENTS = 512 | 4096 | 32768

OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_RESUME = 6
OP_RECONNECT = 7
OP_INVALID_SESSION = 9
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11


class DiscordChannel(BaseChannel):
    """A Discord bot as a front end for the agent."""

    name = "discord"

    def __init__(self, token: str, *, intents: int = DEFAULT_INTENTS) -> None:
        super().__init__()
        self.token = (token or "").strip()
        self.intents = intents
        self.me: dict = {}
        self._thread: threading.Thread | None = None
        self._heartbeat: threading.Thread | None = None
        self._stop = threading.Event()
        self._socket = None
        self._sequence: int | None = None
        self._session_id: str = ""
        self._resume_url: str = ""
        self._send_lock = threading.Lock()
        self._client = (
            httpx.Client(
                base_url=API_ROOT,
                headers={
                    "authorization": f"Bot {self.token}",
                    "content-type": "application/json",
                    "user-agent": "LAI (https://github.com/lai, 0.1)",
                },
                timeout=30.0,
            )
            if self.token
            else None
        )

    @property
    def available(self) -> bool:
        return bool(self.token) and self._client is not None

    def verify(self) -> dict:
        """Confirm the bot token works; returns the bot's own user object."""
        if self._client is None:
            raise RuntimeError("discord channel has no bot token")
        response = self._client.get("/users/@me")
        if response.status_code != 200:
            raise RuntimeError(f"token rejected ({response.status_code}): {response.text[:200]}")
        self.me = response.json()
        return self.me

    # -- lifecycle -------------------------------------------------------

    def start(self, on_message: MessageHandler) -> None:
        if not self.available:
            raise RuntimeError("discord channel has no bot token")
        try:
            import websockets  # noqa: F401,PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("the discord channel needs the 'websockets' package") from exc
        super().start(on_message)
        self._stop.clear()
        self._thread = threading.Thread(target=self._gateway_loop, name="lai-discord", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        super().stop()
        for thread in (self._heartbeat, self._thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self._heartbeat = self._thread = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    # -- gateway ---------------------------------------------------------

    def _gateway_loop(self) -> None:
        import asyncio  # noqa: PLC0415

        backoff = 1.0
        while not self._stop.is_set():
            try:
                asyncio.run(self._session())
                backoff = 1.0
            except Exception as exc:
                self._note(f"gateway: {exc}")
                self._stop.wait(min(backoff, 60.0))
                backoff *= 2

    async def _session(self) -> None:
        import asyncio  # noqa: PLC0415

        import websockets  # noqa: PLC0415

        url = self._resume_url or GATEWAY_URL
        async with websockets.connect(url, max_size=8 * 1024 * 1024) as socket:
            self._socket = socket
            hello = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
            interval = float(hello.get("d", {}).get("heartbeat_interval", 41250)) / 1000.0

            if self._session_id and self._sequence is not None:
                await socket.send(json.dumps({
                    "op": OP_RESUME,
                    "d": {"token": self.token, "session_id": self._session_id, "seq": self._sequence},
                }))
            else:
                await socket.send(json.dumps({
                    "op": OP_IDENTIFY,
                    "d": {
                        "token": self.token,
                        "intents": self.intents,
                        "properties": {"os": "linux", "browser": "lai", "device": "lai"},
                    },
                }))

            heartbeat = asyncio.create_task(self._heartbeat_loop(socket, interval))
            try:
                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(socket.recv(), timeout=interval)
                    except asyncio.TimeoutError:
                        continue
                    self._on_frame(json.loads(raw))
            finally:
                heartbeat.cancel()
                self._socket = None

    async def _heartbeat_loop(self, socket, interval: float) -> None:
        import asyncio  # noqa: PLC0415

        # First beat is jittered as the gateway documentation requires.
        await asyncio.sleep(interval * 0.5)
        while not self._stop.is_set():
            try:
                await socket.send(json.dumps({"op": OP_HEARTBEAT, "d": self._sequence}))
            except Exception:
                return
            await asyncio.sleep(interval)

    def _on_frame(self, frame: dict) -> None:
        op = frame.get("op")
        if frame.get("s") is not None:
            self._sequence = frame["s"]

        if op == OP_DISPATCH:
            event = frame.get("t")
            data = frame.get("d") or {}
            if event == "READY":
                self.me = data.get("user", {})
                self._session_id = data.get("session_id", "")
                resume = data.get("resume_gateway_url", "")
                self._resume_url = f"{resume}/?v=10&encoding=json" if resume else ""
            elif event == "MESSAGE_CREATE":
                message = self._decode(data)
                if message is not None:
                    self._deliver(message)
        elif op == OP_INVALID_SESSION:
            # Cannot resume — drop the session so the next connect identifies.
            self._session_id, self._resume_url, self._sequence = "", "", None
        elif op == OP_RECONNECT:
            self._socket = None

    def _decode(self, data: dict) -> IncomingMessage | None:
        author = data.get("author") or {}
        if author.get("bot"):
            return None  # never react to bots, including ourselves
        text = (data.get("content") or "").strip()

        attachments: list[Attachment] = []
        for item in data.get("attachments") or []:
            url = item.get("url")
            if not url or int(item.get("size", 0)) > 20 * 1024 * 1024:
                continue
            try:
                response = httpx.get(url, timeout=60.0)
                response.raise_for_status()
            except httpx.HTTPError:
                continue
            content_type = item.get("content_type", "application/octet-stream")
            attachments.append(
                Attachment(
                    "image" if content_type.startswith("image/") else "file",
                    response.content,
                    item.get("filename", "file"),
                    content_type,
                )
            )

        if not text and not attachments:
            return None
        return IncomingMessage(
            channel=self.name,
            sender=str(author.get("id", "")),
            sender_name=author.get("global_name") or author.get("username", ""),
            chat=str(data.get("channel_id", "")),
            text=text,
            attachments=tuple(attachments),
            message_id=str(data.get("id", "")),
            raw=data,
        )

    # -- sending ---------------------------------------------------------

    def send(self, message: OutgoingMessage) -> str:
        if self._client is None:
            return ""
        last = ""
        if message.text:
            chunks = _split(message.text, MAX_MESSAGE_CHARS)
            if message.edit_id and chunks:
                last = self._edit(message.chat, message.edit_id, chunks[0])
                chunks = chunks[1:]
            for chunk in chunks:
                last = self._post(message.chat, chunk, reply_to=message.reply_to) or last
        for attachment in message.attachments:
            last = self._upload(message.chat, attachment) or last
        return last

    def _post(self, chat: str, text: str, *, reply_to: str = "") -> str:
        payload: dict = {"content": text[:MAX_MESSAGE_CHARS]}
        if reply_to:
            payload["message_reference"] = {"message_id": reply_to}
        return self._request("POST", f"/channels/{chat}/messages", json=payload)

    def _edit(self, chat: str, message_id: str, text: str) -> str:
        return self._request(
            "PATCH",
            f"/channels/{chat}/messages/{message_id}",
            json={"content": text[:MAX_MESSAGE_CHARS]},
        ) or message_id

    def _upload(self, chat: str, attachment: Attachment) -> str:
        if self._client is None or not attachment.data:
            return ""
        try:
            with self._send_lock:
                response = self._client.post(
                    f"/channels/{chat}/messages",
                    data={"payload_json": json.dumps({"content": attachment.caption[:1900]})},
                    files={"files[0]": (attachment.filename or "file", attachment.data, attachment.mime)},
                    headers={"content-type": None},  # let httpx set the multipart boundary
                    timeout=120.0,
                )
            if response.status_code >= 300:
                self._note(f"upload failed ({response.status_code}): {response.text[:150]}")
                return ""
            return str(response.json().get("id", ""))
        except Exception as exc:
            self._note(f"upload failed: {exc}")
            return ""

    def _request(self, method: str, path: str, **kwargs) -> str:
        if self._client is None:
            return ""
        try:
            with self._send_lock:
                response = self._client.request(method, path, **kwargs)
            if response.status_code == 429:
                retry_after = float(response.json().get("retry_after", 1.0))
                time.sleep(min(retry_after, 10.0))
                with self._send_lock:
                    response = self._client.request(method, path, **kwargs)
            if response.status_code >= 300:
                self._note(f"{method} {path} -> {response.status_code}: {response.text[:150]}")
                return ""
            return str(response.json().get("id", ""))
        except Exception as exc:
            self._note(f"{method} {path} failed: {exc}")
            return ""


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
