"""Channel manager — turns incoming messages into agent runs.

Owns the parts a transport should not know about: who is allowed in, which
conversation maps to which agent session, how progress is reported back, and
how a permission prompt reaches a human who is not sitting at the machine.

Two decisions worth stating:

* **One run per conversation at a time.** Two agents fighting over the same
  mouse is not a feature. A second request while busy is refused with a hint,
  not silently queued behind an unbounded backlog.
* **Approvals travel over the channel.** In ``ask`` mode the agent's permission
  prompt is delivered to the chat and the run blocks on a reply, so remote
  operation does not force you into ``yolo``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..agent.session import Session
from .access import Access, AccessPolicy
from .base import Channel, IncomingMessage, OutgoingMessage

APPROVAL_TIMEOUT = 180.0
PROGRESS_INTERVAL = 2.0
MAX_RESULT_CHARS = 3500

HELP = """\
LAI — desktop agent

Send any instruction in plain language and I will carry it out on the desktop.

/status      what I am doing and how I am configured
/stop        interrupt the current task
/new         start a fresh conversation
/screenshot  send a picture of the screen
/windows     list open windows
/mode M      permission mode: readonly | ask | auto | yolo (admin)
/whoami      your identity here
/help        this message

In `ask` mode I will message you before anything that changes the machine;
reply /yes or /no."""


@dataclass(slots=True)
class Conversation:
    """Per-chat state."""

    route: str
    chat: str
    channel: str
    session: Session = field(default_factory=Session)
    lock: threading.Lock = field(default_factory=threading.Lock)
    agent: object | None = None
    busy: bool = False
    started_at: float = 0.0
    status_message_id: str = ""
    last_progress: float = 0.0
    pending_approval: PendingApproval | None = None
    runs: int = 0


@dataclass(slots=True)
class PendingApproval:
    tool: str
    reason: str
    event: threading.Event = field(default_factory=threading.Event)
    granted: bool = False


class ChannelManager:
    """Wires channels to the agent runtime."""

    def __init__(
        self,
        runtime,
        *,
        access: AccessPolicy | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        approval_timeout: float = APPROVAL_TIMEOUT,
        desktop=None,
    ) -> None:
        self.runtime = runtime
        self.access = access or AccessPolicy()
        self.on_event = on_event
        self.approval_timeout = approval_timeout
        # Optional gate owning the desktop (claim/engage/release). The daemon
        # passes its state so HTTP tasks, channel runs and the scheduler
        # serialise against each other instead of racing for the mouse.
        self.desktop = desktop
        self.channels: dict[str, Channel] = {}
        self.conversations: dict[str, Conversation] = {}
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []

    # -- lifecycle -------------------------------------------------------

    def add(self, channel: Channel) -> Channel:
        self.channels[channel.name] = channel
        return channel

    def start(self) -> list[str]:
        """Start every configured channel; returns the names that came up."""
        started: list[str] = []
        for name, channel in self.channels.items():
            if not channel.available:
                continue
            try:
                channel.start(self.handle)
                started.append(name)
            except Exception as exc:  # one bad channel must not stop the rest
                self._emit("channel_error", {"channel": name, "error": str(exc)})
        return started

    def stop(self) -> None:
        for channel in self.channels.values():
            try:
                channel.stop()
            except Exception:
                continue
        for conversation in list(self.conversations.values()):
            pending = conversation.pending_approval
            if pending is not None:
                pending.granted = False
                pending.event.set()
            agent = conversation.agent
            if agent is not None:
                try:
                    agent.interrupt()
                except Exception:
                    continue
        for worker in list(self._workers):
            if worker.is_alive():
                worker.join(timeout=3.0)

    def status(self) -> dict:
        return {
            "channels": [c.status() for c in self.channels.values() if hasattr(c, "status")],
            "conversations": [
                {"route": c.route, "busy": c.busy, "runs": c.runs, "messages": len(c.session.messages)}
                for c in self.conversations.values()
            ],
            "access": self.access.summary(),
        }

    # -- inbound ---------------------------------------------------------

    def handle(self, message: IncomingMessage) -> None:
        """Entry point every channel calls. Must never raise."""
        try:
            self._handle(message)
        except Exception as exc:
            self._emit("handler_error", {"route": message.route, "error": str(exc)})
            self._reply(message, f"Something went wrong handling that: {exc}")

    def _handle(self, message: IncomingMessage) -> None:
        decision = self.access.check(message)

        if decision.access is Access.PAIRING:
            _, code = message.command
            outcome = self.access.redeem(
                code, message.channel, message.sender, name=message.sender_name
            )
            self._emit(
                "pairing", {"route": message.route, "granted": outcome.allowed, "reason": outcome.reason}
            )
            self._reply(
                message,
                "Paired. You can now send me instructions — try /help."
                if outcome.allowed
                else f"Pairing failed: {outcome.reason}",
            )
            return

        if not decision.allowed:
            self._emit("denied", {"route": message.route, "sender": message.sender})
            # Deliberately uninformative: do not advertise that pairing exists.
            self._reply(message, "Not authorised.")
            return

        conversation = self._conversation(message)

        if message.is_command:
            self._command(message, conversation, admin=decision.admin)
            return

        if not message.text.strip():
            self._reply(message, "Send me an instruction, or /help.")
            return

        if conversation.busy:
            self._reply(
                message,
                f"Still working on the previous task ({int(time.time() - conversation.started_at)}s). "
                "Send /stop to interrupt it.",
            )
            return
        if self._busy_elsewhere(conversation):
            self._reply(
                message,
                "Another task is already running on this desktop — from another chat "
                "or interface. Wait for it to finish (or /stop it there).",
            )
            return

        worker = threading.Thread(
            target=self._run_task,
            args=(message, conversation),
            name=f"lai-chan-{conversation.route}",
            daemon=True,
        )
        with self._lock:
            self._workers = [w for w in self._workers if w.is_alive()]
            self._workers.append(worker)
        worker.start()

    def _busy_elsewhere(self, conversation: Conversation) -> bool:
        """Is any *other* conversation mid-run? (One agent per desktop.)"""
        with self._lock:
            return any(c.busy for c in self.conversations.values() if c is not conversation)

    def _conversation(self, message: IncomingMessage) -> Conversation:
        with self._lock:
            conversation = self.conversations.get(message.route)
            if conversation is None:
                session = Session()
                try:
                    session.bind(self.runtime.config.sessions_dir)
                except Exception:
                    pass
                conversation = Conversation(
                    route=message.route,
                    chat=message.chat,
                    channel=message.channel,
                    session=session,
                )
                self.conversations[message.route] = conversation
            return conversation

    # -- commands --------------------------------------------------------

    def _command(self, message: IncomingMessage, conversation: Conversation, *, admin: bool) -> None:
        name, argument = message.command

        if name in ("start", "help", "h"):
            self._reply(message, HELP)
        elif name == "whoami":
            principal = self.access.get(message.channel, message.sender)
            self._reply(
                message,
                f"{message.sender_name or 'you'} — id {message.sender} on {message.channel}\n"
                f"role: {'admin' if principal and principal.admin else 'user'}",
            )
        elif name == "status":
            self._reply(message, self._status_text(conversation))
        elif name == "stop":
            agent = conversation.agent
            if agent is None or not conversation.busy:
                self._reply(message, "Nothing is running.")
            else:
                # An approval wait would otherwise sit out its full timeout.
                pending = conversation.pending_approval
                if pending is not None:
                    pending.granted = False
                    pending.event.set()
                agent.interrupt()
                self._reply(message, "Stopping after the current step.")
        elif name == "new":
            conversation.session = Session()
            try:
                conversation.session.bind(self.runtime.config.sessions_dir)
            except Exception:
                pass
            self._reply(message, "Fresh conversation started.")
        elif name in ("yes", "y", "no", "n", "approve", "deny"):
            self._resolve_approval(message, conversation, granted=name in ("yes", "y", "approve"))
        elif name == "mode":
            self._set_mode(message, argument, admin=admin)
        elif name in ("screenshot", "screen", "shot"):
            self._send_screenshot(message)
        elif name == "windows":
            self._reply(message, self._windows_text())
        elif name == "skills":
            skills = self.runtime.skills.list()
            listing = "\n".join(f"- {s.name}: {s.description[:70]}" for s in skills[:30])
            self._reply(message, f"{len(skills)} skill(s):\n{listing}" if skills else "No skills installed.")
        elif name == "allow" and admin:
            self._reply(message, self._allow(argument, message.channel))
        elif name == "revoke" and admin:
            target = argument.strip()
            ok = self.access.revoke(message.channel, target)
            self._reply(message, f"Revoked {target}." if ok else f"{target} was not on the list.")
        elif name == "who" and admin:
            people = self.access.principals()
            self._reply(
                message,
                "\n".join(
                    f"- {p.channel}:{p.sender} {p.name}{' (admin)' if p.admin else ''}" for p in people
                )
                or "Nobody is allowlisted.",
            )
        else:
            self._reply(message, f"Unknown command /{name}. Try /help.")

    def _allow(self, argument: str, channel: str) -> str:
        target = argument.strip()
        if not target:
            return "Usage: /allow <sender-id>"
        self.access.allow(channel, target)
        return f"Allowed {channel}:{target}."

    def _set_mode(self, message: IncomingMessage, argument: str, *, admin: bool) -> None:
        from ..config import PERMISSION_MODES  # noqa: PLC0415

        mode = argument.strip().lower()
        if not mode:
            self._reply(message, f"Current mode: {self.runtime.config.safety.mode}")
            return
        if not admin:
            self._reply(message, "Only an admin can change the permission mode.")
            return
        if mode not in PERMISSION_MODES:
            self._reply(message, f"Mode must be one of: {', '.join(PERMISSION_MODES)}")
            return
        from dataclasses import replace  # noqa: PLC0415

        self.runtime.config = self.runtime.config.with_overrides(
            safety=replace(self.runtime.config.safety, mode=mode)
        )
        self.runtime.policy.config = self.runtime.config.safety
        self._emit("mode_changed", {"mode": mode, "by": message.sender})
        self._reply(message, f"Permission mode is now {mode}.")

    def _send_screenshot(self, message: IncomingMessage) -> None:
        try:
            shot = self.runtime.desktop.screen.grab()
        except Exception as exc:
            self._reply(message, f"Could not capture the screen: {exc}")
            return
        window = None
        try:
            window = self.runtime.desktop.windows.active_window()
        except Exception:
            pass
        caption = f"Focused: {window.title}" if window else "Desktop"
        self._send(message.channel, OutgoingMessage.image(message.chat, shot.png, caption=caption[:200]))

    def _windows_text(self) -> str:
        try:
            windows = self.runtime.desktop.windows.list_windows()
        except Exception as exc:
            return f"Could not read the window list: {exc}"
        if not windows:
            return "No windows open."
        return "\n".join(
            f"{'*' if w.active else '-'} {w.wm_class or '?'}: {w.title[:60]}" for w in windows
        )

    def _status_text(self, conversation: Conversation) -> str:
        provider = self.runtime.provider
        lines = [
            f"model    : {provider.name}/{provider.model}" if provider else "model    : none",
            f"mode     : {self.runtime.config.safety.mode}",
            f"tools    : {len(self.runtime.registry)}",
            f"skills   : {len(self.runtime.skills)}",
            f"runs here: {conversation.runs}",
        ]
        if conversation.busy:
            lines.append(f"busy     : {int(time.time() - conversation.started_at)}s so far")
        else:
            lines.append("busy     : idle")
        return "\n".join(lines)

    # -- running a task --------------------------------------------------

    def _run_task(self, message: IncomingMessage, conversation: Conversation) -> None:
        claim = getattr(self.desktop, "claim", None)
        if claim is not None and not claim(f"[{conversation.route}] {message.text[:60]}"):
            self._reply(
                message,
                "Another task is already running on this desktop — from another "
                "interface. Wait for it to finish (or /stop it there).",
            )
            return
        with conversation.lock:
            conversation.busy = True
            conversation.started_at = time.time()
            conversation.runs += 1
            conversation.last_progress = 0.0
            try:
                status_id = self._send(
                    message.channel,
                    OutgoingMessage(chat=message.chat, text="Working…", reply_to=message.message_id),
                )
                conversation.status_message_id = status_id

                agent = self.runtime.agent(
                    session=conversation.session,
                    approver=self._approver(message, conversation),
                    on_event=self._progress(message, conversation),
                )
                conversation.agent = agent
                engage = getattr(self.desktop, "engage", None)
                if engage is not None:
                    engage(agent)
                self._emit("run_started", {"route": conversation.route, "task": message.text, "agent": agent})
                result = agent.run(message.text)
                self._emit("run_finished", {"route": conversation.route, **result.to_dict()})
                self._send_result(message, conversation, result)
            except Exception as exc:
                self._reply(message, f"The task failed: {type(exc).__name__}: {exc}")
                self._emit("run_error", {"route": conversation.route, "error": str(exc)})
            finally:
                conversation.busy = False
                conversation.agent = None
                conversation.pending_approval = None
                if getattr(self.desktop, "release", None) is not None:
                    self.desktop.release()

    def _send_result(self, message: IncomingMessage, conversation: Conversation, result) -> None:
        icon = {
            "completed": "✅", "blocked": "⛔", "budget_exceeded": "⏱",
            "error": "❌", "interrupted": "⏹",
        }.get(result.status, "•")
        parts = [f"{icon} {result.status.replace('_', ' ')} — {result.steps} steps, {result.elapsed:.0f}s"]
        if result.summary:
            parts.append(result.summary)
        if result.verification:
            parts.append(f"Verified: {result.verification}")
        if result.artifacts:
            parts.append("Files: " + ", ".join(result.artifacts))
        if result.error:
            parts.append(f"Error: {result.error}")
        text = "\n\n".join(parts)[:MAX_RESULT_CHARS]

        self._send(
            message.channel,
            OutgoingMessage(chat=message.chat, text=text, edit_id=conversation.status_message_id),
        )

    def _progress(self, message: IncomingMessage, conversation: Conversation):
        """Throttled live progress into the status message."""
        state = {"step": 0, "tool": "", "of": 0}

        def report(kind: str, payload: dict) -> None:
            self._emit(kind, {"route": conversation.route, **payload})
            if kind == "step":
                state["step"] = payload.get("step", 0)
                state["of"] = payload.get("of", 0)
            elif kind == "tool_call":
                state["tool"] = payload.get("name", "")
            else:
                return

            now = time.monotonic()
            if now - conversation.last_progress < PROGRESS_INTERVAL:
                return
            conversation.last_progress = now
            if not conversation.status_message_id:
                return
            text = f"Working… step {state['step']}/{state['of']}"
            if state["tool"]:
                text += f" · {state['tool']}"
            self._send(
                message.channel,
                OutgoingMessage(
                    chat=message.chat, text=text, edit_id=conversation.status_message_id, silent=True
                ),
            )

        return report

    # -- approvals -------------------------------------------------------

    def _approver(self, message: IncomingMessage, conversation: Conversation):
        """Ask the human over the channel and block until they answer."""

        def approve(tool: str, tool_input: dict, verdict) -> bool:
            pending = PendingApproval(tool=tool, reason=verdict.reason)
            conversation.pending_approval = pending
            preview = ", ".join(f"{k}={v!r}" for k, v in list(tool_input.items())[:4])[:300]
            self._send(
                message.channel,
                OutgoingMessage(
                    chat=message.chat,
                    text=(
                        f"⚠️ Approval needed\n\n{tool}\n{preview}\n\n"
                        f"{verdict.reason}\n\nReply /yes to allow or /no to refuse "
                        f"(expires in {int(self.approval_timeout)}s)."
                    ),
                ),
            )
            granted = pending.event.wait(timeout=self.approval_timeout)
            conversation.pending_approval = None
            if not granted:
                self._reply(message, "No answer — treating that as a refusal.")
                return False
            return pending.granted

        return approve

    def _resolve_approval(self, message: IncomingMessage, conversation: Conversation, *, granted: bool) -> None:
        pending = conversation.pending_approval
        if pending is None:
            self._reply(message, "Nothing is waiting for approval.")
            return
        pending.granted = granted
        pending.event.set()
        self._reply(message, "Allowed." if granted else "Refused.")

    # -- outbound --------------------------------------------------------

    def _send(self, channel_name: str, message: OutgoingMessage) -> str:
        channel = self.channels.get(channel_name)
        if channel is None:
            return ""
        try:
            return channel.send(message)
        except Exception as exc:
            self._emit("send_error", {"channel": channel_name, "error": str(exc)})
            return ""

    def _reply(self, message: IncomingMessage, text: str) -> str:
        return self._send(
            message.channel, OutgoingMessage(chat=message.chat, text=text, reply_to=message.message_id)
        )

    def broadcast(self, text: str, *, images: tuple[bytes, ...] = ()) -> int:
        """Push a message to every known conversation — used by scheduled runs."""
        sent = 0
        for conversation in list(self.conversations.values()):
            outgoing = OutgoingMessage(chat=conversation.chat, text=text)
            if images:
                outgoing = OutgoingMessage.image(conversation.chat, images[0], caption=text[:200])
            if self._send(conversation.channel, outgoing):
                sent += 1
        return sent

    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, payload)
        except Exception:
            pass
