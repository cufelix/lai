"""Session state: transcript, persistence, resume and compaction."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .providers.base import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
)

# Screenshots dominate context. Keep only the newest few in full.
MAX_LIVE_IMAGES = 3
COMPACT_KEEP_RECENT = 8


@dataclass(slots=True)
class Session:
    """One conversation with the desktop."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str = ""
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    started_at: float = field(default_factory=time.time)
    path: Path | None = None
    metadata: dict = field(default_factory=dict)

    # -- transcript ------------------------------------------------------

    def append(self, message: Message) -> Message:
        self.messages.append(message)
        self._persist(message)
        return message

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def last_assistant_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant" and message.text.strip():
                return message.text.strip()
        return ""

    # -- context management ----------------------------------------------

    def prune_images(self, keep: int = MAX_LIVE_IMAGES) -> int:
        """Replace all but the newest ``keep`` screenshots with a placeholder.

        Old screenshots are almost never useful once the agent has moved on, and
        they are by far the most expensive thing in the transcript.
        """
        positions: list[tuple[int, int]] = []
        for m_index, message in enumerate(self.messages):
            for b_index, block in enumerate(message.content):
                if isinstance(block, ImageBlock):
                    positions.append((m_index, b_index))
                elif isinstance(block, ToolResultBlock) and block.images:
                    positions.append((m_index, b_index))

        removed = 0
        for m_index, b_index in positions[: max(0, len(positions) - keep)]:
            message = self.messages[m_index]
            block = message.content[b_index]
            if isinstance(block, ImageBlock):
                message.content[b_index] = TextBlock("[older screenshot removed to save context]")
                removed += 1
            elif isinstance(block, ToolResultBlock):
                message.content[b_index] = ToolResultBlock(
                    tool_use_id=block.tool_use_id,
                    content=block.content + "\n[screenshot removed to save context]",
                    images=(),
                    is_error=block.is_error,
                )
                removed += 1
        return removed

    def estimate_tokens(self) -> int:
        """Rough size of the transcript (chars/4, images ~1.1k each)."""
        total = 0
        for message in self.messages:
            for block in message.content:
                if isinstance(block, TextBlock):
                    total += len(block.text) // 4
                elif isinstance(block, ThinkingBlock):
                    total += len(block.thinking) // 4
                elif isinstance(block, ImageBlock):
                    total += 1100
                elif isinstance(block, ToolCall):
                    total += len(json.dumps(block.input)) // 4 + 12
                elif isinstance(block, ToolResultBlock):
                    total += len(block.content) // 4 + 1100 * len(block.images)
        return total

    def compact(self, summary: str, *, keep_recent: int = COMPACT_KEEP_RECENT) -> int:
        """Replace old turns with a summary, preserving tool-call pairing.

        Truncating mid-exchange breaks providers that require every ``tool_use``
        to have a matching ``tool_result``, so the cut point is moved back to a
        message that starts a clean exchange.
        """
        if len(self.messages) <= keep_recent + 1:
            return 0
        cut = len(self.messages) - keep_recent
        while cut > 0 and not _is_clean_boundary(self.messages, cut):
            cut -= 1
        if cut <= 0:
            return 0

        dropped = cut
        head = Message.user(
            "[Earlier steps were compacted to save context. Summary of what happened:]\n" + summary
        )
        self.messages = [head, *self.messages[cut:]]
        return dropped

    # -- persistence -----------------------------------------------------

    def bind(self, sessions_dir: Path) -> Path:
        sessions_dir = Path(sessions_dir)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        self.path = sessions_dir / f"{self.id}.jsonl"
        if not self.path.exists():
            self._write_line({
                "kind": "session_start",
                "id": self.id,
                "task": self.task,
                "at": self.started_at,
            })
        return self.path

    def _persist(self, message: Message) -> None:
        if self.path is None:
            return
        self._write_line({"kind": "message", **message.to_dict()})

    def _write_line(self, payload: dict) -> None:
        if self.path is None:
            return
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def summary(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "steps": self.steps,
            "messages": len(self.messages),
            "elapsed": round(self.elapsed, 1),
            "usage": self.usage.to_dict(),
            "estimated_tokens": self.estimate_tokens(),
            "path": str(self.path) if self.path else None,
        }

    @classmethod
    def load(cls, path: Path) -> Session:
        """Rebuild a session from its JSONL transcript.

        Images are not restored (they are not stored); text and tool structure are.
        """
        path = Path(path)
        session = cls(id=path.stem, path=path)
        if not path.is_file():
            return session
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("kind") == "session_start":
                session.task = payload.get("task", "")
                session.started_at = float(payload.get("at", time.time()))
            elif payload.get("kind") == "message":
                session.messages.append(_message_from_dict(payload))
        return session

    @classmethod
    def list_sessions(cls, sessions_dir: Path, *, limit: int = 20) -> list[dict]:
        directory = Path(sessions_dir)
        if not directory.is_dir():
            return []
        files = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for file in files[:limit]:
            task = _describe_transcript(file)
            out.append({
                "id": file.stem,
                "task": task,
                "modified": file.stat().st_mtime,
                "path": str(file),
            })
        return out


def _is_clean_boundary(messages: list[Message], index: int) -> bool:
    """True if slicing at ``index`` leaves no orphaned tool_result."""
    if index >= len(messages):
        return False
    message = messages[index]
    has_orphan_results = any(isinstance(b, ToolResultBlock) for b in message.content)
    return not has_orphan_results


def _message_from_dict(payload: dict) -> Message:
    blocks: list = []
    for raw in payload.get("content", []):
        kind = raw.get("type")
        if kind == "text":
            blocks.append(TextBlock(raw.get("text", "")))
        elif kind == "thinking":
            blocks.append(ThinkingBlock(raw.get("thinking", "")))
        elif kind == "tool_use":
            blocks.append(ToolCall(raw.get("id", ""), raw.get("name", ""), raw.get("input") or {}))
        elif kind == "tool_result":
            blocks.append(
                ToolResultBlock(
                    tool_use_id=raw.get("tool_use_id", ""),
                    content=raw.get("content", ""),
                    is_error=bool(raw.get("is_error")),
                )
            )
        elif kind == "image":
            blocks.append(TextBlock("[screenshot from an earlier session]"))
    return Message(payload.get("role", "user"), blocks)


# How far into a transcript to look for something to call it. The task is
# written at the top when it is known, but a session is bound before the first
# instruction arrives, so it often is not — and a list of sessions with no
# descriptions makes resuming by id guesswork.
_DESCRIBE_LINES = 30


def _describe_transcript(path: Path) -> str:
    """A one-line name for a session: its declared task, else what was first asked."""
    try:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= _DESCRIBE_LINES:
                    break
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("kind") == "session_start" and payload.get("task"):
                    return str(payload["task"])
                if payload.get("kind") == "message" and payload.get("role") == "user":
                    text = _first_text(payload)
                    if text:
                        return text
    except OSError:
        pass
    return ""


def _first_text(payload: dict) -> str:
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text") or "").strip()
            if text:
                return text
    return ""
