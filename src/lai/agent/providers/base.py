"""Provider-neutral message types.

The loop speaks these types only; each provider translates to and from its own
wire format. Adding a model vendor means adding one file here, not touching the
loop.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True, slots=True)
class ImageBlock:
    data: bytes
    media_type: str = "image/png"
    type: str = "image"

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    thinking: str
    signature: str = ""
    type: str = "thinking"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: dict = field(default_factory=dict)
    type: str = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str = ""
    images: tuple[bytes, ...] = ()
    is_error: bool = False
    type: str = "tool_result"


Block = TextBlock | ImageBlock | ThinkingBlock | ToolCall | ToolResultBlock


@dataclass(slots=True)
class Message:
    role: str
    """user | assistant"""
    content: list[Block] = field(default_factory=list)

    @classmethod
    def user(cls, text: str, *, images: Iterable[bytes] = ()) -> Message:
        blocks: list[Block] = [TextBlock(text)] if text else []
        blocks += [ImageBlock(data) for data in images]
        return cls("user", blocks)

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls("assistant", [TextBlock(text)])

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]

    def to_dict(self) -> dict:
        return {"role": self.role, "content": [_block_to_dict(b) for b in self.content]}


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass(slots=True)
class TurnResult:
    """One assistant turn."""

    message: Message
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: Any = None

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


StreamCallback = Callable[[str, str], None]
"""(event_kind, payload) — kinds: 'text', 'thinking', 'tool', 'status'."""


class Provider(Protocol):
    """What the loop needs from a model backend."""

    name: str
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[dict] | None = None,
        stream: StreamCallback | None = None,
    ) -> TurnResult: ...

    def close(self) -> None: ...


def _block_to_dict(block: Block) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {"type": "image", "media_type": block.media_type, "bytes": len(block.data)}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}
    if isinstance(block, ToolCall):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
            "images": len(block.images),
        }
    return {"type": "unknown"}
