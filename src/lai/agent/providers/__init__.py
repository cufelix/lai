"""Model backends. One neutral message format, several wire protocols."""

from .anthropic_api import AnthropicProvider
from .base import (
    Block,
    ImageBlock,
    Message,
    Provider,
    StreamCallback,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    TurnResult,
    Usage,
)
from .openai_api import OpenAIProvider
from .registry import Credential, build_provider, discover_credentials

__all__ = [
    "AnthropicProvider",
    "Block",
    "Credential",
    "ImageBlock",
    "Message",
    "OpenAIProvider",
    "Provider",
    "StreamCallback",
    "TextBlock",
    "ThinkingBlock",
    "ToolCall",
    "ToolResultBlock",
    "TurnResult",
    "Usage",
    "build_provider",
    "discover_credentials",
]
