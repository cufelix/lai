"""Tool calls a model wrote out as text.

Native function calling is an API feature, and plenty of good models do not
have it — Hermes, Qwen, most things served straight off Ollama or vLLM. They
were trained to write the call instead:

    <tool_call>
    {"name": "window_list", "arguments": {}}
    </tool_call>

An agent that only understands the OpenAI ``tool_calls`` field reads that as
the assistant chatting about a tool it never called, and the run stalls with
the model insisting it already did the thing. Reading both forms costs one
regular expression and makes a whole class of models usable.

The formats here are the ones models actually emit. Anything looser — a bare
JSON object in the prose, a fenced code block — is left alone on purpose: a
model explaining what it *would* call must not be taken as calling it.
"""

from __future__ import annotations

import json
import re

from .base import ToolCall

TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
"""Hermes, Qwen, and most fine-tunes that copied them."""

MISTRAL = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)
"""Mistral writes one JSON array after a marker."""

PYTHON_TAG = re.compile(r"<\|python_tag\|>\s*(\{.*?\})\s*(?:<\|eom_id\|>|$)", re.DOTALL)
"""Llama 3.1's tagged form."""


def parse(text: str, *, start: int = 0) -> tuple[str, list[ToolCall]]:
    """(text with the calls removed, the calls).

    ``start`` numbers the generated ids, so a turn that already carries native
    calls does not end up with two ``call_0``.
    """
    markers = ("<tool_call", "[TOOL_CALLS]", "<|python_tag|>")
    if not text or not any(marker.lower() in text.lower() for marker in markers):
        return text, []

    found: list[ToolCall] = []
    cleaned = text

    for pattern in (TAGGED, PYTHON_TAG, MISTRAL):
        for match in pattern.finditer(cleaned):
            for call in _calls_in(match.group(1)):
                call_id = f"call_{start + len(found)}"
                found.append(ToolCall(id=call_id, name=call[0], input=call[1]))
        if found:
            cleaned = pattern.sub("", cleaned)
            break

    return cleaned.strip(), found


def _calls_in(blob: str) -> list[tuple[str, dict]]:
    """One block can hold a call or a list of them."""
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return []
    entries = payload if isinstance(payload, list) else [payload]
    out: list[tuple[str, dict]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # Some fine-tunes nest it the way the API response does.
        function = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments", function.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        out.append((name, dict(arguments) if isinstance(arguments, dict) else {}))
    return out


def describe(tools: list[dict]) -> str:
    """The system-prompt block that asks for calls in the tagged form.

    Written the way Hermes was trained to read it: the schemas inside a
    ``<tools>`` element, and one worked example of the reply shape. Models that
    learned this format follow it closely; models that did not still manage,
    because the instruction is explicit.
    """
    if not tools:
        return ""
    lines = [
        "# Calling tools",
        "You do not have native function calling here, so write each call as a",
        "JSON object inside a `<tool_call>` element. These are the tools:",
        "",
        "<tools>",
    ]
    lines.extend(json.dumps(tool, ensure_ascii=False) for tool in tools)
    lines.append("</tools>")
    lines += [
        "",
        "To call one, reply with nothing but the element:",
        "",
        "<tool_call>",
        '{"name": "window_list", "arguments": {}}',
        "</tool_call>",
        "",
        "Several may be written one after another. Do not describe a call you",
        "have not written — a call you only talked about did not happen.",
    ]
    return "\n".join(lines)
