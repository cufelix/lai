"""Secret redaction for transcripts, logs and tool output.

An OS agent reads whatever is on screen. Anything that looks like a credential
gets masked before it reaches a log file or a model provider.
"""

from __future__ import annotations

import re

PLACEHOLDER = "«redacted»"

# Ordered: more specific patterns first so they win.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{32,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("google_key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("bearer", re.compile(r"(?i)\b(bearer|authorization:\s*bearer)\s+[A-Za-z0-9._\-]{16,}")),
    ("assigned_secret", re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|token|auth[_-]?token|access[_-]?key)"
        r"\s*[=:]\s*[\"']?([^\s\"',;]{8,})[\"']?"
    )),
    ("card_number", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
)


def redact(text: str, *, enabled: bool = True) -> str:
    """Mask anything that looks like a credential."""
    if not enabled or not text:
        return text
    out = text
    for name, pattern in PATTERNS:
        if name == "assigned_secret":
            out = pattern.sub(lambda m: f"{m.group(1)}={PLACEHOLDER}", out)
        elif name == "card_number":
            out = pattern.sub(lambda m: m.group(0) if not _luhn(m.group(0)) else PLACEHOLDER, out)
        else:
            out = pattern.sub(PLACEHOLDER, out)
    return out


def redact_obj(value, *, enabled: bool = True):
    """Recursively redact strings inside dicts/lists."""
    if not enabled:
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_obj(v, enabled=enabled) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        kind = type(value)
        return kind(redact_obj(v, enabled=enabled) for v in value)
    return value


def contains_secret(text: str) -> bool:
    return bool(text) and redact(text) != text


def _luhn(candidate: str) -> bool:
    """Only mask digit runs that actually validate as card numbers."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0
