"""Writing ``config.toml``.

Reading is in :mod:`lai.config`; writing lives here because it is a different
job with a different priority. A generated config is documentation as much as
data — someone will open this file to learn what LAI can be told, so it is
written with comments, in a stable order, and only for the settings that were
actually chosen. Defaults stay absent so an upgrade can change them.

The file may hold an API key, so it is written ``0600`` via a temp file in the
same directory and an atomic rename: a half-written config must never be
readable, and an interrupted write must never destroy the previous one.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

CONFIG_FILENAME = "config.toml"

HEADER = """\
# LAI configuration — written by `lai setup`. See `lai --help`.
#
# Everything here is optional; delete a line to go back to the default.
# Environment variables win over this file, so LAI_MODE=yolo overrides
# safety.mode below for one run.
"""

SECTION_NOTES = {
    "provider": "# Which model does the thinking. `name = \"auto\"` picks the best key it finds.",
    "safety": "# readonly = look but do not touch · ask = confirm before changes\n"
              "# auto = act, but confirm shell and kill · yolo = no prompts",
    "limits": "# Budgets for one run. The agent stops and reports rather than running away.",
    "channels": "# Remote control. `lai channels` manages who is allowed in.",
    "desktop": "# Perception tuning. max_edge caps screenshot size sent to the model.",
    "learning": "# Notes the agent keeps about this machine, in ~/.lai/notes.\n"
                "# `enabled = false` stops it reading them; `reflect = false` stops it writing.",
}


def config_path(home: Path) -> Path:
    return Path(home) / CONFIG_FILENAME


def render(settings: dict) -> str:
    """Render a nested ``{section: {key: value}}`` mapping as annotated TOML."""
    lines = [HEADER]
    for section in ("provider", "safety", "learning", "limits", "desktop", "channels"):
        values = settings.get(section) or {}
        # A nested dict is a sub-table ([channels.telegram]) and is emitted
        # below; rendering it here would produce a scalar with the same name
        # and TOML would reject the file as overwriting a value.
        values = {
            k: v for k, v in values.items()
            if v not in (None, "") and not isinstance(v, dict)
        }
        if not values:
            continue
        note = SECTION_NOTES.get(section)
        if note:
            lines.append(note)
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")

    # Nested tables (channels.telegram and friends) come last so the flat keys
    # above are not swallowed by the sub-table they would follow.
    for section, subsection in (("channels", "telegram"), ("channels", "discord"), ("channels", "webhook")):
        values = (settings.get(section) or {}).get(subsection) or {}
        values = {k: v for k, v in values.items() if v not in (None, "")}
        if not values:
            continue
        lines.append(f"[{section}.{subsection}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write(home: Path, settings: dict) -> Path:
    """Write config.toml atomically with owner-only permissions."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    target = config_path(home)
    body = render(settings)

    handle, temporary = tempfile.mkstemp(dir=str(home), prefix=".config-", suffix=".toml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def merge(existing: dict, updates: dict) -> dict:
    """Deep-merge ``updates`` over ``existing`` without mutating either."""
    result = {k: dict(v) if isinstance(v, dict) else v for k, v in existing.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def read(home: Path) -> dict:
    """Read the current file back as a plain dict; {} when absent or broken."""
    path = config_path(home)
    if not path.is_file():
        return {}
    try:
        import tomllib  # noqa: PLC0415

        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


__all__ = ["CONFIG_FILENAME", "config_path", "merge", "read", "render", "write"]
