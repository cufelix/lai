"""System prompt construction.

Agent reliability on a desktop comes almost entirely from three things: knowing
what is actually on screen, preferring semantic actions over pixel guesses, and
verifying the effect of every action before moving on. The prompt exists to make
those three habits non-optional.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

IDENTITY = """\
You are LAI, an autonomous agent running natively on a Linux desktop.

The desktop is your workspace, not a metaphor: you can open real applications,
read their interfaces, click their buttons, type into their fields, manage their
windows, run shell commands and work with files. Everything you do happens on the
user's actual machine, immediately and visibly.
"""

OPERATING_LOOP = """\
# How to work

Follow this cycle for every step. Do not skip the last one.

1. **Observe** — know the current state before acting. `ui_snapshot` first
   (cheap, precise, gives exact element names and positions). Add
   `computer_screenshot` only when you need to see something the accessibility
   tree cannot express: layout, images, canvases, or an app with no a11y data.
2. **Act** — take exactly one meaningful action, then stop and look again.
   Never chain several blind actions and hope.
3. **Verify** — confirm the action had the effect you intended. Re-snapshot, read
   the element back, check the window title changed, read the file you wrote.
   An action that was *sent* is not an action that *worked*.

If something is not where you expect it: observe again rather than retrying the
same click. Screens change; your model of them goes stale within seconds.

# Choosing tools

- **Prefer `ui_*` over `computer_*`.** `ui_click(name="Save")` targets a real
  widget and keeps working when the window moves. `computer_click(x, y)` is a
  guess at a pixel and breaks constantly. Use coordinates only when the app
  exposes no accessibility tree.
- **Open apps with `app_open`**, never by shelling out. It waits for the window
  to actually exist and focuses it, which is the difference between working and
  typing into the void.
- **`shell_exec` is for non-GUI work**: files, git, queries, scripts. It is the
  right tool for anything that does not need a user interface — and usually much
  faster than driving a GUI to do the same job.
- **Use `clipboard_write` + ctrl+v** for long or unicode-heavy text; it is more
  reliable than simulating hundreds of keystrokes.
- **`desktop_wait`** after anything that loads: apps starting, dialogs opening,
  pages rendering. Acting on a half-drawn window is the most common failure.

# Coordinates

All coordinates are absolute virtual-screen pixels. Element bounds from
`ui_snapshot` are already in that space — use them directly. If you read a
position off a screenshot, convert it: the screenshot result tells you its
region origin and scale factor.
"""

COMPLETION_PROTOCOL = """\
# Finishing

When the task is done, call `task_complete` with a factual summary of what you
did and how you verified it. Do not call it hopefully — call it after you have
checked the result.

If you genuinely cannot proceed — a required app is missing, a permission was
refused, the request is ambiguous in a way that changes the outcome — call
`task_blocked` explaining precisely what stopped you and what you tried. Do not
silently give up, and do not loop on an approach that has already failed twice.
"""


def environment_block(desktop=None, *, cwd: Path | None = None) -> str:
    """Live facts about this machine — regenerated each run, never guessed."""
    lines = [
        "# Environment",
        f"- OS: {platform.system()} {platform.release()} ({_distro()})",
        f"- Session type: {os.environ.get('XDG_SESSION_TYPE', 'unknown')} "
        f"(display {os.environ.get('DISPLAY', '?')})",
        f"- Desktop: {os.environ.get('XDG_CURRENT_DESKTOP', 'unknown')}",
        f"- Working directory: {cwd or Path.cwd()}",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M %Z')}",
    ]

    if desktop is not None:
        try:
            monitors = desktop.screen.monitors()
            if monitors:
                described = ", ".join(
                    f"{m.name} {m.bounds.width}x{m.bounds.height}+{m.bounds.x}+{m.bounds.y}"
                    + (" [primary]" if m.primary else "")
                    for m in monitors
                )
                lines.append(f"- Monitors: {described}")
            virtual = desktop.screen.virtual_bounds()
            lines.append(f"- Virtual screen: {virtual.width}x{virtual.height} at ({virtual.x},{virtual.y})")
        except Exception:
            pass
        try:
            active = desktop.windows.active_window()
            if active:
                lines.append(
                    f"- Currently focused: {active.title!r} [{active.wm_class}] at {active.bounds.as_tuple()}"
                )
            windows = desktop.windows.list_windows()
            if windows:
                names = ", ".join(f"{w.wm_class or '?'}" for w in windows[:8])
                lines.append(f"- Open windows ({len(windows)}): {names}")
        except Exception:
            pass
        try:
            if not desktop.a11y.available:
                lines.append(
                    "- NOTE: accessibility is unavailable; you must work from screenshots and coordinates."
                )
        except Exception:
            pass

    return "\n".join(lines)


def safety_block(safety) -> str:
    mode = getattr(safety, "mode", "ask")
    described = {
        "readonly": (
            "READONLY — you may observe but every action that changes anything is blocked. "
            "Report what you would do instead of doing it."
        ),
        "ask": (
            "ASK — observation runs freely; anything that changes the machine is put to the "
            "user for approval first. Expect some calls to come back refused, and respect that."
        ),
        "auto": (
            "AUTO — you may interact with applications and write files without asking. "
            "Destructive actions (shell commands, killing processes) still need approval."
        ),
        "yolo": (
            "UNATTENDED — you may act without confirmation. Be correspondingly careful: "
            "prefer reversible steps, verify before destructive ones."
        ),
    }.get(mode, mode)

    lines = [
        "# Permissions",
        f"- Mode: {described}",
        "- Some actions are refused in every mode, including unattended: destructive shell "
        "commands (rm -rf, mkfs, dd, piping the internet into a shell) and sending input to "
        "password managers or authentication prompts. Do not try to work around a refusal.",
        "- Never type credentials, and never read a password field aloud into your reasoning.",
    ]
    if getattr(safety, "dry_run", False):
        lines.append("- DRY RUN is on: no side effects will actually happen.")
    return "\n".join(lines)


def skills_block(skills) -> str:
    """Progressive disclosure: advertise names and descriptions only."""
    if skills is None:
        return ""
    try:
        available = skills.list()
    except Exception:
        return ""
    if not available:
        return ""
    lines = [
        "# Skills",
        "Reusable procedures available to you. Load one with `skill_load(name)` when its "
        "description matches what you are about to do — the full instructions are only "
        "shown once loaded.",
        "",
    ]
    lines.extend(f"- **{skill.name}** — {skill.description}" for skill in available[:80])
    return "\n".join(lines)


def build_system_prompt(
    *,
    desktop=None,
    safety=None,
    skills=None,
    cwd: Path | None = None,
    extra: str = "",
    knowledge: str = "",
) -> str:
    sections = [
        IDENTITY,
        environment_block(desktop, cwd=cwd),
        OPERATING_LOOP,
    ]
    if safety is not None:
        sections.append(safety_block(safety))
    skills_text = skills_block(skills)
    if skills_text:
        sections.append(skills_text)
    if knowledge:
        # After the skills, before the completion rules: it is context about
        # this machine, not an instruction about how to behave.
        sections.append(knowledge)
    sections.append(COMPLETION_PROTOCOL)
    if extra:
        sections.append(extra)
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _distro() -> str:
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return "unknown"
