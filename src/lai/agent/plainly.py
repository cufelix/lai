"""What the agent just did, in words anybody can read.

`ui_click {"ref": 114}` is precise, and it is precise for the person debugging
the tool call — not for the person watching their computer be used. Somebody
who wants to know whether it is doing the right thing needs "Clicked 3", and
they need it without learning what a ref is.

This is a translation, not a summary: every line still corresponds to exactly
one action, in the order it happened. The detailed form is always one toggle
away, because the moment something goes wrong the tool name is what matters.
"""

from __future__ import annotations

MAX_QUOTED = 60
"""Longer than a phrase and the sentence stops being readable."""


def describe(name: str, arguments: dict | None = None) -> str:
    """One plain sentence for one tool call."""
    args = arguments or {}
    handler = _HANDLERS.get(name)
    if handler is not None:
        return handler(args)
    if name.startswith("mcp__"):
        parts = name.split("__")
        service = parts[1].replace("-", " ") if len(parts) > 2 else "a service"
        action = parts[-1].replace("_", " ")
        return f"Used {service} to {action}"
    return name.replace("_", " ").capitalize()


def _quoted(value, fallback: str = "") -> str:
    text = str(value or "").strip().replace("\n", " ")
    if not text:
        return fallback
    if len(text) > MAX_QUOTED:
        text = text[: MAX_QUOTED - 1] + "…"
    return f"“{text}”"


def _target(args: dict) -> str:
    """Whatever identifies what was clicked, in order of how readable it is."""
    for key in ("name", "label", "text", "title"):
        if args.get(key):
            return _quoted(args[key])
    if args.get("ref") is not None:
        return "an item on screen"
    if args.get("x") is not None:
        return f"the screen at {args.get('x')},{args.get('y')}"
    return "something on screen"


_HANDLERS = {
    # looking
    "computer_screenshot": lambda a: "Looked at the screen",
    "ui_snapshot": lambda a: "Read what is on screen",
    "ui_find": lambda a: f"Looked for {_quoted(a.get('query'), 'something')}",
    "ui_read": lambda a: "Read some text on screen",
    "ocr_read": lambda a: "Read the words on screen",
    "ocr_find": lambda a: f"Searched the screen for {_quoted(a.get('text'), 'some text')}",
    "window_list": lambda a: "Checked which windows are open",
    "window_info": lambda a: "Checked a window",
    "observe": lambda a: "Looked around",
    "user_idle": lambda a: "Checked whether you are using the computer",
    "notifications_recent": lambda a: "Checked for notifications",
    "app_list": lambda a: "Looked through the installed applications",
    "desktop_observe": lambda a: "Looked around the desktop",
    "computer_cursor": lambda a: "Checked where the mouse is",
    "workspace_list": lambda a: "Checked the virtual desktops",

    # acting
    "app_open": lambda a: f"Opened {_quoted(a.get('name'), 'an application')}",
    "app_close": lambda a: "Closed an application",
    "ui_click": lambda a: f"Clicked {_target(a)}",
    "computer_click": lambda a: f"Clicked {_target(a)}",
    "computer_double_click": lambda a: f"Double-clicked {_target(a)}",
    "ui_type": lambda a: f"Typed {_quoted(a.get('text'), 'some text')}",
    "computer_type": lambda a: f"Typed {_quoted(a.get('text'), 'some text')}",
    "computer_key": lambda a: f"Pressed {a.get('key', 'a key')}",
    "computer_drag": lambda a: "Dragged the mouse",
    "computer_move": lambda a: "Moved the mouse",
    "computer_scroll": lambda a: "Scrolled",
    "window_focus": lambda a: "Switched to another window",
    "window_close": lambda a: "Closed a window",
    "ui_focus": lambda a: f"Focused {_target(a)}",
    "ui_wait_for": lambda a: f"Waited for {_quoted(a.get('query'), 'something')} to appear",
    "workspace_switch": lambda a: "Switched to another virtual desktop",
    "window_to_workspace": lambda a: "Moved a window to another virtual desktop",
    "window_arrange": lambda a: "Moved a window",
    "clipboard_read": lambda a: "Read the clipboard",
    "clipboard_write": lambda a: "Copied something to the clipboard",
    "desktop_wait": lambda a: (
        f"Waited {a['seconds']:g} seconds" if isinstance(a.get("seconds"), int | float)
        else "Waited for the screen to settle"
    ),

    # files and shell
    "file_read": lambda a: f"Read the file {_quoted(a.get('path'), 'a file')}",
    "file_write": lambda a: f"Saved {_quoted(a.get('path'), 'a file')}",
    "file_list": lambda a: "Looked through some files",
    "shell_exec": lambda a: f"Ran the command {_quoted(a.get('command'), 'something')}",

    # thinking about the job
    "plan_update": lambda a: "Made a plan",
    "task_complete": lambda a: "Finished",
    "task_blocked": lambda a: "Stopped — it could not finish",
    "delegate": lambda a: "Handed part of the job to a helper",
    "code_agent": lambda a: "Asked a coding assistant to write some code",
    "tool_find": lambda a: f"Looked for a tool that can {_quoted(a.get('query'), 'help')}",
    "skill_list": lambda a: "Looked through what it knows how to do",
    "skill_load": lambda a: f"Read up on {_quoted(a.get('name'), 'something it knows')}",
    "memory_save": lambda a: "Remembered something for next time",
    "memory_search": lambda a: "Checked what it remembers",
    "memory_forget": lambda a: "Forgot something it had remembered",
    "schedule_list": lambda a: "Checked what is scheduled",
    "schedule_remove": lambda a: "Cancelled something scheduled",
    "notify_user": lambda a: "Sent you a notification",
    "schedule_task": lambda a: "Scheduled something for later",
    "record_start": lambda a: "Started recording the screen",
    "record_stop": lambda a: "Stopped recording",
}
