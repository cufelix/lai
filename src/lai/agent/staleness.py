"""References that stopped being true.

`ui_snapshot` hands back numbered elements, and every number belongs to the
snapshot it came from. Open a window, close one, switch focus, press a key that
opens a menu — and the numbers now point at something else, or at nothing.

The model has no way to know that from the outside. It sees a list of refs it
was given a moment ago, uses one, and the run spends a whole turn discovering
`element_not_found` before anybody learns anything. Twelve of those in this
machine's logs, each one a model call and about five seconds.

So: note when something moves the tree, and say so the next time a numbered
reference is about to be used. Once — a warning repeated on every call is
scrolled past, and the model has already been told.

Deliberately narrow:

* Only **numbered** references. `ui_click(name="Save")` survives a window
  moving, which is exactly why it is the better habit and why nagging about it
  would teach the wrong lesson.
* Only when a snapshot has actually been taken. A ref invented before any
  snapshot is not stale, it is imaginary, and the error it earns says so more
  clearly than a warning would.
* Only when the disturbance **succeeded**. An `app_open` that failed did not
  open anything.
"""

from __future__ import annotations

DISTURBS = frozenset({
    "app_open",
    "app_close",
    "window_focus",
    "window_close",
    "window_arrange",
    "window_to_workspace",
    "workspace_switch",
    # A keystroke can open a menu, close a dialog or move focus. It is the
    # least obvious member of this list and the one that catches people out.
    "computer_key",
})
"""Tools whose success means the accessibility tree is no longer what it was."""

REFRESHES = frozenset({"ui_snapshot", "desktop_observe"})
"""Tools that hand back a new set of numbers."""

TAKES_A_REF = frozenset({"ui_click", "ui_type", "ui_read", "ui_focus"})


class Staleness:
    """Whether the numbers the model is holding still mean anything."""

    __slots__ = ("_disturbed_by", "_snapshotted", "_told")

    def __init__(self) -> None:
        self._snapshotted = False
        self._disturbed_by = ""
        self._told = False

    def record(self, tool_name: str, *, ok: bool) -> None:
        """Note what a tool just did to the tree."""
        if not ok:
            return
        if tool_name in REFRESHES:
            self._snapshotted = True
            self._disturbed_by = ""
            self._told = False
        elif tool_name in DISTURBS:
            self._disturbed_by = tool_name

    def warning(self, tool_name: str, arguments: dict) -> str:
        """What to tell the model before it uses a number, or "" for nothing."""
        if not self._disturbed_by or not self._snapshotted or self._told:
            return ""
        if tool_name not in TAKES_A_REF or (arguments or {}).get("ref") is None:
            return ""
        self._told = True
        return (
            f"The window changed since your last ui_snapshot — {self._disturbed_by} "
            f"moved it — so the numbered references from that snapshot may no longer "
            f"point at what you think. Take a fresh ui_snapshot, or act by name "
            f"instead: a name survives a window moving and a number does not."
        )
