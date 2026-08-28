"""Mouse and keyboard actuation.

Backed by ``xdotool`` (XTEST under the hood): battle-tested, unicode-safe typing
and correct modifier handling, which hand-rolled XTEST key mapping rarely gets
right. Every call is a short-lived subprocess with a hard timeout, so a wedged
X server can never hang the agent loop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass

from ..errors import BackendUnavailable, InvalidToolInput
from .geometry import Point

DEFAULT_TIMEOUT = 10.0
MOVE_VERIFY_TIMEOUT = 0.35
MOVE_VERIFY_POLL = 0.01

BUTTONS: dict[str, int] = {
    "left": 1,
    "middle": 2,
    "right": 3,
    "wheel_up": 4,
    "wheel_down": 5,
    "wheel_left": 6,
    "wheel_right": 7,
}

# Friendly names the model is likely to emit -> X keysym names xdotool expects.
KEY_ALIASES: dict[str, str] = {
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "space": "space",
    "backspace": "BackSpace",
    "delete": "Delete",
    "del": "Delete",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "pageup": "Prior",
    "page_up": "Prior",
    "pagedown": "Next",
    "page_down": "Next",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "shift": "shift",
    "super": "super",
    "meta": "super",
    "win": "super",
    "cmd": "super",
    "command": "super",
    "capslock": "Caps_Lock",
    "printscreen": "Print",
    "menu": "Menu",
    "plus": "plus",
    "minus": "minus",
    "equal": "equal",
    "comma": "comma",
    "period": "period",
    "slash": "slash",
    "backslash": "backslash",
    "semicolon": "semicolon",
    "apostrophe": "apostrophe",
    "grave": "grave",
    "bracketleft": "bracketleft",
    "bracketright": "bracketright",
}


def normalize_key(spec: str) -> str:
    """Normalize ``"Ctrl+Shift+t"`` -> ``"ctrl+shift+t"`` with X keysym names."""
    if not spec or not spec.strip():
        raise InvalidToolInput("empty key specification")
    parts = [p for p in spec.replace(" ", "").split("+") if p]
    if not parts:
        raise InvalidToolInput(f"cannot parse key specification {spec!r}")
    out: list[str] = []
    for part in parts:
        lowered = part.lower()
        if lowered in KEY_ALIASES:
            out.append(KEY_ALIASES[lowered])
        elif len(part) == 1:
            out.append(part)
        elif lowered.startswith("f") and lowered[1:].isdigit():
            out.append(f"F{int(lowered[1:])}")
        else:
            # Pass through: probably already a valid keysym (e.g. "Num_Lock").
            out.append(part)
    return "+".join(out)


@dataclass(frozen=True, slots=True)
class InputResult:
    action: str
    detail: dict

    def to_dict(self) -> dict:
        return {"action": self.action, **self.detail}


def _no_display_detail() -> str:
    """Why xdotool could not reach a screen, in terms somebody can act on."""
    display = os.environ.get("DISPLAY")
    if not display:
        return (
            "DISPLAY is not set, so there is no session to send input to. This usually "
            "means LAI was started by something that strips the environment — a service, "
            "cron, or an MCP client."
        )
    return (
        f"the X server on DISPLAY={display!r} is not answering. If this was the agent's "
        f"own screen, it has been shut down."
    )


class InputController:
    """Thin, well-validated wrapper around xdotool."""

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT, type_delay_ms: int = 12) -> None:
        self.timeout = timeout
        self.type_delay_ms = max(0, int(type_delay_ms))
        self._xdotool = shutil.which("xdotool")

    @property
    def available(self) -> bool:
        return self._xdotool is not None

    def _run(self, *args: str) -> str:
        if not self._xdotool:
            raise BackendUnavailable(
                "xdotool is not installed", detail="install it with: sudo apt install xdotool"
            )
        try:
            proc = subprocess.run(
                [self._xdotool, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailable(
                f"xdotool timed out after {self.timeout}s", detail=" ".join(args)
            ) from exc
        if proc.returncode != 0:
            complaint = (proc.stderr or proc.stdout).strip()
            if "can't open display" in complaint.lower():
                # xdotool's own words are "Can't open display: (null)", which
                # tells the reader that something is null. The useful fact is
                # which display it tried and whether one was named at all.
                raise BackendUnavailable(f"xdotool {args[0]} failed", detail=_no_display_detail())
            raise BackendUnavailable(f"xdotool {args[0]} failed", detail=complaint)
        return proc.stdout.strip()

    # -- pointer ---------------------------------------------------------

    def cursor_position(self) -> Point:
        out = self._run("getmouselocation", "--shell")
        values = dict(
            line.split("=", 1) for line in out.splitlines() if "=" in line
        )
        return Point(int(values.get("X", 0)), int(values.get("Y", 0)))

    def move(self, point: Point, *, verify: bool = True, tolerance: int = 2) -> InputResult:
        """Move the pointer and confirm it arrived.

        Deliberately not ``xdotool mousemove --sync``: that waits for a motion
        event which never arrives when the pointer is already at the target (or
        under some window managers), so it blocks until the subprocess timeout
        and turns a 5 ms action into a 10 s failure. Polling the position
        ourselves is bounded and cannot hang.
        """
        self._run("mousemove", str(point.x), str(point.y))
        if not verify:
            return InputResult("move", {"x": point.x, "y": point.y})

        deadline = time.monotonic() + MOVE_VERIFY_TIMEOUT
        landed = None
        while time.monotonic() < deadline:
            try:
                landed = self.cursor_position()
            except BackendUnavailable:
                break
            if abs(landed.x - point.x) <= tolerance and abs(landed.y - point.y) <= tolerance:
                return InputResult("move", {"x": landed.x, "y": landed.y})
            time.sleep(MOVE_VERIFY_POLL)

        # Pointer barriers and some WMs legitimately clamp the position; report
        # where it actually is rather than pretending the request was honoured.
        actual = landed or point
        return InputResult(
            "move",
            {"x": actual.x, "y": actual.y, "requested": {"x": point.x, "y": point.y}},
        )

    def click(
        self,
        point: Point | None = None,
        *,
        button: str = "left",
        count: int = 1,
        delay_ms: int = 90,
    ) -> InputResult:
        code = self._button_code(button)
        if count < 1 or count > 5:
            raise InvalidToolInput(f"click count must be 1..5, got {count}")
        if point is not None:
            self.move(point)
            time.sleep(0.02)
        self._run("click", "--repeat", str(count), "--delay", str(max(1, delay_ms)), str(code))
        pos = point or self.cursor_position()
        return InputResult("click", {"x": pos.x, "y": pos.y, "button": button, "count": count})

    def double_click(self, point: Point | None = None, *, button: str = "left") -> InputResult:
        return self.click(point, button=button, count=2)

    def mouse_down(self, button: str = "left") -> InputResult:
        self._run("mousedown", str(self._button_code(button)))
        return InputResult("mouse_down", {"button": button})

    def mouse_up(self, button: str = "left") -> InputResult:
        self._run("mouseup", str(self._button_code(button)))
        return InputResult("mouse_up", {"button": button})

    def drag(
        self,
        start: Point,
        end: Point,
        *,
        button: str = "left",
        steps: int = 24,
        step_delay: float = 0.008,
    ) -> InputResult:
        """Press at ``start``, glide to ``end``, release.

        Intermediate moves matter: many toolkits ignore an instant teleport drag.
        """
        steps = max(2, min(int(steps), 200))
        self.move(start)
        self.mouse_down(button)
        try:
            for i in range(1, steps + 1):
                fraction = i / steps
                # Intermediate points only need to be emitted, not confirmed;
                # verifying each one would cost a subprocess round-trip per step.
                self.move(
                    Point(
                        round(start.x + (end.x - start.x) * fraction),
                        round(start.y + (end.y - start.y) * fraction),
                    ),
                    verify=(i == steps),
                )
                time.sleep(step_delay)
        finally:
            self.mouse_up(button)
        return InputResult(
            "drag",
            {"from": {"x": start.x, "y": start.y}, "to": {"x": end.x, "y": end.y}, "button": button},
        )

    def scroll(
        self, direction: str = "down", *, amount: int = 3, point: Point | None = None
    ) -> InputResult:
        mapping = {
            "up": "wheel_up",
            "down": "wheel_down",
            "left": "wheel_left",
            "right": "wheel_right",
        }
        key = mapping.get(direction.lower())
        if key is None:
            raise InvalidToolInput(
                f"scroll direction must be one of {sorted(mapping)}, got {direction!r}"
            )
        if amount < 1 or amount > 50:
            raise InvalidToolInput(f"scroll amount must be 1..50, got {amount}")
        if point is not None:
            self.move(point)
        self._run("click", "--repeat", str(amount), "--delay", "22", str(BUTTONS[key]))
        return InputResult("scroll", {"direction": direction, "amount": amount})

    # -- keyboard --------------------------------------------------------

    def type_text(self, text: str, *, delay_ms: int | None = None) -> InputResult:
        if not isinstance(text, str):
            raise InvalidToolInput("text must be a string")
        if text == "":
            return InputResult("type", {"length": 0})
        delay = self.type_delay_ms if delay_ms is None else max(0, int(delay_ms))
        self._run("type", "--clearmodifiers", "--delay", str(delay), "--", text)
        return InputResult("type", {"length": len(text)})

    def key(self, spec: str, *, count: int = 1) -> InputResult:
        combo = normalize_key(spec)
        if count < 1 or count > 50:
            raise InvalidToolInput(f"key count must be 1..50, got {count}")
        self._run("key", "--clearmodifiers", "--repeat", str(count), "--delay", "28", combo)
        return InputResult("key", {"key": combo, "count": count})

    def key_sequence(self, specs: list[str], *, gap: float = 0.05) -> InputResult:
        for spec in specs:
            self.key(spec)
            time.sleep(gap)
        return InputResult("key_sequence", {"keys": [normalize_key(s) for s in specs]})

    @staticmethod
    def _button_code(button: str) -> int:
        code = BUTTONS.get(str(button).lower())
        if code is None:
            raise InvalidToolInput(
                f"button must be one of {sorted(BUTTONS)}, got {button!r}"
            )
        return code
