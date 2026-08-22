"""Noticing when the agent is repeating itself.

The most expensive failure in an autonomous run is not a wrong action — it is
the *same* wrong action, taken twelve times, each attempt costing a full model
turn and each one failing for the reason the last one did. A human stops after
the second try and asks what else could be true. An agent with no memory of the
last step will happily spend its entire budget.

So identical calls that fail identically are counted. The second one comes back
with the failure *and* a note saying it has been tried; the third is refused
outright, with the alternatives spelled out. Refusing is the point: a message
the model can ignore does not break the cycle, and by then the evidence is
conclusive.

A click, though, always succeeds. Sending a button press to a pixel is
reported as done whether or not anything happened, so a model hunting for a
toolbar it cannot quite see will click the same coordinates twenty times in a
row and every one of them comes back ✓. That is the same failure wearing a
different mask, so identical *acting* calls are counted too — on a longer
leash, because clicking one button twice is ordinary and clicking it six times
never is.

Deliberately narrow, because false positives here are worse than the disease:

* Only **identical arguments** count. Clicking two different buttons is
  exploration, not repetition.
* Reading is never counted. Polling a window list until it changes, waiting
  twice, taking two screenshots — all legitimate, all identical, none stuck.
* A **failure that stops happening** clears its count, because the world moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

WARN_AT = 2
"""The second identical failure gets a warning attached."""

REFUSE_AT = 3
"""The third is not run at all."""

REPEAT_WARN_AT = 4
"""...but an action that *reports* success gets more rope before the warning."""

REPEAT_REFUSE_AT = 6
"""Six identical clicks at one pixel is not a strategy that needs a seventh."""

MAX_TRACKED = 200
"""Fingerprints kept; a long run must not grow a dictionary without bound."""


@dataclass(slots=True)
class Repetition:
    """How many times each exact call has failed the same way."""

    failures: dict = field(default_factory=dict)
    repeats: dict = field(default_factory=dict)
    """Identical acting calls that all claimed to work."""

    def fingerprint(self, name: str, arguments: dict) -> str:
        try:
            rendered = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)[:400]
        except (TypeError, ValueError):
            rendered = repr(arguments)[:400]
        return f"{name}:{rendered}"

    def should_refuse(self, name: str, arguments: dict) -> str:
        """The message to return instead of running it, or "" to go ahead."""
        key = self.fingerprint(name, arguments)
        repeated = self.repeats.get(key, 0)
        if repeated >= REPEAT_REFUSE_AT - 1:
            return (
                f"Refused: {name} has been called with exactly these arguments "
                f"{repeated} times already. Each one reported success and nothing "
                "changed, which means the click is not landing where you think it "
                "is.\n"
                "Look before acting again: computer_screenshot to see the current "
                "state, ui_snapshot for named elements with exact coordinates, or "
                "ui_click by name instead of guessing a pixel."
            )
        count = self.failures.get(key, 0)
        if count < REFUSE_AT - 1:
            return ""
        return (
            f"Refused: you have already called {name} with exactly these arguments "
            f"{count} times and it failed the same way each time. Running it again "
            "will not produce a different result.\n"
            "Change something real: look at the current state with ui_snapshot or "
            "computer_screenshot, target a different element, use a different tool, "
            "or call task_blocked explaining what stopped you."
        )

    def record(
        self, name: str, arguments: dict, *, ok: bool, content: str = "", acting: bool = False
    ) -> str:
        """Note the outcome. Returns a warning to append to the result, or "".

        ``acting`` marks a call that drives the mouse or keyboard. Those report
        success unconditionally, so repetition is the only evidence available
        that they are not working.
        """
        key = self.fingerprint(name, arguments)
        if ok:
            self.failures.pop(key, None)
            if not acting:
                self.repeats.pop(key, None)
                return ""
            repeated = self._bump(self.repeats, key)
            if repeated < REPEAT_WARN_AT:
                return ""
            return (
                f"\n\n[That is {repeated} identical {name} calls in this run. It reports "
                "success every time because sending the event succeeds — that is not "
                "evidence anything happened. Take a screenshot, or find the element by "
                "name with ui_snapshot, before doing it again.]"
            )

        count = self._bump(self.failures, key)
        if count < WARN_AT:
            return ""
        return (
            f"\n\n[This is attempt {count} at {name} with these exact arguments, and it has "
            "failed the same way each time. Do not repeat it — observe the current state, "
            "try a different approach, or call task_blocked.]"
        )

    @staticmethod
    def _bump(counts: dict, key: str) -> int:
        if len(counts) >= MAX_TRACKED and key not in counts:
            counts.clear()
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    def stuck_on(self) -> list[tuple[str, int]]:
        """Calls that have failed repeatedly, worst first — for the audit log."""
        return sorted(
            ((key, count) for key, count in self.failures.items() if count >= WARN_AT),
            key=lambda pair: -pair[1],
        )
