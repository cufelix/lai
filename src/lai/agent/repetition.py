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

Deliberately narrow, because false positives here are worse than the disease:

* Only **failures** count. Polling a window list until it changes, waiting
  twice, taking two screenshots — all legitimate, all identical, none of them
  stuck.
* Only **identical arguments** count. Clicking two different buttons that both
  fail is exploration, not repetition.
* Anything that **succeeds** clears the count for that call, because the world
  moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

WARN_AT = 2
"""The second identical failure gets a warning attached."""

REFUSE_AT = 3
"""The third is not run at all."""

MAX_TRACKED = 200
"""Fingerprints kept; a long run must not grow a dictionary without bound."""


@dataclass(slots=True)
class Repetition:
    """How many times each exact call has failed the same way."""

    failures: dict = field(default_factory=dict)

    def fingerprint(self, name: str, arguments: dict) -> str:
        try:
            rendered = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)[:400]
        except (TypeError, ValueError):
            rendered = repr(arguments)[:400]
        return f"{name}:{rendered}"

    def should_refuse(self, name: str, arguments: dict) -> str:
        """The message to return instead of running it, or "" to go ahead."""
        count = self.failures.get(self.fingerprint(name, arguments), 0)
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

    def record(self, name: str, arguments: dict, *, ok: bool, content: str = "") -> str:
        """Note the outcome. Returns a warning to append to the result, or "".

        A success clears the count: the world moved, and whatever failed before
        is no longer evidence about now.
        """
        key = self.fingerprint(name, arguments)
        if ok:
            self.failures.pop(key, None)
            return ""

        count = self.failures.get(key, 0) + 1
        if len(self.failures) >= MAX_TRACKED and key not in self.failures:
            self.failures.clear()
        self.failures[key] = count

        if count < WARN_AT:
            return ""
        return (
            f"\n\n[This is attempt {count} at {name} with these exact arguments, and it has "
            "failed the same way each time. Do not repeat it — observe the current state, "
            "try a different approach, or call task_blocked.]"
        )

    def stuck_on(self) -> list[tuple[str, int]]:
        """Calls that have failed repeatedly, worst first — for the audit log."""
        return sorted(
            ((key, count) for key, count in self.failures.items() if count >= WARN_AT),
            key=lambda pair: -pair[1],
        )
