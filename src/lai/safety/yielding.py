"""Standing aside while the human is using their own machine.

A desktop agent shares one mouse and one keyboard with its owner. Two hands on
them at once does not split the work: the agent's click lands in whatever
window the human just switched to, its typing goes into their document, and the
run's next screenshot shows a desktop neither of them arranged.

So before the agent touches the input devices, it asks how long the human has
been still. If they are mid-keystroke it waits, quietly, and carries on when
they stop. That is the whole feature, and it is what makes it reasonable to
leave an agent running on the machine you are sitting at.

Deliberately narrow:

* **Only input.** Reading the screen while somebody types is harmless, so
  observation never waits.
* **Bounded.** If the human simply keeps working, the agent gives up waiting
  and says so rather than hanging until the task's timeout.
* **Fails open.** A machine with no idle extension gets an agent that works,
  not one that refuses to move.
"""

from __future__ import annotations

import time
from collections.abc import Callable

POLL_INTERVAL = 0.5


class Yielded(Exception):
    """The human kept working, so the agent gave up its turn."""

    def __init__(self, waited: float) -> None:
        super().__init__(
            f"waited {waited:.0f}s for you to finish — the desktop is yours. "
            "Run it again when you are done, or set safety.yield_to_user = false."
        )
        self.waited = waited


def wait_for_the_human(
    idle_seconds: Callable[[], float],
    *,
    settle: float = 4.0,
    limit: float = 300.0,
    on_wait: Callable[[float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> float:
    """Block until the human has been still for ``settle`` seconds.

    Returns how long it waited. Raises :class:`Yielded` if they are still
    working after ``limit`` — an agent that blocks forever is worse than one
    that admits it cannot get a turn.
    """
    started = now()
    announced = False
    while True:
        try:
            idle = idle_seconds()
        except Exception:
            # No idle extension, no display, a broken probe: an agent that
            # cannot tell whether you are there must still be able to work.
            return 0.0

        if idle >= settle:
            return max(0.0, now() - started)

        waited = now() - started
        if waited >= limit:
            raise Yielded(waited)
        if on_wait is not None and not announced:
            on_wait(idle)
            announced = True
        sleep(POLL_INTERVAL)
