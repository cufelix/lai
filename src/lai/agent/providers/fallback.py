"""A provider that steps aside when its backend does.

The failure this exists for is mundane and constant: a subscription hits its
five-hour quota, a hosted endpoint has a bad minute, a key expires. Without
this, a long autonomous run dies mid-task and the desktop is left half-way
through whatever it was doing — the worst possible moment to stop.

So the chain is ordered, lazy and sticky. Ordered: the configured backend is
always tried first. Lazy: a fallback is only built when it is actually needed,
because constructing one can mean probing a socket or spawning a CLI. Sticky:
once a run moves to the next backend it stays there, since flapping between two
models mid-task produces incoherent behaviour.

What counts as "step aside" is deliberately narrow — quota, auth and the
far end being broken. A malformed request would fail identically everywhere, so
it is raised rather than used to burn through every backend on the machine.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ...errors import ProviderError
from .base import Message, Provider, TurnResult

# Matched against the whole error text, which for every provider here already
# carries the HTTP status and the vendor's own message.
SWITCH_PATTERNS = (
    r"\b429\b", r"rate[ _-]?limit", r"quota", r"usage limit", r"insufficient",
    r"out of credit", r"credit balance", r"billing",
    r"\b401\b", r"\b403\b", r"unauthorized", r"forbidden", r"invalid[ _-]api[ _-]key",
    r"authentication", r"not signed in", r"please (?:run )?/?login",
    r"\b500\b", r"\b502\b", r"\b503\b", r"\b529\b", r"overloaded",
    r"service unavailable", r"bad gateway", r"internal server error",
    r"timed out", r"timeout", r"connection (?:reset|refused|error)", r"network",
)
_SWITCH = re.compile("|".join(SWITCH_PATTERNS), re.IGNORECASE)


def should_switch(error: BaseException) -> bool:
    """True when another backend could plausibly succeed where this one failed."""
    return bool(_SWITCH.search(str(error)))


@dataclass(slots=True)
class Candidate:
    """One backend in the chain, built only if it is reached."""

    name: str
    build: Callable[[], Provider]
    label: str = ""

    def describe(self) -> str:
        return self.label or self.name


@dataclass(slots=True)
class FallbackProvider:
    """Runs the first backend that works, in order.

    Presents itself as whichever backend is currently answering, so the loop,
    the audit log and every UI keep reporting the truth without knowing this
    class exists.
    """

    candidates: list[Candidate]
    on_switch: Callable[[str, str, str], None] | None = None
    """Called as (from_name, to_name, reason) when the chain advances."""

    index: int = 0
    active: Provider | None = None
    failures: dict = field(default_factory=dict)
    """Backend name → why it stepped aside. Surfaced by the UIs."""

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ProviderError("no model backend available", detail="run `lai setup` to add one")
        self.active = self.candidates[0].build()

    # -- Provider protocol -------------------------------------------------

    @property
    def name(self) -> str:
        return getattr(self.active, "name", self.candidates[self.index].name)

    @property
    def model(self) -> str:
        return getattr(self.active, "model", "")

    @property
    def chain(self) -> list[str]:
        return [c.name for c in self.candidates]

    def complete(self, messages: list[Message], **kwargs) -> TurnResult:
        last: ProviderError | None = None
        while True:
            try:
                return self.active.complete(messages, **kwargs)  # type: ignore[union-attr]
            except ProviderError as exc:
                last = exc
                if not should_switch(exc) or not self._advance(str(exc)):
                    raise
            except Exception as exc:  # a transport blowing up is still a backend failure
                last = ProviderError(f"{self.name}: {exc}")
                if not should_switch(exc) or not self._advance(str(exc)):
                    raise last from exc

    def close(self) -> None:
        closer = getattr(self.active, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:
                pass

    # -- internals ---------------------------------------------------------

    def _advance(self, reason: str) -> bool:
        """Move to the next backend that can be built. False when none is left."""
        failed = self.name
        self.failures[failed] = _short(reason)
        self.close()

        while self.index + 1 < len(self.candidates):
            self.index += 1
            candidate = self.candidates[self.index]
            try:
                self.active = candidate.build()
            except Exception as exc:
                # An unusable fallback is not an error worth surfacing on its
                # own — it is simply not a fallback. Record it and keep going.
                self.failures[candidate.name] = _short(str(exc))
                continue
            if self.on_switch is not None:
                try:
                    self.on_switch(failed, self.name, _short(reason))
                except Exception:
                    pass
            return True
        self.active = None
        return False


def _short(text: str, limit: int = 160) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"
