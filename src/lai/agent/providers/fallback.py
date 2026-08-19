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
    # A CLI backend that has already exhausted its own retries and still exits
    # non-zero is broken for this run, whatever it blames. Handing over to the
    # next backend is strictly better than ending the task here.
    r"api_error", r"exited [1-9]", r"produced no output",
    # A backend that will not follow the protocol is not going to start.
    r"refused the protocol",
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
    home: object = None
    """Where to record which backends are refusing, and for how long."""
    _last: tuple = ("", "")
    """The last backend that was actually answering, kept for honest reporting
    once the chain is exhausted and ``active`` is None."""
    failures: dict = field(default_factory=dict)
    """Backend name → why it stepped aside. Surfaced by the UIs."""

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ProviderError("no model backend available", detail="run `lai setup` to add one")
        self.active = self.candidates[0].build()
        self._remember()

    # -- Provider protocol -------------------------------------------------

    @property
    def name(self) -> str:
        if self.active is not None:
            return self.active.name
        return self._last[0] or self.candidates[self.index].name

    @property
    def model(self) -> str:
        return self.active.model if self.active is not None else self._last[1]

    def _remember(self) -> None:
        if self.active is not None:
            self._remember_pair(self.active.name, self.active.model)

    def _remember_pair(self, name: str, model: str) -> None:
        self._last = (name, model)

    def _clear(self, name: str) -> None:
        """A backend that just answered is healthy, whatever it did before."""
        if self.home is None:
            return
        try:
            from .health import note_success  # noqa: PLC0415

            note_success(self.home, name)
        except Exception:
            pass

    def _record(self, name: str, reason: str) -> None:
        """Write down why a backend stepped aside, so tomorrow's run knows."""
        if self.home is None:
            return
        try:
            from .health import note_failure  # noqa: PLC0415

            note_failure(self.home, name, reason)
        except Exception:
            pass

    @property
    def chain(self) -> list[str]:
        return [c.name for c in self.candidates]

    @property
    def context_chars(self) -> int:
        """Whatever the backend currently answering can take."""
        return int(getattr(self.active, "context_chars", 0) or 0)

    def complete(self, messages: list[Message], **kwargs) -> TurnResult:
        if self.active is None:
            # Every backend has stepped aside. Saying so — with each one's
            # reason — is the difference between a diagnosis and an
            # AttributeError on the next turn.
            raise self.exhausted()

        last: ProviderError | None = None
        while True:
            try:
                turn = self.active.complete(messages, **kwargs)  # type: ignore[union-attr]
            except ProviderError as exc:
                last = exc
                if not should_switch(exc) or not self._advance(str(exc)):
                    raise
            except Exception as exc:  # a transport blowing up is still a backend failure
                last = ProviderError(f"{self.name}: {exc}")
                if not should_switch(exc) or not self._advance(str(exc)):
                    raise last from exc
            else:
                self._clear(self.name)
                return turn

    def exhausted(self) -> ProviderError:
        """The error to raise when nothing is left to try."""
        if self.failures:
            detail = "; ".join(f"{name}: {why}" for name, why in self.failures.items())
        else:
            detail = "no backend was usable"
        return ProviderError(
            f"all {len(self.candidates)} model backend(s) failed",
            detail=detail + " — `lai models` shows what this machine can use, "
            "`lai setup` adds another",
        )

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
        self._record(failed, reason)
        self.close()

        while self.index + 1 < len(self.candidates):
            self.index += 1
            candidate = self.candidates[self.index]
            try:
                self.active = candidate.build()
                self._remember()
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
