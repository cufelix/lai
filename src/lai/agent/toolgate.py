"""Deciding which tools the model is shown this turn.

LAI's own tools are its hands: about fifty schemas, and every one of them can
be needed on a desktop. MCP servers are different. Connect a handful — a
database, a deploy platform, a browser driver — and the registry grows to
several hundred tools whose schemas are re-sent, in full, on every single turn.
Measured on one ordinary machine that was 242 KB of JSON, roughly sixty
thousand tokens, to answer "how many windows do I have open".

So the extensions are gated on relevance to the task, and what is left out is
still named in one line so the model knows it exists. `tool_find` searches the
whole registry and unlocks what it finds, which costs one round trip in the
rare case the first guess was wrong — against sixty thousand tokens a turn in
the common case where the database tools were never going to be used.
"""

from __future__ import annotations

import re

from . import relevance

EXTENSION_PREFIX = "mcp:"
"""Registry group prefix marking a tool that came from an MCP server."""

DEFAULT_LIMIT = 8
"""How many matching extension tools to expose without being asked."""

MIN_SCORE = relevance.NAME_WEIGHT
"""A tool earns its schema by matching in its *name*, not in its prose.

Two hundred tool descriptions share a lot of words. A single hit in one is
coincidence; a hit in the name is usually what was meant.
"""


def _without(task: str, words: set[str]) -> str:
    """The task with the named servers removed."""
    drop = {part for word in words for part in re.split(r"[-_]", word)} | set(words)
    return " ".join(
        term for term in re.split(r"\W+", task)
        if term.lower() not in drop and not any(word.startswith(term.lower()) for word in words)
    )


def is_extension(spec) -> bool:
    """True for a tool that arrived from an MCP server rather than from LAI."""
    return str(getattr(spec, "group", "")).startswith(EXTENSION_PREFIX)


def server_of(spec) -> str:
    return str(getattr(spec, "group", ""))[len(EXTENSION_PREFIX):]


def servers_named(specs, task: str) -> set[str]:
    """Connected servers the task mentions by name."""
    terms = relevance.terms_of(task)
    if not terms:
        return set()
    servers = {server_of(spec) for spec in specs}
    return {
        server for server in servers
        if server and any(
            term == server or term in re.split(r"[-_]", server) or server.startswith(term)
            for term in terms
        )
    }


def rank_extensions(specs, task: str, *, limit: int, floor: int = MIN_SCORE) -> list:
    """The extension tools worth showing for this task, best first.

    ``floor`` is the score a tool must reach. The default demands a hit in the
    name, because this runs unprompted on every task and a coincidence in prose
    is not worth a schema. A model that has explicitly gone looking should pass
    1 instead: it asked, so a weak match is still worth showing it.

    Naming the service is the strongest signal there is — "query the users
    table in supabase" should not have to compete with every tool that happens
    to mention a user. Servers are matched before the word frequencies are
    consulted, because a server's own name appears across all of its tools by
    construction, and the noise filter would throw it away for that reason.
    """
    if limit <= 0 or not specs:
        return []
    named = servers_named(specs, task)
    pool = [spec for spec in specs if server_of(spec) in named] if named else list(specs)
    # Inside one server the server's own name is in every tool, so it cannot
    # say which tool — and it would take the rest of the sentence down with it
    # as noise. Drop it and let "open the page" do the choosing.
    scored = relevance.rank_scored(
        pool, _without(task, named) if named else task,
        name_of=lambda spec: spec.name.replace("__", " ").replace("_", " "),
        text_of=lambda spec: spec.description or "",
    )
    # A named server's tools are wanted even when the rest of the sentence says
    # nothing about which of them; anything else has to earn its place.
    floor = 1 if named else floor
    chosen = [spec for score, spec in scored if score >= floor][:limit]
    if named and not chosen:
        chosen = [spec for _, spec in scored][:limit]
    return chosen


class ToolGate:
    """Chooses the tool schemas for a turn, and remembers what was unlocked.

    Stateful on purpose: once the model has asked for a tool by name it stays
    available for the rest of the run. Taking it away again after one use would
    make a multi-step job with a database re-discover the same tool each turn.
    """

    def __init__(self, registry, *, limit: int = DEFAULT_LIMIT) -> None:
        self.registry = registry
        self.limit = limit
        self.unlocked: set[str] = set()

    # -- selection -------------------------------------------------------

    def _partition(self) -> tuple[list, list]:
        specs = self.registry.specs()
        core = [spec for spec in specs if not is_extension(spec)]
        extensions = [spec for spec in specs if is_extension(spec)]
        return core, extensions

    def choose(self, task: str) -> tuple[list, list]:
        """(shown, withheld) for this task."""
        core, extensions = self._partition()
        if not extensions:
            return core, []

        wanted = [spec for spec in extensions if spec.name in self.unlocked]
        rest = [spec for spec in extensions if spec.name not in self.unlocked]
        room = max(0, self.limit - len(wanted))
        chosen = rank_extensions(rest, task, limit=room)
        keep = {spec.name for spec in chosen}
        withheld = [spec for spec in rest if spec.name not in keep]
        return core + wanted + chosen, withheld

    def schemas(self, task: str, *, dialect: str = "anthropic") -> list[dict]:
        shown, _ = self.choose(task)
        render = "to_anthropic" if dialect == "anthropic" else f"to_{dialect}"
        return [getattr(spec, render)() for spec in shown]

    # -- discovery -------------------------------------------------------

    def unlock(self, names) -> list[str]:
        """Make named tools visible for the rest of the run. Returns what took."""
        took = [name for name in names if name in self.registry]
        self.unlocked.update(took)
        return took

    def describe_withheld(self, task: str) -> str:
        """One line for the system prompt naming what is connected but not shown.

        Hiding a tool silently would be the worst of both worlds: the model
        cannot use it and does not know to ask.
        """
        _, withheld = self.choose(task)
        if not withheld:
            return ""
        counts: dict[str, int] = {}
        for spec in withheld:
            counts[server_of(spec)] = counts.get(server_of(spec), 0) + 1
        listed = ", ".join(f"{server} ({count})" for server, count in sorted(counts.items()))
        return (
            "# Connected services\n"
            f"{len(withheld)} further tools are connected but not listed above, from: {listed}. "
            "They are not loaded because they do not match this task. If you need one, call "
            "`tool_find(query)` — it searches every connected tool and makes the matches "
            "callable for the rest of this run."
        )
