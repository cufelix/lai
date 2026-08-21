"""Slash-command completion.

Typing `/` should not require remembering the table. The moment a line starts
with a slash the candidate commands appear above the composer and narrow as you
type — prefix matches first, then anything whose letters appear in order, so
`/mdl` still finds `/model`.

The matching lives here as a plain function because that is the part worth
testing; the widget below is only how it is drawn.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import OptionList
from textual.widgets.option_list import Option


def visible_commands() -> list[tuple[str, str]]:
    """The commands worth offering — aliases are real but not advertised."""
    from ..chat.commands import COMMANDS  # noqa: PLC0415

    return [(name, spec[1]) for name, spec in COMMANDS.items() if not spec[2]]


def _subsequence(needle: str, haystack: str) -> bool:
    position = 0
    for character in needle:
        position = haystack.find(character, position) + 1
        if position == 0:
            return False
    return True


def search(prefix: str, commands: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    """Commands matching what has been typed so far, best first.

    An exact prefix always outranks a scattered match, so `/s` offers `status`
    before `sessions` gets reordered by some cleverer score — predictable beats
    smart when the list is eight items long and you are already typing.
    """
    table = commands if commands is not None else visible_commands()
    wanted = prefix.strip().lower()
    if not wanted:
        return list(table)
    exact = [entry for entry in table if entry[0].startswith(wanted)]
    loose = [
        entry for entry in table
        if entry not in exact and _subsequence(wanted, entry[0])
    ]
    return exact + loose


def line_prefix(text: str) -> str | None:
    """The command being typed, or None when this line is not a command.

    Only the first line counts, and only before the first space: once you have
    typed `/model open` you are giving an argument, not choosing a command.
    """
    first = text.split("\n", 1)[0]
    if not first.startswith("/"):
        return None
    body = first[1:]
    if " " in body:
        return None
    return body


class Palette(OptionList):
    """The completion list itself. Hidden until a line starts with a slash."""

    DEFAULT_CSS = """
    Palette { display: none; height: auto; max-height: 10;
              border: round $accent; background: $surface; }
    Palette.open { display: block; }
    """

    def show(self, matches: list[tuple[str, str]]) -> None:
        self.clear_options()
        if not matches:
            self.close()
            return
        width = max(len(name) for name, _ in matches) + 2
        for name, description in matches:
            label = Text()
            label.append(f"/{name.ljust(width)}", style="bold cyan")
            # The table's descriptions are written for a markup-capable log;
            # here they are drawn literally, so the backticks would show.
            label.append(description.replace("`", ""), style="dim")
            self.add_option(Option(label, id=name))
        self.add_class("open")
        self.highlighted = 0

    def close(self) -> None:
        self.remove_class("open")

    @property
    def open(self) -> bool:
        return self.has_class("open")

    def chosen(self) -> str:
        """The highlighted command name, or "" when nothing is highlighted."""
        index = self.highlighted
        if index is None or not self.option_count:
            return ""
        return str(self.get_option_at_index(index).id or "")
