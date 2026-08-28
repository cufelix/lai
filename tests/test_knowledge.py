"""The learning journal.

The properties that matter: a note is a markdown file a human can read and
correct, the same lesson learned twice does not become two lines, a note name
coming from a model or an HTTP request cannot escape the directory, and an
unreadable journal degrades to "no notes" rather than a failed run.
"""

from __future__ import annotations

import pytest

from lai.knowledge import MAX_CONTEXT_CHARS, Journal, Note, parse, slugify


@pytest.fixture
def journal(tmp_path):
    return Journal.open(tmp_path)


# -- parsing -------------------------------------------------------------


def test_frontmatter_is_read():
    note = parse("---\ntitle: The Editor\ntags: app, editor\nupdated: 2026-08-18\n---\n\n- it is Xed\n")
    assert note.title == "The Editor"
    assert note.tags == ("app", "editor")
    assert note.body == "- it is Xed"


def test_a_bare_markdown_file_is_still_a_note():
    """People will drop plain notes in the directory; that must work."""
    note = parse("just some hard-won knowledge", name="drawing")
    assert note.body == "just some hard-won knowledge"
    assert note.title == "drawing"


def test_the_summary_is_the_first_real_line():
    note = parse("---\ntitle: X\n---\n\n\n- the canvas starts at y=140\n- and ends lower\n")
    assert note.summary == "the canvas starts at y=140"


@pytest.mark.parametrize(("raw", "expected"), [
    ("Drawing App", "drawing-app"),
    ("../../etc/passwd", "etc-passwd"),
    ("  Spaces  ", "spaces"),
    ("!!!", "note"),
    ("a" * 200, "a" * 64),
])
def test_slugs_are_safe_and_stable(raw, expected):
    assert slugify(raw) == expected


# -- writing and reading -------------------------------------------------


def test_a_written_note_is_readable_markdown(journal, tmp_path):
    journal.write("Drawing", "- the canvas starts below the toolbar", tags=("app",))
    text = (tmp_path / "notes" / "drawing.md").read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "title: Drawing" in text and "tags: app" in text
    assert "- the canvas starts below the toolbar" in text


def test_a_note_round_trips(journal):
    journal.write("Editor", "- it is Xed", tags=("app", "editor"))
    note = journal.get("editor")
    assert note.title == "Editor" and note.tags == ("app", "editor")
    assert "Xed" in note.body


def test_writing_the_same_name_replaces_it(journal):
    journal.write("editor", "- old")
    journal.write("editor", "- new")
    assert journal.get("editor").body == "- new"
    assert len(journal.list()) == 1


def test_append_adds_a_line(journal):
    journal.append("editor", "it is Xed")
    journal.append("editor", "the save field is called Name:")
    body = journal.get("editor").body
    assert "- it is Xed" in body and "- the save field is called Name:" in body


def test_learning_the_same_thing_twice_does_not_duplicate_it(journal):
    """An agent rediscovers the same fact constantly; the journal must not grow."""
    journal.append("editor", "The save dialog field is called Name:")
    journal.append("editor", "the save dialog field is called name:")
    journal.append("editor", "The save dialog field is called Name")
    assert journal.get("editor").body.count("save dialog") == 1


def test_append_keeps_the_tags_it_already_had(journal):
    journal.append("editor", "one", tags=("app",))
    journal.append("editor", "two", tags=("editor",))
    assert set(journal.get("editor").tags) == {"app", "editor"}


def test_an_empty_lesson_does_not_create_noise(journal):
    journal.append("editor", "   ")
    assert journal.get("editor").body == ""


def test_delete_removes_the_file(journal):
    journal.write("editor", "- x")
    assert journal.delete("editor") is True
    assert journal.get("editor") is None
    assert journal.delete("editor") is False


def test_listing_is_newest_first(journal):
    import os
    import time

    journal.write("old", "- x")
    journal.write("new", "- y")
    os.utime(journal.directory / "old.md", (time.time() - 500, time.time() - 500))
    assert [n.name for n in journal.list()] == ["new", "old"]


# -- safety --------------------------------------------------------------


def test_a_note_name_cannot_escape_the_directory(journal, tmp_path):
    """Names arrive from the model and from HTTP; both are untrusted."""
    journal.write("../../evil", "- pwned")
    assert not (tmp_path.parent / "evil.md").exists()
    assert not (tmp_path / "evil.md").exists()
    assert [p.parent.name for p in (tmp_path / "notes").glob("*.md")] == ["notes"]


def test_reading_a_traversing_name_finds_nothing(journal, tmp_path):
    (tmp_path / "secret.md").write_text("private", encoding="utf-8")
    assert journal.get("../secret") is None


def test_a_missing_directory_is_simply_empty(tmp_path):
    assert Journal(tmp_path / "nope").list() == []
    assert Journal(tmp_path / "nope").context_block("anything") == ""


def test_an_oversized_note_is_capped(journal):
    from lai.knowledge import MAX_NOTE_CHARS

    journal.write("huge", "x" * (MAX_NOTE_CHARS * 2))
    assert len(journal.get("huge").body) <= MAX_NOTE_CHARS


# -- retrieval -----------------------------------------------------------


def test_search_prefers_the_matching_topic(journal):
    journal.write("drawing", "- the canvas starts below the toolbar", tags=("drawing",))
    journal.write("firefox", "- the url bar is focused with ctrl+l", tags=("browser",))
    found = journal.search("draw a house in the drawing app")
    assert found[0].name == "drawing"


def test_search_falls_back_to_everything_for_a_vague_task(journal):
    journal.write("a", "- x")
    journal.write("b", "- y")
    assert len(journal.search("go")) == 2


def test_the_context_block_tells_the_model_to_verify(journal):
    journal.write("drawing", "- the canvas starts at y=140")
    block = journal.context_block("draw something")
    assert "learned on this machine" in block
    assert "verify" in block.lower(), "a stale note followed blindly is worse than none"
    assert "y=140" in block


def test_the_context_block_is_bounded(journal):
    for index in range(40):
        journal.write(f"note-{index}", "- " + ("padding " * 200))
    assert len(journal.context_block("padding")) <= MAX_CONTEXT_CHARS + 500


def test_no_notes_means_no_block(journal):
    assert journal.context_block("anything") == ""


def test_to_dict_is_serialisable(journal):
    import json

    journal.write("editor", "- it is Xed", tags=("app",))
    payload = json.loads(json.dumps(journal.get("editor").to_dict()))
    assert payload["name"] == "editor" and payload["tags"] == ["app"]


def test_a_note_renders_its_own_frontmatter():
    note = Note(name="x", title="X", body="- a", tags=("t",), updated=0.0)
    assert note.render().startswith("---\ntitle: X\ntags: t\nupdated: ")


def test_a_note_that_turns_out_to_be_wrong_is_corrected_not_mentioned(tmp_path):
    """A wrong lesson persists and compounds. This machine learned 'app_open
    Text Editor starts a process but no window appears — use Cursor instead',
    which is untrue, and it steered every run afterwards. Telling the agent to
    mention it in a summary nobody reads does not undo that; telling it to
    correct the note does."""
    from lai.knowledge import Journal

    journal = Journal.open(tmp_path)
    journal.write("editor", "- the editor is Xed")
    block = journal.context_block("open the editor")
    lower = block.lower()
    assert "correct" in lower
    assert "learn" in lower or "note" in lower
    assert "summary" not in lower, "nothing acts on a summary"


def test_notes_carry_their_age(tmp_path):
    """Advice learned once, months ago, from a single bad run should not read
    like something established."""
    from lai.knowledge import Journal

    journal = Journal.open(tmp_path)
    journal.write("editor", "- the editor is Xed")
    block = journal.context_block("open the editor")
    assert "today" in block.lower() or "ago" in block.lower()
