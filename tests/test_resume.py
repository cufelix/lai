"""Continuing a conversation.

An agent that forgets between invocations makes you re-explain the machine,
the task and what already failed — the most tedious thing about tools like
this. What matters here: the transcript really comes back, new turns append to
the same file rather than starting a second one for a single conversation, and
a session that cannot be found is a sentence rather than a crash.
"""

from __future__ import annotations

import pytest

from lai.agent.providers.base import Message
from lai.agent.session import Session
from lai.chat.session_pick import resume


@pytest.fixture
def sessions(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    return directory


def make(sessions, task, *texts):
    session = Session()
    session.task = task
    session.bind(sessions)
    for text in texts:
        session.append(Message.user(text))
    return session


def test_the_most_recent_conversation_comes_back(sessions):
    make(sessions, "older", "first")
    newest = make(sessions, "the latest thing", "hello there")

    found, why = resume(sessions)
    assert found is not None
    assert found.id == newest.id
    assert any("hello there" in str(m.content) for m in found.messages)
    assert "the latest thing" in why


def test_a_session_can_be_named(sessions):
    wanted = make(sessions, "wanted", "keep me")
    make(sessions, "other", "not me")

    found, _ = resume(sessions, wanted.id)
    assert found is not None and found.id == wanted.id


def test_a_prefix_is_enough(sessions):
    wanted = make(sessions, "t", "x")
    found, _ = resume(sessions, wanted.id[:6])
    assert found is not None and found.id == wanted.id


def test_an_ambiguous_prefix_says_so_rather_than_guessing(sessions, monkeypatch):
    """Silently picking one of two conversations is worse than asking again."""
    first = make(sessions, "a", "x")
    second = Session(id=first.id[:4] + "zzzzzzzz")
    second.task = "b"
    second.bind(sessions)
    second.append(Message.user("y"))

    found, why = resume(sessions, first.id[:4])
    assert found is None
    assert "matches several" in why


def test_an_unknown_id_explains_where_to_look(sessions):
    make(sessions, "a", "x")
    found, why = resume(sessions, "nosuchthing")
    assert found is None and "lai sessions" in why


def test_no_sessions_at_all_is_not_an_error(tmp_path):
    found, why = resume(tmp_path / "nothing")
    assert found is None and "no past sessions" in why


def test_an_empty_transcript_is_refused(sessions):
    session = Session()
    session.task = "started and abandoned"
    session.bind(sessions)
    found, why = resume(sessions)
    assert found is None and "nothing in it" in why


def test_new_turns_append_to_the_same_transcript(sessions):
    original = make(sessions, "keep going", "first message")
    found, _ = resume(sessions)
    found.append(Message.user("second message"))

    reloaded = Session.load(sessions / f"{original.id}.jsonl")
    texts = " ".join(str(m.content) for m in reloaded.messages)
    assert "first message" in texts and "second message" in texts
    assert len(list(sessions.glob("*.jsonl"))) == 1, "one conversation, one file"


# -- the command -----------------------------------------------------------


def test_the_resume_command_signals_rather_than_printing():
    from lai.chat.commands import RESUME, Context, run

    assert run(Context(runtime=None), "/resume abc123") == RESUME + "abc123"
    assert run(Context(runtime=None), "/resume") == RESUME


def test_sessions_lists_past_conversations(sessions, tmp_path):
    from lai.chat.commands import Context, run
    from lai.config import load_config

    make(sessions, "what windows are open", "x")
    runtime = type("R", (), {"config": load_config().with_overrides(home=tmp_path)})()
    text = run(Context(runtime=runtime), "/sessions")
    assert "what windows are open" in text
    assert "/resume" in text


def test_the_cli_maps_its_flags(tmp_path, monkeypatch):
    from lai.cli import _resume_choice, build_parser

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    assert _resume_choice(build_parser().parse_args(["chat"])) == ""
    assert _resume_choice(build_parser().parse_args(["chat", "--continue"])) == "last"
    assert _resume_choice(build_parser().parse_args(["chat", "--resume", "abc"])) == "abc"


def test_a_session_is_named_by_its_first_instruction(sessions):
    """A session is bound before the first instruction arrives, so the declared
    task is usually empty — and a list with no descriptions makes resuming by
    id guesswork."""
    session = Session()
    session.bind(sessions)  # no task known yet, as in a real run
    session.append(Message.user("open the editor and write a haiku"))

    listing = Session.list_sessions(sessions)
    assert listing[0]["task"].startswith("open the editor")


def test_a_declared_task_still_wins(sessions):
    make(sessions, "the declared task", "some later message")
    assert Session.list_sessions(sessions)[0]["task"] == "the declared task"


def test_a_transcript_with_nothing_to_say_is_simply_unnamed(sessions):
    Session().bind(sessions)
    assert Session.list_sessions(sessions)[0]["task"] == ""
