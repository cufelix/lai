"""Learning from a finished run.

Reflection is an improvement, never a prerequisite: it must never fail a task,
never invent a lesson out of a model that answered badly, and never turn one
rediscovered fact into a growing pile of duplicates.
"""

from __future__ import annotations

import json

import pytest

from lai.agent.loop import RunResult
from lai.agent.providers.base import Message, TextBlock, ToolCall, ToolResultBlock, TurnResult, Usage
from lai.agent.reflect import MIN_STEPS, build_trace, reflect
from lai.knowledge import Journal


class FakeProvider:
    """Answers reflection with whatever it was given."""

    name, model = "fake", "fake-1"

    def __init__(self, answer: str = "", explode: bool = False):
        self.answer = answer
        self.explode = explode
        self.prompts: list[str] = []

    def complete(self, messages, **kwargs):
        if self.explode:
            raise RuntimeError("backend is down")
        self.prompts.append(messages[-1].content[0].text)
        return TurnResult(
            message=Message("assistant", [TextBlock(self.answer)]),
            stop_reason="end_turn", usage=Usage(), model=self.model,
        )


class FakeAudit:
    def __init__(self):
        self.written: list[tuple] = []

    def write(self, event, **fields):
        self.written.append((event, fields))


@pytest.fixture
def journal(tmp_path):
    return Journal.open(tmp_path)


def answer(*notes) -> str:
    return json.dumps({"notes": list(notes)})


def lesson(topic="editor", title="The editor", text="it is actually Xed", tags=("app",)):
    return {"topic": topic, "title": title, "lesson": text, "tags": list(tags)}


def done(steps=6, status="completed"):
    return RunResult(status=status, steps=steps, summary="did the thing")


# -- the happy path ------------------------------------------------------


def test_a_lesson_becomes_a_note(journal):
    written = reflect(provider=FakeProvider(answer(lesson())), journal=journal,
                      task="open the editor", result=done(), trace="→ app_open")
    assert [n.name for n in written] == ["editor"]
    assert "Xed" in journal.get("editor").body
    assert journal.get("editor").tags == ("app",)


def test_the_same_lesson_twice_stays_one_line(journal):
    for _ in range(3):
        reflect(provider=FakeProvider(answer(lesson())), journal=journal,
                task="open the editor", result=done(), trace="→ app_open")
    assert journal.get("editor").body.count("Xed") == 1


def test_the_model_is_shown_what_it_already_knows(journal):
    """Without the existing topics it invents a new file for every lesson."""
    journal.write("editor", "- it is Xed", title="The editor")
    provider = FakeProvider(answer(lesson()))
    reflect(provider=provider, journal=journal, task="t", result=done(), trace="x")
    assert "editor: The editor" in provider.prompts[0]


def test_the_trace_and_outcome_reach_the_model(journal):
    provider = FakeProvider(answer())
    reflect(provider=provider, journal=journal, task="draw a house",
            result=done(status="blocked"), trace="→ computer_drag {}")
    prompt = provider.prompts[0]
    assert "draw a house" in prompt and "blocked" in prompt and "computer_drag" in prompt


def test_a_reply_in_a_fence_is_still_understood(journal):
    raw = "Sure:\n```json\n" + answer(lesson()) + "\n```"
    assert reflect(provider=FakeProvider(raw), journal=journal, task="t",
                   result=done(), trace="x")


def test_a_bare_list_is_accepted(journal):
    raw = json.dumps([lesson()])
    assert reflect(provider=FakeProvider(raw), journal=journal, task="t",
                   result=done(), trace="x")


# -- restraint -----------------------------------------------------------


def test_nothing_learned_is_a_valid_answer(journal):
    assert reflect(provider=FakeProvider(answer()), journal=journal, task="t",
                   result=done(), trace="x") == []
    assert journal.list() == []


def test_prose_never_becomes_a_note(journal):
    """A model that ignores the format must not have its chatter filed as fact."""
    provider = FakeProvider("I learned that the editor is nice, I think.")
    assert reflect(provider=provider, journal=journal, task="t", result=done(), trace="x") == []
    assert journal.list() == []


def test_a_trivial_run_is_not_worth_reflecting_on(journal):
    provider = FakeProvider(answer(lesson()))
    assert reflect(provider=provider, journal=journal, task="t",
                   result=done(steps=MIN_STEPS - 1), trace="x") == []
    assert provider.prompts == [], "it must not spend a model call to learn nothing"


def test_incomplete_lessons_are_dropped(journal):
    raw = answer({"topic": "a"}, {"lesson": "orphan"}, lesson())
    written = reflect(provider=FakeProvider(raw), journal=journal, task="t",
                      result=done(), trace="x")
    assert [n.name for n in written] == ["editor"]


def test_only_a_handful_of_lessons_are_kept(journal):
    from lai.agent.reflect import MAX_LESSONS

    raw = answer(*[lesson(topic=f"topic-{i}", text=f"fact {i}") for i in range(12)])
    written = reflect(provider=FakeProvider(raw), journal=journal, task="t",
                      result=done(), trace="x")
    assert len(written) == MAX_LESSONS


# -- never breaking the run ----------------------------------------------


def test_a_broken_provider_is_recorded_not_raised(journal):
    audit = FakeAudit()
    assert reflect(provider=FakeProvider(explode=True), journal=journal, task="t",
                   result=done(), trace="x", audit=audit) == []
    assert audit.written and audit.written[0][0] == "reflect_failed"


def test_what_was_learned_is_audited(journal):
    audit = FakeAudit()
    reflect(provider=FakeProvider(answer(lesson())), journal=journal, task="t",
            result=done(), trace="x", audit=audit)
    assert ("learned", {"notes": ["editor"]}) in audit.written


def test_no_result_means_no_reflection(journal):
    provider = FakeProvider(answer(lesson()))
    assert reflect(provider=provider, journal=journal, task="t", result=None, trace="x") == []


# -- the trace -----------------------------------------------------------


def test_the_trace_shows_calls_results_and_speech():
    session = type("S", (), {"messages": [
        Message("assistant", [TextBlock("Opening it."), ToolCall("1", "app_open", {"name": "Xed"})]),
        Message("user", [ToolResultBlock("1", "Opened 'Text Editor'")]),
        Message("assistant", [ToolCall("2", "ui_click", {"name": "Save"})]),
        Message("user", [ToolResultBlock("2", "no such control", is_error=True)]),
    ]})()
    trace = build_trace(session)
    assert "→ app_open" in trace and "Xed" in trace
    assert "✓ Opened 'Text Editor'" in trace
    assert "✗ no such control" in trace, "failures are the most valuable thing to learn from"
    assert "· Opening it." in trace


def test_the_trace_survives_an_empty_session():
    assert build_trace(type("S", (), {"messages": []})()) == ""


# -- compaction is when facts move out of the transcript -----------------


def test_compacting_promotes_durable_facts_into_the_journal(journal, tmp_path):
    """A summary put back into the transcript survives until the next
    compaction and then goes too. What will still be true tomorrow has to
    leave the conversation entirely."""
    from lai.agent.loop import Agent
    from lai.agent.session import Session
    from lai.config import load_config

    agent = Agent.__new__(Agent)
    agent.config = load_config().with_overrides(home=tmp_path)
    agent.journal = journal
    agent.provider = FakeProvider(answer(lesson(text="the save dialog field is called Name:")))
    agent.session = Session()
    agent.session.task = "save a file"
    agent.session.messages = [Message.user("save a file")] * 12  # a transcript worth compacting
    agent.audit = FakeAudit()
    agent.events: list = []
    agent._emit = lambda kind, payload: agent.events.append((kind, payload))

    agent._promote_to_memory("worked through the save dialog; the field is called Name:")

    assert "Name:" in journal.get("editor").body
    assert any(kind == "learned" for kind, _ in agent.events)


def test_nothing_is_promoted_when_learning_is_off(journal, tmp_path):
    from dataclasses import replace

    from lai.agent.loop import Agent
    from lai.agent.session import Session
    from lai.config import load_config

    config = load_config().with_overrides(home=tmp_path)
    config = config.with_overrides(learning=replace(config.learning, reflect=False))

    agent = Agent.__new__(Agent)
    agent.config = config
    agent.journal = journal
    agent.provider = FakeProvider(answer(lesson()))
    agent.session = Session()
    agent.audit = FakeAudit()
    agent._emit = lambda kind, payload: None

    agent._promote_to_memory("something happened")
    assert journal.list() == []


def test_a_failure_while_promoting_never_breaks_the_compaction(journal, tmp_path):
    from lai.agent.loop import Agent
    from lai.agent.session import Session
    from lai.config import load_config

    agent = Agent.__new__(Agent)
    agent.config = load_config().with_overrides(home=tmp_path)
    agent.journal = journal
    agent.provider = FakeProvider(explode=True)
    agent.session = Session()
    agent.audit = FakeAudit()
    agent._emit = lambda kind, payload: None

    agent._promote_to_memory("a summary")  # must not raise
