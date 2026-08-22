"""What the agent brings into a run.

An agent that starts every task from nothing repeats solved problems, re-derives
facts it was explicitly told, and cannot answer "carry on with what we were
doing". Three sources fix three different failures — and all of them must stay
bounded, because the point is to spend fewer tokens, not more.
"""

from __future__ import annotations

import time

import pytest

from lai.agent.memory import MemoryStore
from lai.agent.providers.base import Message
from lai.agent.recall import MAX_RECALL_CHARS, RECENT_WINDOW, build
from lai.agent.session import Session
from lai.knowledge import Journal


@pytest.fixture
def journal(tmp_path):
    return Journal.open(tmp_path)


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore.open(tmp_path)
    yield store
    store.close()


@pytest.fixture
def sessions(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    return directory


def a_session(sessions, task, *, age_seconds=60):
    import os

    session = Session()
    session.task = task
    session.bind(sessions)
    session.append(Message.user(task))
    stamp = time.time() - age_seconds
    os.utime(sessions / f"{session.id}.jsonl", (stamp, stamp))
    return session


# -- the three sources ---------------------------------------------------


def test_notes_about_this_machine_come_back(journal):
    journal.write("drawing", "- the canvas starts below the toolbar at y=140", tags=("drawing",))
    block = build(journal=journal, task="draw a house in the drawing app")
    assert "y=140" in block


def test_facts_it_was_told_to_remember_come_back(memory):
    """Without this the store is write-only: memory_save puts things in and
    nothing comes out unless the model thinks to search — which, having
    forgotten, it has no reason to do."""
    memory.remember("always save exports to ~/Documents/reports", kind="preference")
    block = build(memory=memory, task="export the report")
    assert "~/Documents/reports" in block
    assert "memory_search" in block, "and how to ask for more"


def test_recent_work_gives_carry_on_a_referent(sessions):
    a_session(sessions, "rename the invoices in ~/Documents", age_seconds=600)
    block = build(sessions_dir=sessions, task="carry on with that")
    assert "rename the invoices" in block
    assert "10 min ago" in block


def test_all_three_appear_together(journal, memory, sessions):
    journal.write("editor", "- the editor is Xed")
    memory.remember("the user prefers dark themes", kind="preference")
    a_session(sessions, "open the editor and write a note")
    block = build(journal=journal, memory=memory, sessions_dir=sessions, task="open the editor")
    assert "Xed" in block and "dark themes" in block and "write a note" in block


# -- restraint -----------------------------------------------------------


def test_nothing_known_means_nothing_added():
    assert build(task="anything at all") == ""


def test_stale_sessions_are_not_continuity(sessions):
    """Last week's task is not what "carry on" means."""
    a_session(sessions, "something from ages ago", age_seconds=RECENT_WINDOW + 3600)
    assert "ages ago" not in build(sessions_dir=sessions, task="carry on")


def test_only_a_handful_of_sessions_are_listed(sessions):
    for index in range(10):
        a_session(sessions, f"task number {index}", age_seconds=60 + index)
    block = build(sessions_dir=sessions, task="carry on")
    assert block.count("- ") <= 4


def test_the_whole_block_is_bounded(journal, sessions):
    for index in range(60):
        journal.write(f"note-{index}", "- " + ("padding " * 150))
    for index in range(10):
        a_session(sessions, f"task {index} " + "x" * 500, age_seconds=60)
    block = build(journal=journal, sessions_dir=sessions, task="padding")
    assert len(block) <= MAX_RECALL_CHARS


def test_memory_is_not_consulted_without_a_task(memory):
    """A blank task matches everything, which is the same as matching nothing."""
    memory.remember("some fact", kind="fact")
    assert build(memory=memory, task="") == ""


# -- failing soft --------------------------------------------------------


def test_a_broken_source_costs_its_section_not_the_run(journal):
    class Exploding:
        def context_block(self, *args, **kwargs):
            raise RuntimeError("the database is gone")

    journal.write("editor", "- the editor is Xed")
    block = build(journal=journal, memory=Exploding(), task="open the editor")
    assert "Xed" in block, "the working source still contributes"


def test_every_source_broken_is_simply_empty():
    class Exploding:
        def context_block(self, *args, **kwargs):
            raise RuntimeError("nope")

        def list(self, *args, **kwargs):
            raise RuntimeError("nope")

    assert build(journal=Exploding(), memory=Exploding(), task="x") == ""


# -- the agent uses it ---------------------------------------------------


def test_the_agent_puts_it_in_the_system_prompt(tmp_path, journal, memory):
    from lai.agent.loop import Agent
    from lai.agent.toolgate import ToolGate
    from lai.config import load_config
    from lai.tools.base import ToolRegistry

    journal.write("editor", "- the editor is actually Xed")
    memory.remember("the user prefers dark themes", kind="preference")

    agent = Agent.__new__(Agent)
    agent.config = load_config().with_overrides(home=tmp_path)
    agent.journal = journal
    agent.memory = memory
    agent.desktop = None
    agent.skills = None
    agent.cwd = None
    agent.system_extra = ""
    agent.session = Session()
    agent.gate = ToolGate(ToolRegistry())
    agent.provider = type("P", (), {"supports_vision": True})()
    agent.on_own_screen = False

    prompt = agent._build_system_prompt("open the editor with a dark theme")
    assert "Xed" in prompt and "dark themes" in prompt


def test_learning_switched_off_recalls_nothing(tmp_path, journal):
    from dataclasses import replace

    from lai.agent.loop import Agent
    from lai.config import load_config

    journal.write("editor", "- the editor is Xed")
    config = load_config().with_overrides(home=tmp_path)
    config = config.with_overrides(learning=replace(config.learning, enabled=False))

    agent = Agent.__new__(Agent)
    agent.config = config
    agent.journal = journal
    agent.memory = None
    assert agent._knowledge_block("open the editor") == ""
