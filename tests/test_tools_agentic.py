"""The agentic tool family: memory, delegation and scheduling.

These tools are the agent acting on *itself*, so the thing worth testing is the
wiring: each one must find its store through ``ctx.extra`` when the runtime
injected one, fall back to the config home when it did not, and fail with a
usable message rather than an exception when the service is absent.
"""

from __future__ import annotations

import pytest

from lai.agent.memory import MemoryStore
from lai.config import Config
from lai.scheduler import Scheduler, TaskStore, make_task
from lai.tools import build_registry
from lai.tools.base import ToolContext


@pytest.fixture
def registry():
    return build_registry()


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    yield store
    store.close()


@pytest.fixture
def tasks(tmp_path):
    return TaskStore(tmp_path / "schedule.json")


def context(tmp_path, **extra) -> ToolContext:
    return ToolContext(config=Config(home=tmp_path), extra=extra)


def call(registry, name: str, args: dict, ctx: ToolContext):
    return registry.call(name, args, ctx)


# -- memory --------------------------------------------------------------


def test_memory_save_then_search_roundtrip(registry, memory, tmp_path):
    ctx = context(tmp_path, memory_store=memory)
    saved = call(registry, "memory_save", {
        "content": "Xed's save dialog names its filename field 'Name:'",
        "kind": "app",
        "key": "xed.save_field",
    }, ctx)
    assert saved.ok and "Remembered" in saved.content

    found = call(registry, "memory_search", {"query": "Xed save dialog"}, ctx)
    assert found.ok
    assert "Name:" in found.content
    assert found.data["results"][0]["key"] == "xed.save_field"


def test_memory_save_upserts_on_the_same_key(registry, memory, tmp_path):
    ctx = context(tmp_path, memory_store=memory)
    call(registry, "memory_save", {"content": "old fact", "key": "k1", "kind": "fact"}, ctx)
    call(registry, "memory_save", {"content": "corrected fact", "key": "k1", "kind": "fact"}, ctx)

    found = call(registry, "memory_search", {"query": "fact"}, ctx)
    contents = [r["content"] for r in found.data["results"]]
    assert "corrected fact" in contents
    assert "old fact" not in contents, "a same-key save must replace, not duplicate"


def test_memory_search_with_nothing_stored(registry, memory, tmp_path):
    result = call(registry, "memory_search", {"query": "anything"}, context(tmp_path, memory_store=memory))
    assert result.ok and result.data["results"] == []
    assert "No relevant memories" in result.content


def test_memory_forget_by_key_and_by_id(registry, memory, tmp_path):
    ctx = context(tmp_path, memory_store=memory)
    call(registry, "memory_save", {"content": "by key", "key": "gone", "kind": "fact"}, ctx)
    forgotten = call(registry, "memory_forget", {"id_or_key": "gone"}, ctx)
    assert forgotten.ok and forgotten.data["removed"] == 1

    saved = call(registry, "memory_save", {"content": "by id", "kind": "fact"}, ctx)
    entry_id = str(saved.data["id"])
    assert call(registry, "memory_forget", {"id_or_key": entry_id}, ctx).ok


def test_memory_forget_reports_a_miss(registry, memory, tmp_path):
    result = call(registry, "memory_forget", {"id_or_key": "never-saved"}, context(tmp_path, memory_store=memory))
    assert not result.ok and "no memory found" in result.content


def test_memory_falls_back_to_the_config_home(registry, tmp_path):
    """With nothing injected, the tools must still work off ctx.config.home."""
    ctx = context(tmp_path)
    assert call(registry, "memory_save", {"content": "stored on disk", "kind": "fact"}, ctx).ok
    assert (tmp_path / "memory.db").is_file()

    found = call(registry, "memory_search", {"query": "stored"}, ctx)
    assert "stored on disk" in found.content


# -- scheduling ----------------------------------------------------------


def test_schedule_task_then_list_and_remove(registry, tasks, tmp_path):
    ctx = context(tmp_path, task_store=tasks)
    created = call(registry, "schedule_task", {
        "name": "morning-check", "task": "summarise overnight mail", "schedule": "0 9 * * 1-5",
    }, ctx)
    assert created.ok
    task_id = created.data["id"]

    listed = call(registry, "schedule_list", {}, ctx)
    assert listed.ok and "morning-check" in listed.content
    assert listed.data["tasks"][0]["schedule"] == "0 9 * * 1-5"

    removed = call(registry, "schedule_remove", {"id": task_id}, ctx)
    assert removed.ok
    assert call(registry, "schedule_list", {}, ctx).data["tasks"] == []


def test_schedule_task_rejects_a_bad_expression(registry, tasks, tmp_path):
    result = call(registry, "schedule_task", {
        "name": "broken", "task": "never runs", "schedule": "not a cron",
    }, context(tmp_path, task_store=tasks))
    assert not result.ok and "invalid schedule" in result.content


def test_schedule_list_enabled_only(registry, tasks, tmp_path):
    from dataclasses import replace

    tasks.add(make_task(name="on", task="t", schedule="@daily"))
    tasks.add(replace(make_task(name="off", task="t", schedule="@daily"), enabled=False))

    ctx = context(tmp_path, task_store=tasks)
    assert len(call(registry, "schedule_list", {}, ctx).data["tasks"]) == 2
    only = call(registry, "schedule_list", {"enabled_only": True}, ctx)
    assert [t["name"] for t in only.data["tasks"]] == ["on"]


def test_schedule_remove_reports_a_miss(registry, tasks, tmp_path):
    result = call(registry, "schedule_remove", {"id": "nosuch"}, context(tmp_path, task_store=tasks))
    assert not result.ok and "no scheduled task" in result.content


def test_schedule_prefers_a_live_scheduler_over_a_bare_store(registry, tmp_path):
    """A running Scheduler owns the store; tools must edit that one, not a second copy."""
    live = TaskStore(tmp_path / "live.json")
    scheduler = Scheduler(live, lambda task: None)
    other = TaskStore(tmp_path / "unused.json")
    ctx = context(tmp_path, scheduler=scheduler, task_store=other)

    call(registry, "schedule_task", {"name": "x", "task": "t", "schedule": "@daily"}, ctx)
    assert len(live.list()) == 1
    assert other.list() == [], "the scheduler's own store must win"


def test_schedule_falls_back_to_the_config_home(registry, tmp_path):
    ctx = context(tmp_path)
    assert call(registry, "schedule_task", {"name": "x", "task": "t", "schedule": "@daily"}, ctx).ok
    assert (tmp_path / "schedule.json").is_file()


# -- delegation ----------------------------------------------------------


def test_delegate_without_a_running_agent_fails_cleanly(registry, tmp_path):
    """The tool needs the live Agent; absent it, say so rather than raising."""
    result = call(registry, "delegate", {"task": "do a subtask"}, context(tmp_path))
    assert not result.ok
    assert "no running agent" in result.content


def test_delegate_passes_the_task_through_and_reports_the_conclusion(registry, tmp_path, monkeypatch):
    captured: dict = {}

    class FakeResult:
        status = "completed"
        summary = "subtask done"

        def to_dict(self):
            return {"status": self.status, "summary": self.summary}

    def fake_run_subagent(*, task, parent, max_steps, system_extra):
        captured.update(task=task, parent=parent, max_steps=max_steps, system_extra=system_extra)
        return FakeResult()

    monkeypatch.setattr("lai.tools.agentic.run_subagent", fake_run_subagent)
    parent = object()
    result = call(registry, "delegate", {
        "task": "read the logs", "max_steps": 7, "system_extra": "be terse",
    }, context(tmp_path, agent=parent))

    assert result.ok and "subtask done" in result.content
    assert captured["task"] == "read the logs"
    assert captured["parent"] is parent
    assert captured["max_steps"] == 7
    assert captured["system_extra"] == "be terse"


def test_delegate_uses_the_default_step_budget(registry, tmp_path, monkeypatch):
    from lai.tools.agentic import DEFAULT_DELEGATE_STEPS

    captured: dict = {}

    class FakeResult:
        status = "completed"
        summary = "ok"

        def to_dict(self):
            return {}

    def fake_run_subagent(*, task, parent, max_steps, system_extra):
        captured["max_steps"] = max_steps
        return FakeResult()

    monkeypatch.setattr("lai.tools.agentic.run_subagent", fake_run_subagent)
    call(registry, "delegate", {"task": "x"}, context(tmp_path, agent=object()))
    assert captured["max_steps"] == DEFAULT_DELEGATE_STEPS


def test_delegate_surfaces_the_depth_cap(registry, tmp_path, monkeypatch):
    from lai.agent.subagent import SubagentDepthExceeded

    def fake_run_subagent(**kwargs):
        raise SubagentDepthExceeded(depth=2, limit=2)

    monkeypatch.setattr("lai.tools.agentic.run_subagent", fake_run_subagent)
    result = call(registry, "delegate", {"task": "recurse forever"}, context(tmp_path, agent=object()))
    assert not result.ok
    assert "depth" in result.content.lower()


def test_a_failed_subagent_is_reported_as_not_ok(registry, tmp_path, monkeypatch):
    class FakeResult:
        status = "blocked"
        summary = "could not find the window"

        def to_dict(self):
            return {"status": self.status}

    monkeypatch.setattr("lai.tools.agentic.run_subagent", lambda **kw: FakeResult())
    result = call(registry, "delegate", {"task": "x"}, context(tmp_path, agent=object()))
    assert not result.ok, "a blocked subagent must not read as success"
    assert "could not find the window" in result.content
