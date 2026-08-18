"""Delegating code to a coding agent.

The division of labour is the point: a coding CLI writes the code, LAI does
the thing no coding CLI can — open the result on a real screen and check it.
So the properties that matter here are about honesty and confinement. What
comes back must be evidence (files that actually changed on disk), not the
worker's own claim; the worker is confined to one named directory; and a
worker that is missing, unauthenticated or slow must fail in a way the agent
can act on.
"""

from __future__ import annotations

import subprocess

import pytest

from lai.config import Config
from lai.tools import build_registry
from lai.tools.base import ToolContext
from lai.tools.coding import CODERS, PREFERENCE, available_coders


@pytest.fixture
def registry():
    return build_registry()


@pytest.fixture
def ctx():
    return ToolContext(desktop=None, config=Config())


def fake_coder(monkeypatch, *, returncode=0, stdout="done", stderr="", writes=None, record=None):
    """Stand in for the CLI: optionally writes files, like a real one would."""
    monkeypatch.setattr("lai.tools.coding.shutil.which", lambda name: "/usr/bin/" + name)

    class Completed:
        def __init__(self):
            self.returncode, self.stdout, self.stderr = returncode, stdout, stderr

    def run(argv, **kwargs):
        if record is not None:
            record.append((argv, kwargs))
        for name, content in (writes or {}).items():
            from pathlib import Path

            path = Path(kwargs["cwd"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return Completed()

    monkeypatch.setattr(subprocess, "run", run)


# -- the happy path ------------------------------------------------------


def test_the_work_is_reported_as_files_that_actually_changed(registry, ctx, tmp_path, monkeypatch):
    fake_coder(monkeypatch, stdout="Built the page.", writes={"index.html": "<h1>hi</h1>"})
    result = registry.call(
        "code_agent", {"task": "build a page", "workspace": str(tmp_path)}, ctx
    )
    assert result.ok
    assert result.data["changed_count"] == 1
    assert "index.html" in result.content
    assert "Built the page." in result.content
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<h1>hi</h1>"


def test_the_agent_is_told_to_verify_rather_than_believe(registry, ctx, tmp_path, monkeypatch):
    """A coding agent's summary is a claim; the screen is the evidence."""
    fake_coder(monkeypatch, stdout="Everything works perfectly!", writes={"a.py": "x = 1"})
    result = registry.call("code_agent", {"task": "t", "workspace": str(tmp_path)}, ctx)
    assert "Verify it" in result.content


def test_a_worker_that_wrote_nothing_is_called_out(registry, ctx, tmp_path, monkeypatch):
    fake_coder(monkeypatch, stdout="I have created the file for you.")
    result = registry.call("code_agent", {"task": "t", "workspace": str(tmp_path)}, ctx)
    assert result.ok
    assert "No files changed" in result.content, "the claim must not stand unchallenged"


def test_the_task_and_the_workspace_reach_the_cli(registry, ctx, tmp_path, monkeypatch):
    calls: list = []
    fake_coder(monkeypatch, record=calls)
    registry.call(
        "code_agent", {"task": "write a haiku module", "workspace": str(tmp_path), "coder": "claude"}, ctx
    )
    argv, kwargs = calls[0]
    assert "write a haiku module" in argv
    assert str(tmp_path) in argv, "the CLI must be told which directory it may touch"
    assert kwargs["cwd"] == str(tmp_path)


def test_a_missing_workspace_is_created(registry, ctx, tmp_path, monkeypatch):
    fake_coder(monkeypatch)
    target = tmp_path / "new" / "project"
    assert registry.call("code_agent", {"task": "t", "workspace": str(target)}, ctx).ok
    assert target.is_dir()


def test_every_known_coder_can_be_selected(registry, ctx, tmp_path, monkeypatch):
    for coder in PREFERENCE:
        calls: list = []
        fake_coder(monkeypatch, record=calls)
        result = registry.call(
            "code_agent", {"task": "t", "workspace": str(tmp_path), "coder": coder}, ctx
        )
        assert result.ok, coder
        assert calls[0][0][0] == CODERS[coder][0]


# -- refusing usefully ---------------------------------------------------


def test_a_job_with_no_task_is_refused(registry, ctx, tmp_path):
    result = registry.call("code_agent", {"task": "  ", "workspace": str(tmp_path)}, ctx)
    assert not result.ok and "task is required" in result.content


def test_a_workspace_is_required(registry, ctx):
    result = registry.call("code_agent", {"task": "t"}, ctx)
    assert not result.ok and "workspace" in result.content


def test_a_file_cannot_be_used_as_a_workspace(registry, ctx, tmp_path):
    target = tmp_path / "a-file.txt"
    target.write_text("x", encoding="utf-8")
    result = registry.call("code_agent", {"task": "t", "workspace": str(target)}, ctx)
    assert not result.ok and "not a directory" in result.content


def test_an_unknown_coder_lists_the_real_ones(registry, ctx, tmp_path, monkeypatch):
    fake_coder(monkeypatch)
    result = registry.call(
        "code_agent", {"task": "t", "workspace": str(tmp_path), "coder": "copilot"}, ctx
    )
    assert not result.ok and "claude" in result.content


def test_a_coder_that_is_not_installed_says_what_is(registry, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("lai.tools.coding.shutil.which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    result = registry.call(
        "code_agent", {"task": "t", "workspace": str(tmp_path), "coder": "codex"}, ctx
    )
    assert not result.ok
    assert "not installed" in result.content and "claude" in result.content


def test_with_no_coder_installed_it_says_to_write_the_files_itself(registry, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("lai.tools.coding.shutil.which", lambda name: None)
    result = registry.call("code_agent", {"task": "t", "workspace": str(tmp_path)}, ctx)
    assert not result.ok
    assert "file_write" in result.content


def test_a_failing_worker_that_changed_nothing_suggests_checking_its_login(
    registry, ctx, tmp_path, monkeypatch
):
    fake_coder(monkeypatch, returncode=1, stderr="Invalid API key")
    result = registry.call("code_agent", {"task": "t", "workspace": str(tmp_path)}, ctx)
    assert not result.ok
    assert "lai models test cli:" in result.content


def test_a_failure_that_still_wrote_files_is_reported_as_partial_work(
    registry, ctx, tmp_path, monkeypatch
):
    """Half-done work must be visible, not thrown away as an error."""
    fake_coder(monkeypatch, returncode=1, stdout="ran out of context", writes={"half.py": "x"})
    result = registry.call("code_agent", {"task": "t", "workspace": str(tmp_path)}, ctx)
    assert result.ok and result.data["changed_count"] == 1
    assert result.data["exit_code"] == 1


def test_a_slow_worker_times_out_with_advice(registry, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("lai.tools.coding.shutil.which", lambda name: "/usr/bin/" + name)

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(subprocess, "run", slow)
    result = registry.call(
        "code_agent", {"task": "t", "workspace": str(tmp_path), "timeout": 1}, ctx
    )
    assert not result.ok
    assert "smaller pieces" in result.content


def test_the_timeout_is_capped(registry, ctx, tmp_path, monkeypatch):
    from lai.tools.coding import MAX_TIMEOUT

    calls: list = []
    fake_coder(monkeypatch, record=calls)
    registry.call(
        "code_agent", {"task": "t", "workspace": str(tmp_path), "timeout": 99_999}, ctx
    )
    assert calls[0][1]["timeout"] <= MAX_TIMEOUT


# -- safety --------------------------------------------------------------


def test_it_is_classified_as_destructive(registry):
    """It writes files and runs commands: `ask` mode must gate it like shell_exec."""
    from lai.safety.policy import Risk

    spec = next(s for s in registry.specs() if s.name == "code_agent")
    assert spec.risk is Risk.DESTRUCTIVE


def test_noisy_output_is_trimmed_from_the_front(registry, ctx, tmp_path, monkeypatch):
    from lai.tools.coding import MAX_OUTPUT_CHARS

    fake_coder(monkeypatch, stdout="START" + ("chatter " * 5000) + "THE CONCLUSION")
    result = registry.call("code_agent", {"task": "t", "workspace": str(tmp_path)}, ctx)
    assert "THE CONCLUSION" in result.content, "the end is the part that matters"
    assert len(result.content) < MAX_OUTPUT_CHARS * 2


def test_available_coders_reflects_the_machine(monkeypatch):
    monkeypatch.setattr("lai.tools.coding.shutil.which", lambda name: "/usr/bin/x" if name == "codex" else None)
    assert available_coders() == ["codex"]
