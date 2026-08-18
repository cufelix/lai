"""System prompt construction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lai.agent.prompt import (
    build_system_prompt,
    environment_block,
    safety_block,
    skills_block,
)
from lai.config import SafetyConfig
from lai.osl.geometry import Monitor, Rect


class Exploding:
    """Every attribute access raises — perception must never break the prompt."""

    def __getattr__(self, name):
        raise RuntimeError(f"no display ({name})")


class FakeScreen:
    def monitors(self):
        return [Monitor("HDMI-1", Rect(0, 0, 1920, 1080), primary=True)]

    def virtual_bounds(self):
        return Rect(0, 0, 1920, 1080)


class FakeWindow:
    def __init__(self, title, wm_class):
        self.title = title
        self.wm_class = wm_class
        self.bounds = Rect(10, 20, 800, 600)

    def as_tuple(self):
        return self.bounds.as_tuple()


class FakeWindows:
    def active_window(self):
        return FakeWindow("Untitled Document", "Xed")

    def list_windows(self):
        return [FakeWindow("a", "Firefox"), FakeWindow("b", "Xed")]


class FakeA11y:
    def __init__(self, available=True):
        self.available = available


class FakeDesktop:
    def __init__(self, a11y_available=True):
        self.screen = FakeScreen()
        self.windows = FakeWindows()
        self.a11y = FakeA11y(a11y_available)


class FakeSkill:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class FakeSkills:
    def __init__(self, skills):
        self._skills = skills

    def list(self):
        return self._skills


# -- assembly ------------------------------------------------------------


def test_prompt_contains_the_core_sections():
    prompt = build_system_prompt()
    assert "You are LAI" in prompt
    assert "# How to work" in prompt
    assert "Observe" in prompt and "Verify" in prompt
    assert "task_complete" in prompt and "task_blocked" in prompt


def test_prompt_teaches_semantic_first_tool_choice():
    prompt = build_system_prompt()
    assert "ui_*" in prompt and "computer_*" in prompt
    assert "app_open" in prompt


def test_prompt_includes_optional_sections_when_supplied():
    prompt = build_system_prompt(
        desktop=FakeDesktop(),
        safety=SafetyConfig(mode="auto"),
        skills=FakeSkills([FakeSkill("filing", "use when filing invoices")]),
        extra="PROJECT NOTE: be careful with the printer.",
    )
    assert "# Permissions" in prompt
    assert "# Skills" in prompt and "filing" in prompt
    assert "PROJECT NOTE" in prompt
    assert "1920x1080" in prompt


def test_prompt_omits_skills_when_there_are_none():
    assert "# Skills" not in build_system_prompt(skills=FakeSkills([]))
    assert "# Skills" not in build_system_prompt(skills=None)


def test_prompt_has_no_blank_runs():
    prompt = build_system_prompt(desktop=FakeDesktop(), safety=SafetyConfig())
    assert "\n\n\n" not in prompt


# -- environment ---------------------------------------------------------


def test_environment_block_without_a_desktop():
    block = environment_block()
    assert "# Environment" in block
    assert "OS:" in block and "Working directory:" in block


def test_environment_block_with_a_desktop():
    block = environment_block(FakeDesktop())
    assert "HDMI-1 1920x1080" in block
    assert "Virtual screen: 1920x1080" in block
    assert "Currently focused: 'Untitled Document'" in block
    assert "Open windows (2)" in block


def test_environment_block_warns_when_accessibility_is_missing():
    assert "accessibility is unavailable" in environment_block(FakeDesktop(a11y_available=False))


def test_environment_block_survives_a_broken_desktop():
    block = environment_block(Exploding())
    assert "# Environment" in block  # degraded, but produced


def test_environment_block_honours_an_explicit_cwd(tmp_path):
    assert str(tmp_path) in environment_block(cwd=tmp_path)


# -- safety --------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "fragment"),
    [
        ("readonly", "READONLY"),
        ("ask", "ASK"),
        ("auto", "AUTO"),
        ("yolo", "UNATTENDED"),
    ],
)
def test_safety_block_describes_each_mode(mode, fragment):
    block = safety_block(SafetyConfig(mode=mode))
    assert fragment in block
    assert "# Permissions" in block


def test_safety_block_always_states_the_hard_limits():
    for mode in ("readonly", "ask", "auto", "yolo"):
        block = safety_block(SafetyConfig(mode=mode))
        assert "password managers" in block
        assert "rm -rf" in block


def test_safety_block_mentions_dry_run():
    block = safety_block(replace(SafetyConfig(), dry_run=True))
    assert "DRY RUN" in block


def test_safety_block_tolerates_an_unknown_object():
    assert "# Permissions" in safety_block(object())


# -- skills --------------------------------------------------------------


def test_skills_block_lists_names_and_descriptions_only():
    block = skills_block(FakeSkills([
        FakeSkill("alpha", "use when doing alpha"),
        FakeSkill("beta", "use when doing beta"),
    ]))
    assert "**alpha**" in block and "use when doing alpha" in block
    assert "**beta**" in block
    assert "skill_load" in block, "the prompt must explain how to load a skill"


def test_skills_block_is_empty_for_no_skills():
    assert skills_block(None) == ""
    assert skills_block(FakeSkills([])) == ""


def test_skills_block_survives_a_broken_registry():
    class Broken:
        def list(self):
            raise RuntimeError("disk gone")

    assert skills_block(Broken()) == ""


def test_skills_block_caps_the_listing():
    many = FakeSkills([FakeSkill(f"s{i}", f"desc {i}") for i in range(200)])
    block = skills_block(many)
    assert block.count("**s") <= 80
