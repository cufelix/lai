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


# -- skills cost tokens on every turn ------------------------------------


SKILLS = [
    FakeSkill("python-testing", "Testing strategies using pytest, fixtures and coverage."),
    FakeSkill("python-patterns", "Pythonic idioms, PEP 8 and type hints."),
    FakeSkill("ffmpeg-analyse-video", "Analyse video content by extracting frames."),
    FakeSkill("travel-planner", "Plan trips, itineraries and budgets."),
    FakeSkill("dmux-workflows", "Multi-agent orchestration across tmux panes."),
    FakeSkill("clickhouse-io", "ClickHouse query optimisation and analytics."),
    FakeSkill("gsap", "GSAP animation reference for compositions."),
    FakeSkill("postgres-patterns", "PostgreSQL schema design and indexing."),
]


def test_a_matching_skill_is_described_in_full():
    from lai.agent.prompt import skills_block

    block = skills_block(FakeSkills(SKILLS), "write python tests for this module")
    assert "python-testing" in block
    assert "pytest" in block, "the matching skill's description is what makes it useful"


def test_the_rest_are_still_listed_by_name():
    """Nothing may become invisible just because it did not match."""
    from lai.agent.prompt import skills_block

    block = skills_block(FakeSkills(SKILLS), "write python tests")
    assert "travel-planner" in block
    assert "Plan trips" not in block, "but without spending tokens on its description"


def test_an_irrelevant_task_describes_nothing():
    """Describing six arbitrary skills is exactly the waste this avoids."""
    from lai.agent.prompt import skills_block

    block = skills_block(FakeSkills(SKILLS), "open the calculator and add two numbers")
    assert "pytest" not in block and "Plan trips" not in block
    assert "python-testing" in block and "travel-planner" in block, "all still named"


def test_a_common_word_does_not_count_as_a_match():
    """'open the calculator' used to match every skill containing 'workflow'."""
    from lai.agent.prompt import _rank_skills

    corpus = [FakeSkill(f"skill-{i}", "a workflow for working with work") for i in range(10)]
    _ranked, matched = _rank_skills(corpus, "work out the workflow")
    assert matched == 0, "a term matching the whole corpus distinguishes nothing"


def test_whole_words_only():
    from lai.agent.prompt import _rank_skills

    corpus = [FakeSkill("dmux-workflows", "orchestration"), FakeSkill("kitten-videos", "cute animals")]
    ranked, matched = _rank_skills(corpus, "work with kitten footage")
    assert matched == 1
    assert ranked[0].name == "kitten-videos", "'work' must not match 'workflows'"


def test_a_plural_still_matches():
    from lai.agent.prompt import _rank_skills

    corpus = [FakeSkill("video-tools", "editing"), FakeSkill("other", "nothing")]
    ranked, matched = _rank_skills(corpus, "make some videos")
    assert matched == 1 and ranked[0].name == "video-tools"


def test_no_task_keeps_the_original_order():
    from lai.agent.prompt import _rank_skills

    ranked, matched = _rank_skills(SKILLS, "")
    assert matched == 0
    assert [s.name for s in ranked] == [s.name for s in SKILLS]


def test_the_skills_block_shrinks_when_none_of_them_apply():
    """The measurable point: describing 40 skills costs the same on every turn
    of every run, whether or not any of them has anything to do with the task."""
    from lai.agent.prompt import build_system_prompt, skills_block

    many = [FakeSkill(f"skill-{i}", "A fairly wordy description. " * 12) for i in range(40)]
    focused = skills_block(FakeSkills(many), "open the calculator")
    everything = skills_block(FakeSkills(many))
    assert len(focused) < len(everything) / 4

    # And the whole prompt gets meaningfully cheaper, not just this section.
    whole = build_system_prompt(skills=FakeSkills(many), task="open the calculator")
    before = build_system_prompt(skills=FakeSkills(many))
    assert len(whole) < len(before) * 0.7


def test_a_broken_skill_registry_costs_nothing():
    from lai.agent.prompt import skills_block

    class Broken:
        def list(self):
            raise RuntimeError("skills directory vanished")

    assert skills_block(Broken(), "anything") == ""
