"""Skills: frontmatter parsing, discovery, precedence, and safe installation."""

from __future__ import annotations

import pytest

from lai.errors import SkillError
from lai.skills.install import _safe_extract_path, _safe_name, install, uninstall
from lai.skills.registry import SkillRegistry, _parse_frontmatter, _parse_skill


def make_skill(root, name: str, description: str = "does a thing", body: str = "Step one.") -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8"
    )


# -- frontmatter ---------------------------------------------------------


def test_simple_keys():
    data = _parse_frontmatter("name: my-skill\ndescription: does things")
    assert data == {"name": "my-skill", "description": "does things"}


def test_quoted_values_are_unwrapped():
    data = _parse_frontmatter("name: \"quoted\"\ndescription: 'single'")
    assert data["name"] == "quoted" and data["description"] == "single"


def test_inline_list():
    assert _parse_frontmatter("tags: [a, b, c]")["tags"] == ["a", "b", "c"]
    assert _parse_frontmatter("tags: []")["tags"] == []


def test_booleans_are_coerced():
    data = _parse_frontmatter("enabled: true\nhidden: FALSE")
    assert data["enabled"] is True and data["hidden"] is False


def test_folded_multiline_value():
    data = _parse_frontmatter(
        "description: this is a long\n  description that wraps\n  across lines\nname: x"
    )
    assert data["description"] == "this is a long description that wraps across lines"
    assert data["name"] == "x"


def test_comments_and_blank_lines_ignored():
    data = _parse_frontmatter("# a comment\n\nname: x\n\n# another\ndescription: y")
    assert data == {"name": "x", "description": "y"}


def test_malformed_lines_are_skipped():
    assert _parse_frontmatter("this has no colon\nname: x")["name"] == "x"


def test_empty_frontmatter():
    assert _parse_frontmatter("") == {}


# -- parsing a skill file ------------------------------------------------


def test_parse_skill_reads_name_body_and_description(tmp_path):
    make_skill(tmp_path, "alpha", "use when alpha", "Do alpha things.")
    skill = _parse_skill(tmp_path / "alpha" / "SKILL.md")
    assert skill.name == "alpha"
    assert skill.description == "use when alpha"
    assert "Do alpha things." in skill.body


def test_missing_frontmatter_falls_back_to_folder_and_first_prose(tmp_path):
    directory = tmp_path / "bare"
    directory.mkdir()
    (directory / "SKILL.md").write_text("# Title\n\nThis is what it does.\n", encoding="utf-8")
    skill = _parse_skill(directory / "SKILL.md")
    assert skill.name == "bare"
    assert skill.description == "This is what it does."


def test_skill_with_no_description_at_all(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: empty\n---\n", encoding="utf-8")
    assert _parse_skill(directory / "SKILL.md").description == "(no description)"


def test_oversized_skill_is_skipped(tmp_path):
    from lai.skills.registry import MAX_SKILL_BYTES

    directory = tmp_path / "huge"
    directory.mkdir()
    (directory / "SKILL.md").write_text("x" * (MAX_SKILL_BYTES + 10), encoding="utf-8")
    assert _parse_skill(directory / "SKILL.md") is None


def test_unreadable_file_returns_none(tmp_path):
    assert _parse_skill(tmp_path / "ghost" / "SKILL.md") is None


# -- registry ------------------------------------------------------------


def test_discovery_lists_skills(tmp_path):
    for name in ("beta", "alpha", "gamma"):
        make_skill(tmp_path, name)
    registry = SkillRegistry([tmp_path])
    assert [s.name for s in registry.list()] == ["alpha", "beta", "gamma"]
    assert len(registry) == 3


def test_missing_search_path_is_ignored(tmp_path):
    make_skill(tmp_path, "alpha")
    registry = SkillRegistry([tmp_path / "nope", tmp_path])
    assert len(registry) == 1


def test_first_path_wins(tmp_path):
    high, low = tmp_path / "high", tmp_path / "low"
    make_skill(high, "shared", "the winning one")
    make_skill(low, "shared", "the losing one")
    registry = SkillRegistry([high, low])
    assert registry.get("shared").description == "the winning one"


def test_nested_discovery_respects_max_depth(tmp_path):
    make_skill(tmp_path / "a" / "b" / "c" / "d" / "e", "deep")
    make_skill(tmp_path / "a", "shallow")
    assert {s.name for s in SkillRegistry([tmp_path], max_depth=2).list()} == {"shallow"}
    assert "deep" in {s.name for s in SkillRegistry([tmp_path], max_depth=8).list()}


def test_hidden_directories_are_skipped(tmp_path):
    make_skill(tmp_path / ".git", "hidden")
    make_skill(tmp_path, "visible")
    assert {s.name for s in SkillRegistry([tmp_path]).list()} == {"visible"}


def test_get_by_exact_and_partial_name(tmp_path):
    make_skill(tmp_path, "invoice-filing")
    registry = SkillRegistry([tmp_path])
    assert registry.get("invoice-filing").name == "invoice-filing"
    assert registry.get("invoice").name == "invoice-filing"


def test_ambiguous_partial_name_raises(tmp_path):
    make_skill(tmp_path, "invoice-filing")
    make_skill(tmp_path, "invoice-parsing")
    with pytest.raises(SkillError, match="ambiguous"):
        SkillRegistry([tmp_path]).get("invoice")


def test_unknown_skill_raises_with_suggestions(tmp_path):
    make_skill(tmp_path, "alpha")
    with pytest.raises(SkillError, match="no skill named"):
        SkillRegistry([tmp_path]).get("omega")


def test_search_ranks_name_above_description(tmp_path):
    make_skill(tmp_path, "pdf-tools", "work with documents")
    make_skill(tmp_path, "other", "this mentions pdf in the description")
    results = SkillRegistry([tmp_path]).search("pdf")
    assert results[0].name == "pdf-tools"
    assert len(results) == 2


def test_search_with_no_match(tmp_path):
    make_skill(tmp_path, "alpha")
    assert SkillRegistry([tmp_path]).search("zzzz") == []


def test_empty_search_returns_everything(tmp_path):
    make_skill(tmp_path, "alpha")
    assert len(SkillRegistry([tmp_path]).search("")) == 1


def test_refresh_picks_up_new_skills(tmp_path):
    make_skill(tmp_path, "one")
    registry = SkillRegistry([tmp_path])
    assert len(registry) == 1
    make_skill(tmp_path, "two")
    assert len(registry) == 1, "results should be cached"
    assert len(registry.refresh()) == 2


# -- rendering -----------------------------------------------------------


def test_render_includes_scripts_and_references(tmp_path):
    make_skill(tmp_path, "bundled", "has extras", "Follow the script.")
    directory = tmp_path / "bundled"
    (directory / "scripts").mkdir()
    (directory / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "references").mkdir()
    (directory / "references" / "api.md").write_text("# API\n", encoding="utf-8")

    skill = SkillRegistry([tmp_path]).get("bundled")
    rendered = skill.render()
    assert "# Skill: bundled" in rendered
    assert "Follow the script." in rendered
    assert "run.sh" in rendered and "Bundled scripts" in rendered
    assert "api.md" in rendered and "Reference files" in rendered
    assert len(skill.scripts) == 1 and len(skill.references) == 1


def test_render_without_extras_omits_those_sections(tmp_path):
    make_skill(tmp_path, "plain")
    rendered = SkillRegistry([tmp_path]).get("plain").render()
    assert "Bundled scripts" not in rendered and "Reference files" not in rendered


def test_skill_to_dict(tmp_path):
    make_skill(tmp_path, "alpha")
    data = SkillRegistry([tmp_path]).get("alpha").to_dict()
    assert set(data) >= {"name", "description", "path", "scripts", "references"}


def test_skill_is_frozen(tmp_path):
    from dataclasses import FrozenInstanceError

    make_skill(tmp_path, "alpha")
    skill = SkillRegistry([tmp_path]).get("alpha")
    with pytest.raises(FrozenInstanceError):
        skill.name = "beta"  # type: ignore[misc]


# -- install -------------------------------------------------------------


def test_install_from_a_local_directory(tmp_path):
    source, target = tmp_path / "src", tmp_path / "installed"
    make_skill(source, "portable", "works anywhere")
    result = install(str(source / "portable"), target)
    assert result.installed == ("portable",)
    assert (target / "portable" / "SKILL.md").is_file()
    assert SkillRegistry([target]).get("portable").description == "works anywhere"


def test_install_finds_several_skills_in_one_repo(tmp_path):
    source, target = tmp_path / "repo", tmp_path / "installed"
    make_skill(source / "skills", "one")
    make_skill(source / "skills", "two")
    result = install(str(source), target)
    assert set(result.installed) == {"one", "two"}


def test_install_skips_an_existing_skill_unless_overwriting(tmp_path):
    source, target = tmp_path / "src", tmp_path / "installed"
    make_skill(source, "dup", "first version")
    install(str(source / "dup"), target)

    make_skill(source, "dup", "second version")
    with pytest.raises(SkillError, match="nothing installed"):
        install(str(source / "dup"), target)

    result = install(str(source / "dup"), target, overwrite=True)
    assert result.installed == ("dup",)
    assert SkillRegistry([target]).refresh().get("dup").description == "second version"


def test_install_from_a_directory_without_a_skill_file(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SkillError, match=r"no SKILL\.md"):
        install(str(empty), tmp_path / "target")


def test_install_accepts_a_skill_file_path_directly(tmp_path):
    source, target = tmp_path / "src", tmp_path / "installed"
    make_skill(source, "direct")
    result = install(str(source / "direct" / "SKILL.md"), target)
    assert result.installed == ("direct",)


def test_install_rejects_an_unrecognisable_source(tmp_path):
    with pytest.raises(SkillError, match="cannot work out"):
        install("not a url or a path!!", tmp_path)


def test_uninstall(tmp_path):
    source, target = tmp_path / "src", tmp_path / "installed"
    make_skill(source, "temporary")
    install(str(source / "temporary"), target)
    assert uninstall("temporary", target) is True
    assert uninstall("temporary", target) is False


def test_install_result_to_dict(tmp_path):
    source, target = tmp_path / "src", tmp_path / "installed"
    make_skill(source, "alpha")
    data = install(str(source / "alpha"), target).to_dict()
    assert data["installed"] == ["alpha"] and "destination" in data


# -- installation safety -------------------------------------------------


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", "..", ".", "a/b"])
def test_safe_name_sanitises_or_rejects(name):
    try:
        cleaned = _safe_name(name)
    except SkillError:
        return  # rejecting outright is fine
    assert "/" not in cleaned and cleaned not in ("", ".", "..")


def test_safe_name_keeps_ordinary_names():
    assert _safe_name("invoice-filing_v2") == "invoice-filing_v2"


@pytest.mark.parametrize("member", ["../../etc/passwd", "/etc/passwd", "a/../../../b"])
def test_archive_traversal_is_rejected(tmp_path, member):
    with pytest.raises(SkillError, match="unsafe path"):
        _safe_extract_path(tmp_path, member)


def test_ordinary_archive_member_is_allowed(tmp_path):
    _safe_extract_path(tmp_path, "skill/SKILL.md")  # must not raise


def test_sibling_directory_escape_is_rejected(tmp_path):
    """A prefix check would pass this: /dest/../dest-evil starts with /dest."""
    destination = tmp_path / "skills"
    destination.mkdir()
    with pytest.raises(SkillError, match="unsafe path"):
        _safe_extract_path(destination, "../skills-evil/payload.sh")


# -- tools ---------------------------------------------------------------


def test_skill_tools_list_load_and_install(tmp_path):
    from lai.config import Config
    from lai.skills.tools import register
    from lai.tools.base import ToolContext, ToolRegistry

    source = tmp_path / "src"
    make_skill(source, "helper", "use when helping", "Be helpful.")
    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True)

    registry = ToolRegistry()
    register(registry)
    skills = SkillRegistry([home / "skills"])
    ctx = ToolContext(config=Config(home=home), skills=skills)

    empty = registry.call("skill_list", {}, ctx)
    assert "No skill" in empty.content

    installed = registry.call("skill_install", {"source": str(source / "helper")}, ctx)
    assert installed.ok and "helper" in installed.content

    listed = registry.call("skill_list", {}, ctx)
    assert "helper" in listed.content

    loaded = registry.call("skill_load", {"name": "helper"}, ctx)
    assert "Be helpful." in loaded.content

    missing = registry.call("skill_load", {"name": "ghost"}, ctx)
    assert missing.ok is False


def test_skill_load_records_what_was_loaded(tmp_path):
    from lai.agent.session import Session
    from lai.config import Config
    from lai.skills.tools import register
    from lai.tools.base import ToolContext, ToolRegistry

    make_skill(tmp_path, "noted")
    registry = ToolRegistry()
    register(registry)
    session = Session()
    ctx = ToolContext(config=Config(home=tmp_path), skills=SkillRegistry([tmp_path]), session=session)
    registry.call("skill_load", {"name": "noted"}, ctx)
    assert session.metadata["loaded_skills"] == ["noted"]
