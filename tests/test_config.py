"""Configuration: defaults, file, environment, precedence, immutability."""

from __future__ import annotations

from pathlib import Path

import pytest

from lai.config import (
    Config,
    DesktopConfig,
    LimitsConfig,
    ProviderConfig,
    SafetyConfig,
    _bool,
    _tuple,
    load_config,
)
from lai.errors import ConfigError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Isolate every test from the developer's real environment."""
    for name in (
        "LAI_HOME", "LAI_PROVIDER", "LAI_MODEL", "LAI_BASE_URL", "LAI_API_KEY",
        "LAI_MODE", "LAI_MAX_STEPS", "LAI_MAX_SECONDS", "LAI_MAX_TOKENS",
        "LAI_MAX_EDGE", "LAI_DRY_RUN", "LAI_THINKING", "LAI_LOG_LEVEL", "LAI_DISPLAY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAI_HOME", str(tmp_path / "home"))


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_are_sane():
    config = load_config()
    assert config.safety.mode == "ask"
    assert config.provider.name == "auto"
    assert config.limits.max_steps > 0
    assert config.desktop.max_edge >= 320


def test_missing_config_file_is_fine(tmp_path):
    assert load_config(tmp_path / "nope.toml").safety.mode == "ask"


def test_full_config_file_is_read(tmp_path):
    path = write_config(tmp_path, """
log_level = "debug"

[provider]
name = "anthropic"
model = "claude-sonnet-4-5"
max_tokens = 4096
temperature = 0.5
thinking_budget = 2000

[safety]
mode = "auto"
allow_tools = ["ui_click"]
deny_tools = ["shell_exec"]
max_actions_per_minute = 30
dry_run = true

[desktop]
max_edge = 900
max_elements = 50

[limits]
max_steps = 12
max_seconds = 300.0
""")
    config = load_config(path)
    assert config.provider.name == "anthropic"
    assert config.provider.model == "claude-sonnet-4-5"
    assert config.provider.max_tokens == 4096
    assert config.provider.thinking_budget == 2000
    assert config.safety.mode == "auto"
    assert config.safety.allow_tools == ("ui_click",)
    assert config.safety.deny_tools == ("shell_exec",)
    assert config.safety.max_actions_per_minute == 30
    assert config.safety.dry_run is True
    assert config.desktop.max_edge == 900
    assert config.limits.max_steps == 12
    assert config.log_level == "debug"


def test_defaults_survive_a_partial_file(tmp_path):
    path = write_config(tmp_path, "[safety]\nmode = 'yolo'\n")
    config = load_config(path)
    assert config.safety.mode == "yolo"
    assert config.safety.protected_apps  # defaults preserved
    assert config.safety.deny_shell_patterns


def test_env_overrides_the_file(tmp_path, monkeypatch):
    path = write_config(tmp_path, "[safety]\nmode = 'readonly'\n[provider]\nmodel = 'from-file'\n")
    monkeypatch.setenv("LAI_MODE", "yolo")
    monkeypatch.setenv("LAI_MODEL", "from-env")
    monkeypatch.setenv("LAI_MAX_STEPS", "99")
    config = load_config(path)
    assert config.safety.mode == "yolo"
    assert config.provider.model == "from-env"
    assert config.limits.max_steps == 99


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
     ("0", False), ("false", False), ("no", False), ("", False)],
)
def test_dry_run_env_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("LAI_DRY_RUN", value)
    assert load_config().safety.dry_run is expected


def test_bool_helper_falls_back_when_unset():
    assert _bool(None, True) is True
    assert _bool(None, False) is False


def test_invalid_mode_raises(tmp_path):
    with pytest.raises(ConfigError, match=r"safety\.mode"):
        load_config(write_config(tmp_path, "[safety]\nmode = 'banana'\n"))


def test_non_table_section_raises(tmp_path):
    with pytest.raises(ConfigError, match=r"\[safety\]"):
        load_config(write_config(tmp_path, "safety = 'yes'\n"))


def test_malformed_toml_raises(tmp_path):
    with pytest.raises(ConfigError, match="cannot parse"):
        load_config(write_config(tmp_path, "this is not = = toml ["))


def test_tuple_helper():
    assert _tuple(None, ("a",)) == ("a",)
    assert _tuple("x, y ,z", ()) == ("x", "y", "z")
    assert _tuple(["a", 2], ()) == ("a", "2")
    assert _tuple("", ()) == ()
    with pytest.raises(ConfigError):
        _tuple(42, ())


def test_skill_paths_expand_and_deduplicate(tmp_path):
    config = Config(home=tmp_path / "h", skill_paths=(
        "{home}/skills", "{cwd}/.lai/skills", "~/x", "{home}/skills",
    ))
    paths = config.resolved_skill_paths(cwd=tmp_path / "work")
    assert paths[0] == tmp_path / "h" / "skills"
    assert paths[1] == tmp_path / "work" / ".lai" / "skills"
    assert str(paths[2]).startswith(str(Path.home()))
    assert len(paths) == 3  # duplicate collapsed


def test_mcp_paths_resolve(tmp_path):
    paths = Config(home=tmp_path).resolved_mcp_paths(cwd=tmp_path)
    assert any(p.name == "mcp.json" for p in paths)
    assert any(p.name == ".mcp.json" for p in paths)


def test_derived_directories_and_ensure(tmp_path):
    config = Config(home=tmp_path / "lai")
    assert config.sessions_dir == tmp_path / "lai" / "sessions"
    config.ensure_dirs()
    for directory in (config.home, config.sessions_dir, config.logs_dir, config.skills_dir, config.artifacts_dir):
        assert directory.is_dir()
    config.ensure_dirs()  # idempotent


def test_with_overrides_returns_a_new_object():
    original = Config()
    changed = original.with_overrides(log_level="debug")
    assert original.log_level != "debug"
    assert changed.log_level == "debug"
    assert changed is not original


@pytest.mark.parametrize(
    ("obj", "field", "value"),
    [
        (Config(), "log_level", "debug"),
        (ProviderConfig(), "model", "x"),
        (SafetyConfig(), "mode", "yolo"),
        (DesktopConfig(), "max_edge", 100),
        (LimitsConfig(), "max_steps", 1),
    ],
)
def test_config_objects_are_frozen(obj, field, value):
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        setattr(obj, field, value)


def test_redacted_never_leaks_the_key():
    config = Config(provider=ProviderConfig(api_key="sk-ant-supersecret", model="m"))
    dumped = config.redacted()
    assert dumped["provider"]["api_key"] == "set"
    assert "supersecret" not in str(dumped)


def test_redacted_reports_an_unset_key():
    assert Config().redacted()["provider"]["api_key"] == "unset"


def test_lai_home_env_moves_everything(monkeypatch, tmp_path):
    monkeypatch.setenv("LAI_HOME", str(tmp_path / "custom"))
    config = load_config()
    assert config.home == tmp_path / "custom"
    assert config.sessions_dir.parent == tmp_path / "custom"


# -- whose screen the agent works on -------------------------------------


def test_the_agent_gets_its_own_screen_unless_told_otherwise():
    from lai.config import DesktopConfig

    assert DesktopConfig().own_display == "auto"
    assert DesktopConfig().watch is True, "an agent you cannot see is one you cannot supervise"


def test_an_unknown_screen_mode_is_refused():
    from lai.config import ConfigError, DesktopConfig

    with pytest.raises(ConfigError, match="own_display"):
        DesktopConfig(own_display="sometimes")


def test_the_screen_settings_are_read_from_the_file(tmp_path):
    from lai.config import load_config

    path = tmp_path / "config.toml"
    path.write_text('[desktop]\nown_display = "never"\nwatch = false\n', encoding="utf-8")
    config = load_config(path)
    assert config.desktop.own_display == "never"
    assert config.desktop.watch is False


def test_how_tools_are_offered_is_configurable():
    """Hermes, Qwen and most local servers were trained to be asked in the
    prompt, not through a function-calling API they do not have."""
    from lai.config import ProviderConfig

    assert ProviderConfig().tool_dialect == "auto"

    from lai.agent.providers import registry

    provider = registry._instantiate(
        "openai", ProviderConfig(name="openai", api_key="k", model="m", tool_dialect="text"), None
    )
    assert provider.tool_dialect == "text"
