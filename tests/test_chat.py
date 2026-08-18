"""The chat interface and the backend switching behind it.

Two things carry the weight: a slash command must never be able to exist
without being discoverable, and switching backend must be verified before
anything is written to disk — a typo that persists an unusable provider would
lock the next start out of a working model.
"""

from __future__ import annotations

import pytest

from lai.chat import backends
from lai.chat.commands import COMMANDS, NEW, QUIT, Context, run
from lai.errors import ProviderError


class FakeProvider:
    def __init__(self, name="zai", model="glm-5"):
        self.name, self.model = name, model
        self.closed = False

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, tmp_path, provider=None):
        from lai.config import load_config

        self.config = load_config().with_overrides(home=tmp_path)
        self.provider = provider or FakeProvider()
        self.provider_error = ""
        self.registry = type("R", (), {"__len__": lambda self: 12, "specs": lambda self: []})()
        self.skills = type("S", (), {"__len__": lambda self: 3, "list": lambda self: [],
                                     "search": lambda self, q: []})()
        self.policy = type("P", (), {"config": self.config.safety})()
        self.desktop = None


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    """Never read the developer's own ~/.lai — these tests write config files."""
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    for name in ("LAI_MODE", "LAI_PROVIDER", "LAI_MODEL", "LAI_FALLBACK"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def runtime(tmp_path):
    return FakeRuntime(tmp_path)


# -- the command table ---------------------------------------------------


def test_every_command_is_dispatchable(runtime):
    """A command in the table that cannot run is worse than a missing one."""
    for name, spec in COMMANDS.items():
        assert callable(spec[0]), name
        assert isinstance(spec[1], str) and isinstance(spec[2], bool), name


def test_help_lists_the_visible_commands(runtime):
    text = run(Context(runtime=runtime), "/help")
    for name, spec in COMMANDS.items():
        if not spec[2]:
            assert f"/{name}" in text


def test_an_unknown_command_suggests_a_real_one(runtime):
    text = run(Context(runtime=runtime), "/moddel")
    assert "unknown command" in text
    assert "/mode" in text or "/model" in text


def test_quit_and_new_are_signalled_not_printed(runtime):
    ctx = Context(runtime=runtime)
    assert run(ctx, "/quit") == QUIT
    assert run(ctx, "/exit") == QUIT
    assert run(ctx, "/new") == NEW


def test_status_names_who_is_answering(runtime):
    text = run(Context(runtime=runtime), "/status")
    assert "zai/glm-5" in text and "tools" in text


def test_status_reports_a_backend_that_stepped_aside(runtime, tmp_path):
    runtime.provider = type("P", (), {
        "name": "ollama", "model": "qwen", "chain": ["zai", "ollama"],
        "failures": {"zai": "HTTP 429 Usage limit reached"},
    })()
    text = run(Context(runtime=runtime), "/status")
    assert "stepped aside" in text and "429" in text
    assert "standby: ollama" in text


# -- mode ----------------------------------------------------------------


def test_mode_changes_and_persists(runtime, tmp_path):
    from lai import config_file

    assert "yolo" in run(Context(runtime=runtime), "/mode yolo")
    assert runtime.config.safety.mode == "yolo"
    assert runtime.policy.config.mode == "yolo", "the live policy must change too"
    assert config_file.read(tmp_path)["safety"]["mode"] == "yolo"


def test_an_invalid_mode_is_refused(runtime):
    assert "must be one of" in run(Context(runtime=runtime), "/mode banana")


def test_mode_can_be_picked_from_a_menu(runtime):
    ctx = Context(runtime=runtime, ask=lambda question, options: 3)
    assert "yolo" in run(ctx, "/mode")


def test_a_cancelled_menu_changes_nothing(runtime):
    before = runtime.config.safety.mode
    ctx = Context(runtime=runtime, ask=lambda question, options: -1)
    assert "unchanged" in run(ctx, "/mode")
    assert runtime.config.safety.mode == before


# -- fallback ------------------------------------------------------------


def test_fallback_can_be_set_listed_and_turned_off(runtime, tmp_path):
    from lai import config_file

    text = run(Context(runtime=runtime), "/fallback cli:claude ollama")
    assert "cli:claude" in text and "ollama" in text
    assert runtime.config.provider.fallback == ("cli:claude", "ollama")
    assert config_file.read(tmp_path)["provider"]["fallback"] == ["cli:claude", "ollama"]

    assert "off" in run(Context(runtime=runtime), "/fallback off")
    assert runtime.config.provider.fallback == ()

    assert "auto" in run(Context(runtime=runtime), "/fallback auto")
    assert runtime.config.provider.fallback == ("auto",)


def test_fallback_with_no_argument_reports_the_chain(runtime):
    backends.set_fallback(runtime, ["ollama"], persist=False)
    assert "ollama" in run(Context(runtime=runtime), "/fallback")


def test_a_duplicate_in_the_chain_is_collapsed(runtime):
    assert backends.set_fallback(runtime, ["ollama", "ollama", "zai"], persist=False) == ["ollama", "zai"]


# -- switching backend ---------------------------------------------------


def test_switching_verifies_before_it_persists(runtime, tmp_path, monkeypatch):
    """A backend that cannot be built must not become the saved default."""
    from lai import config_file

    def refuse(config, **kwargs):
        raise ProviderError("openai: no API key configured")

    monkeypatch.setattr("lai.agent.providers.registry.build_provider", refuse)
    with pytest.raises(ProviderError):
        backends.use(runtime, "openai")
    assert config_file.read(tmp_path) == {}, "nothing may be written until it works"
    assert runtime.provider.name == "zai", "the working backend must survive a failed switch"


def test_a_successful_switch_replaces_and_closes_the_old_one(runtime, tmp_path, monkeypatch):
    from lai import config_file

    previous = runtime.provider
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: FakeProvider(name=config.name, model="m"),
    )
    label = backends.use(runtime, "ollama")
    assert label == "ollama/m"
    assert runtime.provider.name == "ollama"
    assert previous.closed, "the replaced backend must not leak its connections"
    assert config_file.read(tmp_path)["provider"]["name"] == "ollama"


def test_switching_never_carries_the_old_key_across(runtime, monkeypatch):
    seen = {}
    runtime.config = runtime.config.with_overrides(
        provider=runtime.config.provider.__class__(name="zai", api_key="secret", model="glm-5")
    )
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: seen.update(key=config.api_key, model=config.model) or FakeProvider(),
    )
    backends.use(runtime, "openai", persist=False)
    assert seen == {"key": "", "model": ""}


def test_describe_reports_the_live_chain(runtime):
    runtime.provider = type("P", (), {
        "name": "ollama", "model": "q", "chain": ["zai", "ollama"], "failures": {"zai": "429"},
    })()
    info = backends.describe(runtime)
    assert info["chain"] == ["zai", "ollama"] and info["failures"] == {"zai": "429"}


# -- the status line -----------------------------------------------------


def test_the_status_line_shows_who_and_what_stands_by(runtime):
    from lai.chat.repl import status_line

    runtime.provider = type("P", (), {
        "name": "zai", "model": "glm-5", "chain": ["zai", "ollama", "cli:claude"], "failures": {},
    })()
    line = status_line(runtime)
    assert "zai/glm-5" in line and "ask" in line and "ollama" in line and "+1" in line


# -- the learning journal in the chat -------------------------------------


@pytest.fixture
def learning_runtime(tmp_path):
    from lai.knowledge import Journal

    runtime = FakeRuntime(tmp_path)
    runtime.journal = Journal.open(tmp_path)
    return runtime


def test_learn_writes_a_note(learning_runtime, tmp_path):
    text = run(Context(runtime=learning_runtime), "/learn drawing: the canvas starts at y=140")
    assert "noted" in text
    assert "y=140" in learning_runtime.journal.get("drawing").body
    assert (tmp_path / "notes" / "drawing.md").is_file()


def test_learn_without_a_lesson_explains_itself(learning_runtime):
    assert "usage" in run(Context(runtime=learning_runtime), "/learn drawing")


def test_notes_lists_what_it_knows(learning_runtime):
    learning_runtime.journal.write("drawing", "- the canvas starts at y=140")
    text = run(Context(runtime=learning_runtime), "/notes")
    assert "drawing" in text and "y=140" in text


def test_notes_with_a_name_shows_that_note(learning_runtime):
    learning_runtime.journal.write("drawing", "- the canvas starts at y=140", title="Drawing")
    text = run(Context(runtime=learning_runtime), "/notes drawing")
    assert "Drawing" in text and "y=140" in text


def test_an_empty_journal_says_how_to_fill_it(learning_runtime):
    assert "/learn" in run(Context(runtime=learning_runtime), "/notes")


def test_forget_removes_a_note(learning_runtime):
    learning_runtime.journal.write("wrong", "- nonsense")
    assert "forgot" in run(Context(runtime=learning_runtime), "/forget wrong")
    assert learning_runtime.journal.get("wrong") is None


def test_forgetting_something_absent_says_so(learning_runtime):
    assert "no note" in run(Context(runtime=learning_runtime), "/forget nope")


def test_learning_can_be_switched_off_and_persists(learning_runtime, tmp_path):
    from lai import config_file

    assert "off" in run(Context(runtime=learning_runtime), "/learning off")
    assert learning_runtime.config.learning.enabled is False
    assert config_file.read(tmp_path)["learning"]["enabled"] is False
    assert "on" in run(Context(runtime=learning_runtime), "/learning on")
    assert learning_runtime.config.learning.enabled is True


def test_learning_with_no_argument_reports_the_state(learning_runtime):
    assert "learning is" in run(Context(runtime=learning_runtime), "/learning")


# -- the settings page ---------------------------------------------------


def test_settings_shows_everything_that_can_be_changed(learning_runtime):
    learning_runtime.journal.write("a", "- x")
    text = run(Context(runtime=learning_runtime), "/settings")
    for label in ("model", "failover", "permissions", "learning", "tools", "skills", "config"):
        assert label in text
    assert "1 note(s)" in text


def test_settings_names_the_command_that_changes_each_thing(learning_runtime):
    text = run(Context(runtime=learning_runtime), "/settings")
    assert "/model" in text and "/mode" in text and "/fallback" in text


# -- editing ---------------------------------------------------------------


def test_edit_saves_what_the_editor_wrote(learning_runtime, monkeypatch, tmp_path):
    from lai.chat import notes as notes_module

    def fake_editor(command, **kwargs):
        path = command[-1]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("- edited by a human\n")
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(notes_module, "editor_command", lambda: ["fake-editor"])
    monkeypatch.setattr(notes_module.subprocess, "run", fake_editor)
    learning_runtime.journal.write("drawing", "- what the agent believed")

    assert "saved" in run(Context(runtime=learning_runtime), "/edit drawing")
    assert learning_runtime.journal.get("drawing").body == "- edited by a human"


def test_emptying_a_note_in_the_editor_deletes_it(learning_runtime, monkeypatch):
    from lai.chat import notes as notes_module

    def empty_it(command, **kwargs):
        with open(command[-1], "w", encoding="utf-8") as fh:
            fh.write("\n")
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(notes_module, "editor_command", lambda: ["fake-editor"])
    monkeypatch.setattr(notes_module.subprocess, "run", empty_it)
    learning_runtime.journal.write("wrong", "- nonsense")

    assert "deleted" in run(Context(runtime=learning_runtime), "/edit wrong")
    assert learning_runtime.journal.get("wrong") is None


def test_an_unchanged_edit_changes_nothing(learning_runtime, monkeypatch):
    from lai.chat import notes as notes_module

    monkeypatch.setattr(notes_module, "editor_command", lambda: ["fake-editor"])
    monkeypatch.setattr(notes_module.subprocess, "run",
                        lambda command, **kw: type("P", (), {"returncode": 0})())
    learning_runtime.journal.write("drawing", "- original")

    assert "unchanged" in run(Context(runtime=learning_runtime), "/edit drawing")
    assert learning_runtime.journal.get("drawing").body == "- original"


def test_edit_without_an_editor_says_where_the_files_are(learning_runtime, monkeypatch):
    from lai.chat import notes as notes_module

    monkeypatch.setattr(notes_module, "editor_command", list)
    text = run(Context(runtime=learning_runtime), "/edit drawing")
    assert "EDITOR" in text and "~/.lai/notes" in text
