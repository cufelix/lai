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
    def __init__(self, name="zai", model="glm-5", answer="OK"):
        self.name, self.model = name, model
        self.answer = answer
        self.closed = False

    def complete(self, messages, **kwargs):
        from lai.agent.providers.base import Message, TextBlock, TurnResult, Usage

        return TurnResult(
            message=Message("assistant", [TextBlock(self.answer)]),
            stop_reason="end_turn", usage=Usage(), model=self.model,
        )

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
        self.extra: dict = {}


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


# -- the rest of the command surface -------------------------------------


def backend(name="zai", model="glm-5", detail="via key", resting=""):
    from lai.models import READY, Backend

    return Backend(name=name, label=name, kind="api", status=READY,
                   detail=detail, model=model, resting=resting)


def test_model_with_a_name_switches_directly(runtime, monkeypatch):
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: FakeProvider(name=config.name, model=config.model or "m"),
    )
    assert "ollama/big" in run(Context(runtime=runtime), "/model ollama big")


def test_the_model_menu_marks_a_resting_backend(runtime, monkeypatch):
    """Offering an exhausted key as a choice without saying so wastes a turn."""
    monkeypatch.setattr(
        "lai.chat.backends.catalogue",
        lambda rt=None, **kw: [backend(), backend(name="openai", resting="out of quota, retry in 40 min")],
    )
    seen: dict = {}
    ctx = Context(runtime=runtime, ask=lambda question, options: seen.setdefault("options", options) and -1)
    run(ctx, "/model")
    assert any("out of quota" in option for option in seen["options"])


def test_the_model_list_without_a_menu_still_names_them(runtime, monkeypatch):
    monkeypatch.setattr("lai.chat.backends.catalogue", lambda rt=None, **kw: [backend()])
    assert "zai" in run(Context(runtime=runtime), "/model")


def test_no_ready_backend_points_at_setup(runtime, monkeypatch):
    monkeypatch.setattr("lai.chat.backends.catalogue", lambda rt=None, **kw: [])
    assert "lai setup" in run(Context(runtime=runtime), "/model")


def test_tools_lists_and_filters(runtime):
    from lai.safety.policy import Risk

    spec = type("S", (), {"name": "window_list", "risk": Risk.READ, "description": "List windows."})()
    other = type("S", (), {"name": "shell_exec", "risk": Risk.DESTRUCTIVE, "description": "Run a command."})()
    runtime.registry = type("R", (), {"__len__": lambda self: 2,
                                      "specs": lambda self: [spec, other]})()
    assert "window_list" in run(Context(runtime=runtime), "/tools")
    filtered = run(Context(runtime=runtime), "/tools shell")
    assert "shell_exec" in filtered and "window_list" not in filtered


def test_skills_lists_and_searches(runtime):
    skill = type("S", (), {"name": "invoicing", "description": "File invoices."})()
    runtime.skills = type("S", (), {"__len__": lambda self: 1, "list": lambda self: [skill],
                                    "search": lambda self, q: [skill] if q in skill.name else []})()
    assert "invoicing" in run(Context(runtime=runtime), "/skills")
    assert "invoicing" in run(Context(runtime=runtime), "/skills invoic")
    assert "0 skill" in run(Context(runtime=runtime), "/skills nothing")


def test_observe_reports_what_it_sees(runtime):
    runtime.desktop = type("D", (), {
        "observe": lambda self, **kw: type("O", (), {"summary": lambda self: "FOCUSED: 'Calculator'"})()
    })()
    assert "Calculator" in run(Context(runtime=runtime), "/observe")


def test_doctor_reports_and_says_whether_it_is_ready(runtime, monkeypatch):
    from lai.checks import FAIL, OK, Check, Report

    monkeypatch.setattr(
        "lai.checks.run_checks",
        lambda rt=None, **kw: Report([Check("a", "display server", OK, "x11"),
                                      Check("b", "input", FAIL, "xdotool missing")]),
    )
    text = run(Context(runtime=runtime), "/doctor")
    assert "display server" in text and "xdotool missing" in text
    assert "not ready" in text


def test_session_reports_the_current_conversation(runtime):
    from lai.agent.session import Session

    runtime.extra = {"chat_session": Session()}
    assert "messages" in run(Context(runtime=runtime), "/session")


def test_web_points_at_the_command_that_opens_it(runtime):
    assert "lai web" in run(Context(runtime=runtime), "/web")


# -- adding a key and picking a model without leaving the chat -----------


def test_a_key_is_verified_before_it_is_saved(runtime, tmp_path, monkeypatch):
    """A key you find out is wrong during your next task is worse than one
    that never got saved."""
    from lai import config_file

    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: FakeProvider(name=config.name, model="glm-5.2"),
    )
    text = run(Context(runtime=runtime), "/key openrouter sk-or-v1-testkey")
    assert "saved and verified" in text
    saved = config_file.read(tmp_path)["provider"]
    assert saved["name"] == "openrouter" and saved["api_key"] == "sk-or-v1-testkey"


def test_a_key_that_does_not_work_is_not_saved(runtime, tmp_path, monkeypatch):
    from lai import config_file

    def refuse(config, **kwargs):
        raise ProviderError("openrouter: HTTP 401 invalid api key")

    monkeypatch.setattr("lai.agent.providers.registry.build_provider", refuse)
    text = run(Context(runtime=runtime), "/key openrouter sk-or-wrong")
    assert "401" in text
    assert config_file.read(tmp_path) == {}, "nothing may be written until it works"


def test_a_model_can_be_pinned_with_the_key(runtime, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: seen.setdefault("model", config.model) or FakeProvider(
            name=config.name, model=config.model or "default"
        ),
    )
    run(Context(runtime=runtime), "/key openrouter sk-or-v1-x z-ai/glm-5.2:free")
    assert seen["model"] == "z-ai/glm-5.2:free"


def test_the_key_command_explains_itself(runtime):
    assert "usage" in run(Context(runtime=runtime), "/key openrouter")


def test_models_lists_what_a_backend_serves(runtime, monkeypatch):
    from lai.agent.providers.listing import ModelInfo

    monkeypatch.setattr(
        "lai.models.available_models",
        lambda name, **kwargs: [ModelInfo(id="z-ai/glm-5.2:free", context=256000, free=True)],
    )
    text = run(Context(runtime=runtime), "/models openrouter glm")
    assert "z-ai/glm-5.2:free" in text and "free" in text


def test_models_offers_a_menu_when_there_is_one(runtime, monkeypatch):
    from lai.agent.providers.listing import ModelInfo

    monkeypatch.setattr(
        "lai.models.available_models",
        lambda name, **kwargs: [ModelInfo(id="a/one"), ModelInfo(id="b/two")],
    )
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: FakeProvider(name=config.name, model=config.model),
    )
    ctx = Context(runtime=runtime, ask=lambda question, options: 1)
    assert "b/two" in run(ctx, "/models openrouter")


def test_an_unknown_backend_says_so(runtime, monkeypatch):
    def missing(name, **kwargs):
        raise LookupError(name)

    monkeypatch.setattr("lai.models.available_models", missing)
    assert "unknown backend" in run(Context(runtime=runtime), "/models nonsense")


def test_a_search_matching_nothing_says_so(runtime, monkeypatch):
    from lai.agent.providers.listing import ModelInfo

    monkeypatch.setattr("lai.models.available_models",
                        lambda name, **kwargs: [ModelInfo(id="a/one")])
    assert "nothing matching" in run(Context(runtime=runtime), "/models openrouter zzz")


# -- discovering and adding a backend you do not have yet ----------------


def guided(runtime, *, choices=(), secrets=(), confirms=(), said=None):
    """A Context that answers a guided flow the way a person would."""
    picks, keys, yesses = list(choices), list(secrets), list(confirms)
    return Context(
        runtime=runtime,
        ask=lambda question, options: picks.pop(0) if picks else -1,
        secret=lambda question: keys.pop(0) if keys else "",
        confirm=lambda question: yesses.pop(0) if yesses else False,
        say=(said.append if said is not None else None),
    )


def catalogue_of(*backends_):
    return lambda runtime=None, **kwargs: list(backends_)


def a_backend(name, status="ready", **kwargs):
    from lai.models import Backend

    kwargs.setdefault("label", name)
    kwargs.setdefault("kind", "api")
    kwargs.setdefault("model", "some-model")
    kwargs.setdefault("detail", "detail")
    return Backend(name=name, status=status, **kwargs)


def test_a_backend_you_have_not_set_up_is_still_offered(runtime, monkeypatch):
    """A choice you cannot see is a choice you do not have — and somebody who
    just installed this has no key for anything."""
    monkeypatch.setattr(
        "lai.chat.backends.catalogue",
        catalogue_of(a_backend("ollama"), a_backend("openrouter", status="known",
                                                    label="OpenRouter — one key, most models")),
    )
    seen: dict = {}
    ctx = Context(runtime=runtime, ask=lambda q, options: seen.setdefault("options", options) and -1)
    run(ctx, "/model")
    assert any("OpenRouter" in option for option in seen["options"])
    assert any(option.startswith("+") for option in seen["options"]), "marked as needing setup"


def test_choosing_one_walks_through_key_then_model(runtime, monkeypatch):
    from lai.agent.providers.listing import ModelInfo

    monkeypatch.setattr(
        "lai.chat.backends.catalogue",
        catalogue_of(a_backend("openrouter", status="known", signup="https://openrouter.ai/keys")),
    )
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: FakeProvider(name=config.name, model=config.model or "default"),
    )
    monkeypatch.setattr(
        "lai.models.available_models",
        lambda name, **kwargs: [ModelInfo(id="z-ai/glm-5.2:free", free=True),
                                ModelInfo(id="openai/gpt-4o", prompt_price=2.5)],
    )
    said: list = []
    ctx = guided(
        runtime,
        choices=[0, 1],            # the backend, then the second model
        secrets=["sk-or-v1-test"],
        confirms=[False, True],    # no browser, yes to picking a model
        said=said,
    )
    result = run(ctx, "/model")
    assert "openai/gpt-4o" in result
    assert any("openrouter.ai/keys" in line for line in said), "it says where to get one"


def test_an_empty_paste_changes_nothing(runtime, monkeypatch):
    monkeypatch.setattr(
        "lai.chat.backends.catalogue",
        catalogue_of(a_backend("openrouter", status="known")),
    )
    ctx = guided(runtime, choices=[0], secrets=[""], confirms=[False])
    assert "unchanged" in run(ctx, "/model")


def test_a_key_that_does_not_work_says_so_and_saves_nothing(runtime, tmp_path, monkeypatch):
    from lai import config_file

    monkeypatch.setattr(
        "lai.chat.backends.catalogue",
        catalogue_of(a_backend("openrouter", status="known")),
    )

    def refuse(config, **kwargs):
        raise ProviderError("HTTP 401 invalid api key")

    monkeypatch.setattr("lai.agent.providers.registry.build_provider", refuse)
    ctx = guided(runtime, choices=[0], secrets=["sk-or-wrong"], confirms=[False])
    assert "did not work" in run(ctx, "/model")
    assert config_file.read(tmp_path) == {}


def test_without_a_way_to_ask_it_names_the_command_instead(runtime, monkeypatch):
    """Piped input has no secret prompt; the flow must still be reachable."""
    monkeypatch.setattr(
        "lai.chat.backends.catalogue",
        catalogue_of(a_backend("openrouter", status="known")),
    )
    ctx = Context(runtime=runtime, ask=lambda q, options: 0)
    assert "/key openrouter" in run(ctx, "/model")


def test_switching_model_keeps_the_key_that_was_just_verified(runtime, monkeypatch):
    """Changing which OpenRouter model to use must not throw away OpenRouter's
    key — that is how the guided flow used to fail one step after succeeding."""
    from dataclasses import replace

    runtime.config = runtime.config.with_overrides(
        provider=replace(runtime.config.provider, name="openrouter", api_key="sk-or-live")
    )
    seen: dict = {}
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: seen.update(key=config.api_key) or FakeProvider(
            name=config.name, model=config.model
        ),
    )
    backends.use(runtime, "openrouter", model="some/model", persist=False)
    assert seen["key"] == "sk-or-live"


def test_switching_to_a_different_backend_still_drops_the_key(runtime, monkeypatch):
    """Carrying z.ai's key across to OpenAI would be nonsense."""
    from dataclasses import replace

    runtime.config = runtime.config.with_overrides(
        provider=replace(runtime.config.provider, name="zai", api_key="zai-secret")
    )
    seen: dict = {}
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: seen.update(key=config.api_key) or FakeProvider(name=config.name),
    )
    backends.use(runtime, "openai", persist=False)
    assert seen["key"] == ""


def test_a_key_is_verified_against_that_backend_alone(runtime, monkeypatch):
    """With failover on, a standby would answer and any key would look valid."""
    seen: dict = {}
    monkeypatch.setattr(
        "lai.agent.providers.registry.build_provider",
        lambda config, **kwargs: seen.setdefault("fallback", config.fallback) or FakeProvider(
            name=config.name, model="m"
        ),
    )
    backends.set_key(runtime, "openrouter", "sk-or-v1-x", persist=False)
    assert seen["fallback"] == (), "verification must not be answerable by a standby"
