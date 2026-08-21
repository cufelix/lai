"""Config writing and the setup wizard.

Two properties carry the weight here: the wizard must never write a
configuration it has not verified, and it must be safe to run twice — a second
pass must not clobber what the first one set up.
"""

from __future__ import annotations

import stat

import pytest

from lai import config_file
from lai.checks import FAIL, OK, WARN, Check, Fix, Report
from lai.config import load_config
from lai.setup_wizard import Answers, Prompt, _plausible, _repair, _setup_mode, needs_setup, run_setup


class FakeOut:
    """Captures what the wizard would print."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str = "", **kwargs) -> None:
        self.lines.append(str(text))

    def rule(self, title: str = "") -> None:
        self.lines.append("---")

    def raw(self, text: str) -> None:
        self.lines.append(str(text))

    def error(self, text: str) -> None:
        self.lines.append(f"error: {text}")

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class ScriptedPrompt(Prompt):
    """A prompt with answers queued up, for driving the wizard in a test."""

    def __init__(self, *, confirms=None, choices=None, secrets=None) -> None:
        super().__init__(interactive=True)
        self.confirms = list(confirms or [])
        self.choices = list(choices or [])
        self.secrets = list(secrets or [])
        self.asked: list[str] = []

    def confirm(self, question: str, *, default: bool = True) -> bool:
        self.asked.append(question)
        return self.confirms.pop(0) if self.confirms else default

    def choose(self, question: str, options: list[str], *, default: int = 0) -> int:
        self.asked.append(question)
        return self.choices.pop(0) if self.choices else default

    def secret(self, question: str) -> str:
        self.asked.append(question)
        return self.secrets.pop(0) if self.secrets else ""


def no_backends(monkeypatch):
    """Isolate the provider step: nothing detected, nothing probed."""
    monkeypatch.setattr("lai.models.discover", lambda **kwargs: [])
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)


def fake_backend(name="zai", model="glm-5", kind="api", detail="via ZAI_API_KEY"):
    from lai.models import READY, Backend

    return Backend(name=name, label=name, kind=kind, status=READY, detail=detail, model=model)


# -- config_file ---------------------------------------------------------


def test_render_emits_only_what_was_chosen():
    body = config_file.render({"safety": {"mode": "auto"}})
    assert '[safety]' in body and 'mode = "auto"' in body
    assert "[provider]" not in body, "an unset section must stay out of the file"


def test_render_skips_empty_values(tmp_path):
    import tomllib

    config_file.write(tmp_path, {"provider": {"name": "zai", "api_key": "", "model": None}})
    parsed = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert parsed["provider"] == {"name": "zai"}, "empty values must not be written at all"


def test_render_types():
    body = config_file.render({"limits": {"max_steps": 40, "max_seconds": 900.0},
                               "safety": {"dry_run": True},
                               "channels": {"enabled": ["telegram", "webhook"]}})
    assert "max_steps = 40" in body
    assert "max_seconds = 900.0" in body
    assert "dry_run = true" in body
    assert 'enabled = ["telegram", "webhook"]' in body


def test_render_escapes_quotes():
    body = config_file.render({"provider": {"model": 'we"ird'}})
    assert r'model = "we\"ird"' in body


def test_render_is_valid_toml_and_round_trips(tmp_path):
    import tomllib

    settings = {
        "provider": {"name": "anthropic", "model": "claude-sonnet-4-5"},
        "safety": {"mode": "ask"},
        "limits": {"max_steps": 30},
        "channels": {"enabled": ["telegram"], "telegram": {"token": "123:abc"}},
    }
    config_file.write(tmp_path, settings)
    parsed = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert parsed["provider"]["name"] == "anthropic"
    assert parsed["safety"]["mode"] == "ask"
    assert parsed["channels"]["enabled"] == ["telegram"]
    assert parsed["channels"]["telegram"]["token"] == "123:abc"


def test_written_config_is_owner_only(tmp_path):
    """It can hold an API key, so nobody else on the machine may read it."""
    path = config_file.write(tmp_path, {"provider": {"api_key": "sk-secret"}})
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & 0o077 == 0, f"config.toml is group/world readable: {oct(mode)}"


def test_write_leaves_no_temp_files_behind(tmp_path):
    config_file.write(tmp_path, {"safety": {"mode": "ask"}})
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".config-")]
    assert leftovers == []


def test_write_replaces_the_previous_file(tmp_path):
    config_file.write(tmp_path, {"safety": {"mode": "ask"}})
    config_file.write(tmp_path, {"safety": {"mode": "yolo"}})
    assert config_file.read(tmp_path)["safety"]["mode"] == "yolo"


def test_read_of_a_missing_or_broken_file_is_empty(tmp_path):
    assert config_file.read(tmp_path) == {}
    (tmp_path / "config.toml").write_text("this is not toml {{{", encoding="utf-8")
    assert config_file.read(tmp_path) == {}


def test_merge_is_deep_and_non_mutating():
    existing = {"provider": {"name": "zai", "model": "glm"}, "safety": {"mode": "ask"}}
    updates = {"provider": {"model": "glm-5"}}
    merged = config_file.merge(existing, updates)
    assert merged["provider"] == {"name": "zai", "model": "glm-5"}
    assert merged["safety"] == {"mode": "ask"}
    assert existing["provider"]["model"] == "glm", "the input must not be mutated"


# -- Prompt --------------------------------------------------------------


def test_prompt_without_a_tty_takes_defaults():
    prompt = Prompt(interactive=False)
    assert prompt.confirm("really?", default=True) is True
    assert prompt.confirm("really?", default=False) is False
    assert prompt.choose("pick", ["a", "b"], default=1) == 1
    assert prompt.secret("key?") == ""


def test_assume_yes_takes_the_default_rather_than_always_yes():
    """--yes must not flip a question whose safe answer is no."""
    prompt = Prompt(assume_yes=True, interactive=True)
    assert prompt.confirm("install this?", default=True) is True
    assert prompt.confirm("switch backends?", default=False) is False


@pytest.mark.parametrize(
    ("provider", "key", "expected"),
    [
        ("anthropic", "sk-ant-abcdefghijklmnop", True),
        ("anthropic", "glm-abcdefghijklmnop", False),
        ("openai", "sk-abcdefghijklmnop", True),
        ("openrouter", "sk-or-abcdefghijklmnop", True),
        ("openrouter", "sk-abcdefghijklmnop", False),
        ("zai", "any-long-token-here", True),
        ("anthropic", "short", False),
        ("zai", "has a space in it", False),
    ],
)
def test_key_plausibility_catches_an_obvious_mispaste(provider, key, expected):
    assert _plausible(provider, key) is expected


# -- repair step ---------------------------------------------------------


def test_repair_runs_an_accepted_fix():
    answers = Answers(fixed=[], skipped=[])
    out, prompt = FakeOut(), ScriptedPrompt(confirms=[True])
    check = Check("ocr", "OCR", WARN, "missing", fix=Fix("install", command=("echo", "installed")))
    _repair(out, prompt, [check], answers)
    assert answers.fixed == ["ocr"] and answers.skipped == []
    assert "done" in out.text


def test_repair_respects_a_refusal():
    answers = Answers(fixed=[], skipped=[])
    out, prompt = FakeOut(), ScriptedPrompt(confirms=[False])
    check = Check("ocr", "OCR", WARN, "missing", fix=Fix("install", command=("echo", "hi")))
    _repair(out, prompt, [check], answers)
    assert answers.skipped == ["ocr"] and answers.fixed == []


def test_repair_reports_a_failing_fix_without_raising():
    answers = Answers(fixed=[], skipped=[])
    out, prompt = FakeOut(), ScriptedPrompt(confirms=[True])
    check = Check("x", "X", FAIL, "broken", fix=Fix("try", command=("false",)))
    _repair(out, prompt, [check], answers)
    assert answers.skipped == ["x"]
    assert "failed" in out.text


def test_repair_never_runs_sudo_without_a_terminal():
    """sudo would block on a password prompt nobody can answer."""
    answers = Answers(fixed=[], skipped=[])
    out = FakeOut()
    prompt = Prompt(assume_yes=True, interactive=False)
    check = Check("x", "X", FAIL, "missing",
                  fix=Fix("install", command=("echo", "should-not-run"), needs_sudo=True))
    _repair(out, prompt, [check], answers)
    assert answers.skipped == ["x"]
    assert "not an interactive terminal" in out.text
    assert "should-not-run" not in [line for line in out.lines if "done" in line]


def test_repair_prints_manual_instructions_verbatim():
    answers = Answers(fixed=[], skipped=[])
    out, prompt = FakeOut(), ScriptedPrompt()
    check = Check("display", "display", FAIL, "wayland",
                  fix=Fix("switch", manual="Log out and pick the Xorg session."))
    _repair(out, prompt, [check], answers)
    assert "Log out and pick the Xorg session." in out.text
    assert answers.skipped == ["display"]


# -- mode step -----------------------------------------------------------


def test_mode_step_defaults_to_ask():
    assert _setup_mode(FakeOut(), ScriptedPrompt()) == "ask"


def test_mode_step_maps_every_choice():
    assert _setup_mode(FakeOut(), ScriptedPrompt(choices=[1])) == "auto"
    assert _setup_mode(FakeOut(), ScriptedPrompt(choices=[2])) == "readonly"


def test_yolo_requires_a_second_confirmation():
    out = FakeOut()
    prompt = ScriptedPrompt(choices=[3], confirms=[True])
    assert _setup_mode(out, prompt) == "yolo"
    assert any("really" in q.lower() for q in prompt.asked), "yolo must be confirmed twice"
    assert "will not ask before anything" in out.text


def test_yolo_falls_back_to_ask_when_declined():
    assert _setup_mode(FakeOut(), ScriptedPrompt(choices=[3], confirms=[False])) == "ask"


# -- needs_setup ---------------------------------------------------------


def test_needs_setup_is_false_once_a_key_is_saved(tmp_path, monkeypatch):
    """A key written by a previous setup is invisible to env discovery."""
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    config_file.write(tmp_path, {"provider": {"name": "anthropic", "api_key": "sk-ant-xyz"}})
    assert needs_setup(load_config()) is False


def test_needs_setup_is_true_when_setup_was_skipped(tmp_path, monkeypatch):
    """A config with no backend still leaves LAI unable to do anything."""
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    config_file.write(tmp_path, {"safety": {"mode": "ask"}})
    assert needs_setup(load_config()) is True


def test_needs_setup_is_false_when_a_key_is_already_exported(tmp_path, monkeypatch):
    """Someone with ANTHROPIC_API_KEY in their profile must not be made to run a wizard."""
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials",
                        lambda: [type("C", (), {"provider": "anthropic"})()])
    assert needs_setup(load_config()) is False


def test_needs_setup_is_true_on_a_fresh_machine(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    assert needs_setup(load_config()) is True


# -- the whole wizard ----------------------------------------------------


@pytest.fixture
def wizard_env(tmp_path, monkeypatch):
    """A wizard that touches nothing real: no runtime, no packages, no model."""
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setattr(
        "lai.setup_wizard._probe",
        lambda config: Report([
            Check("platform", "platform", OK, "Linux"),
            Check("display", "display server", OK, "x11"),
            Check("provider", "model provider", OK, "zai / glm-5"),
        ]),
    )
    monkeypatch.setattr("lai.models.discover", lambda **kwargs: [fake_backend()])
    monkeypatch.setattr(
        "lai.agent.providers.registry.discover_credentials",
        lambda: [type("C", (), {"provider": "zai", "model": "glm-5",
                                "describe": lambda self: "zai (glm-5) via ZAI_API_KEY"})()],
    )
    return tmp_path


def test_wizard_writes_a_config_and_reports_ready(wizard_env):
    out = FakeOut()
    code, answers = run_setup(out, assume_yes=True, interactive=False, skip_demo=True)
    assert code == 0
    assert answers.config_written == wizard_env / "config.toml"
    assert answers.mode == "ask"
    assert config_file.read(wizard_env)["safety"]["mode"] == "ask"
    assert "Setup complete" in out.text


def test_wizard_keeps_an_existing_backend_without_asking_for_a_key(wizard_env):
    out = FakeOut()
    _, answers = run_setup(out, assume_yes=True, interactive=False, skip_demo=True)
    assert answers.provider == "zai"
    assert "zai" in out.text
    assert "api_key" not in config_file.render(config_file.read(wizard_env))


def test_wizard_is_safe_to_run_twice(wizard_env):
    run_setup(FakeOut(), assume_yes=True, interactive=False, skip_demo=True)
    config_file.write(wizard_env, config_file.merge(
        config_file.read(wizard_env), {"channels": {"enabled": ["telegram"]}}
    ))
    run_setup(FakeOut(), assume_yes=True, interactive=False, skip_demo=True)
    settings = config_file.read(wizard_env)
    assert settings["channels"]["enabled"] == ["telegram"], "a second run must not drop earlier settings"
    assert settings["safety"]["mode"] == "ask"


def test_wizard_reports_blockers_it_could_not_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setattr(
        "lai.setup_wizard._probe",
        lambda config: Report([
            Check("display", "display server", FAIL, "wayland",
                  fix=Fix("switch", manual="Log in to an Xorg session.")),
        ]),
    )
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    out = FakeOut()
    code, _ = run_setup(out, assume_yes=True, interactive=False, skip_demo=True)
    assert code == 1
    assert "still need attention" in out.text
    assert "Log in to an Xorg session." in out.text


def test_wizard_always_ends_with_what_to_try_next(wizard_env):
    out = FakeOut()
    run_setup(out, assume_yes=True, interactive=False, skip_demo=True)
    assert "lai do" in out.text and "lai doctor" in out.text


# -- getting a key -------------------------------------------------------


def test_backend_urls_are_real_links():
    """The wizard offers to open these, so they must be openable."""
    from lai.setup_wizard import BACKENDS

    for _name, _label, url, _example in BACKENDS:
        assert url.startswith("https://"), f"{url} is not a URL the browser can open"


def test_opening_the_key_page_is_never_fatal(monkeypatch):
    from lai import setup_wizard

    monkeypatch.setattr(setup_wizard.shutil, "which", lambda name: None)
    out = FakeOut()
    setup_wizard._open_url(out, "https://example.test/keys")
    assert "open it yourself" in out.text


def test_opening_the_key_page_uses_xdg_open(monkeypatch):
    from lai import setup_wizard

    launched: list[list[str]] = []
    monkeypatch.setattr(setup_wizard.shutil, "which",
                        lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None)
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: launched.append(cmd))
    out = FakeOut()
    setup_wizard._open_url(out, "https://example.test/keys")
    assert launched == [["/usr/bin/xdg-open", "https://example.test/keys"]]
    assert "opened in your browser" in out.text


def test_a_pasted_key_is_verified_before_it_is_saved(tmp_path, monkeypatch):
    """A typo must be caught in the wizard, not on the first real task."""
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    no_backends(monkeypatch)
    monkeypatch.setattr("lai.setup_wizard._probe", lambda config: Report([]))

    verified: list[tuple[str, str]] = []

    def fake_verify(provider, key):
        verified.append((provider, key))
        return True, "replied 'OK'", "claude-sonnet-4-5"

    monkeypatch.setattr("lai.setup_wizard._verify_key", fake_verify)
    monkeypatch.setattr("lai.setup_wizard._open_url", lambda out, url: None)

    from lai.setup_wizard import _setup_provider

    out = FakeOut()
    # Nothing is ready, so the menu starts at the key-paste entries.
    prompt = ScriptedPrompt(choices=[0], confirms=[False], secrets=["sk-ant-abcdefghijklmnop"])
    settings = _setup_provider(out, prompt, Report([]), Answers(fixed=[], skipped=[]))

    assert verified == [("anthropic", "sk-ant-abcdefghijklmnop")]
    assert settings["api_key"] == "sk-ant-abcdefghijklmnop"
    assert settings["name"] == "anthropic"
    assert "works" in out.text


def test_a_rejected_key_is_not_saved_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    no_backends(monkeypatch)
    monkeypatch.setattr("lai.setup_wizard._verify_key",
                        lambda p, k: (False, "AuthenticationError: invalid x-api-key", "m"))
    monkeypatch.setattr("lai.setup_wizard._open_url", lambda out, url: None)

    from lai.setup_wizard import _setup_provider

    out = FakeOut()
    # open browser? no · save anyway? no
    prompt = ScriptedPrompt(choices=[0], confirms=[False, False], secrets=["sk-ant-abcdefghijklmnop"])
    settings = _setup_provider(out, prompt, Report([]), Answers(fixed=[], skipped=[]))

    assert settings == {}, "a rejected key must not be written"
    assert "rejected" in out.text


def test_skipping_the_backend_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    no_backends(monkeypatch)

    from lai.setup_wizard import BACKENDS, _setup_provider

    out = FakeOut()
    # keys… + ollama + something-else + skip, with nothing ready above them.
    skip_index = len(BACKENDS) + 2
    prompt = ScriptedPrompt(choices=[skip_index])
    assert _setup_provider(out, prompt, Report([]), Answers(fixed=[], skipped=[])) == {}
    assert "lai setup" in out.text


# -- ollama, the no-key path ---------------------------------------------


def test_ollama_needs_to_be_installed(monkeypatch):
    from lai import setup_wizard

    monkeypatch.setattr(setup_wizard.shutil, "which", lambda name: None)
    out = FakeOut()
    settings = setup_wizard._setup_ollama(out, ScriptedPrompt(), Answers(fixed=[], skipped=[]))
    assert settings == {}
    assert "ollama.com" in out.text


def test_ollama_with_a_served_model_is_configured(monkeypatch):
    from lai import setup_wizard

    monkeypatch.setattr(setup_wizard.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr("lai.agent.providers.registry._ollama_model", lambda: "qwen3-vl:2b")
    out, answers = FakeOut(), Answers(fixed=[], skipped=[])
    settings = setup_wizard._setup_ollama(out, ScriptedPrompt(), answers)
    assert settings["name"] == "ollama" and settings["model"] == "qwen3-vl:2b"
    assert settings["base_url"].startswith("http")
    assert answers.provider == "ollama"
    assert "struggle with dense interfaces" in out.text, "the tradeoff must be stated"


def test_ollama_installed_but_not_serving(monkeypatch):
    from lai import setup_wizard

    monkeypatch.setattr(setup_wizard.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr("lai.agent.providers.registry._ollama_model", lambda: "")
    out = FakeOut()
    settings = setup_wizard._setup_ollama(out, ScriptedPrompt(confirms=[True]), Answers(fixed=[], skipped=[]))
    assert "ollama serve" in out.text
    assert settings["model"] == "qwen3-vl:2b", "a default model lets them continue"


def test_ollama_probe_failure_is_not_fatal(monkeypatch):
    from lai import setup_wizard

    def explode():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(setup_wizard.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr("lai.agent.providers.registry._ollama_model", explode)
    out = FakeOut()
    settings = setup_wizard._setup_ollama(out, ScriptedPrompt(confirms=[False]), Answers(fixed=[], skipped=[]))
    assert settings == {}


# -- the demo run --------------------------------------------------------


class DemoResult:
    def __init__(self, ok=True):
        self.ok = ok
        self.status = "completed" if ok else "blocked"
        self.summary = "Seven windows are open."
        self.error = "" if ok else "could not read the window list"
        self.steps = 2
        self.elapsed = 11.0


class DemoRuntime:
    def __init__(self, result=None, provider=object()):
        self.provider = provider
        self.provider_error = "no backend"
        self._result = result or DemoResult()
        self.closed = False
        self.task = ""

    def agent(self, **kwargs):
        runtime = self

        class Agent:
            def run(self, task):
                runtime.task = task
                return runtime._result

        return Agent()

    def close(self):
        self.closed = True


def test_demo_runs_a_real_task_and_reports_it(tmp_path, monkeypatch):
    from lai import setup_wizard

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    runtime = DemoRuntime()
    monkeypatch.setattr("lai.runtime.build_runtime", lambda *a, **kw: runtime)

    out, answers = FakeOut(), Answers(fixed=[], skipped=[], provider="zai")
    setup_wizard._run_demo(out, ScriptedPrompt(confirms=[True]), answers)

    assert answers.demo_ran and answers.demo_ok
    assert "Seven windows are open." in out.text
    assert runtime.task == setup_wizard.DEMO_TASK
    assert runtime.closed, "the demo runtime must be closed"


def test_demo_runs_readonly(tmp_path, monkeypatch):
    """The first thing a new user sees must not change their machine."""
    from lai import setup_wizard

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    seen: dict = {}

    def fake_build(config, **kwargs):
        seen["mode"] = config.safety.mode
        seen["steps"] = config.limits.max_steps
        return DemoRuntime()

    monkeypatch.setattr("lai.runtime.build_runtime", fake_build)
    setup_wizard._run_demo(FakeOut(), ScriptedPrompt(confirms=[True]),
                           Answers(fixed=[], skipped=[], provider="zai"))
    assert seen["mode"] == "readonly"
    assert seen["steps"] <= 6, "the demo must be short"


def test_demo_reports_a_failure_without_raising(tmp_path, monkeypatch):
    from lai import setup_wizard

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    monkeypatch.setattr("lai.runtime.build_runtime",
                        lambda *a, **kw: DemoRuntime(result=DemoResult(ok=False)))
    out, answers = FakeOut(), Answers(fixed=[], skipped=[], provider="zai")
    setup_wizard._run_demo(out, ScriptedPrompt(confirms=[True]), answers)
    assert answers.demo_ran and not answers.demo_ok
    assert "could not read the window list" in out.text


def test_demo_survives_a_runtime_that_will_not_build(tmp_path, monkeypatch):
    from lai import setup_wizard

    monkeypatch.setenv("LAI_HOME", str(tmp_path))

    def explode(*a, **kw):
        raise RuntimeError("no display")

    monkeypatch.setattr("lai.runtime.build_runtime", explode)
    out = FakeOut()
    setup_wizard._run_demo(out, ScriptedPrompt(confirms=[True]),
                           Answers(fixed=[], skipped=[], provider="zai"))
    assert "could not start" in out.text and "no display" in out.text


def test_demo_is_skipped_without_a_backend(monkeypatch):
    from lai import setup_wizard

    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    out, answers = FakeOut(), Answers(fixed=[], skipped=[])
    setup_wizard._run_demo(out, ScriptedPrompt(), answers)
    assert not answers.demo_ran
    assert "skipping the demo" in out.text


def test_demo_can_be_declined(tmp_path, monkeypatch):
    from lai import setup_wizard

    monkeypatch.setenv("LAI_HOME", str(tmp_path))
    out, answers = FakeOut(), Answers(fixed=[], skipped=[], provider="zai")
    setup_wizard._run_demo(out, ScriptedPrompt(confirms=[False]), answers)
    assert not answers.demo_ran


# -- key verification ----------------------------------------------------


def test_verify_key_reports_a_working_key(monkeypatch):
    from lai import setup_wizard

    class Turn:
        text = "OK"

    class Instance:
        def complete(self, messages, system=""):
            return Turn()

        def close(self):
            Instance.closed = True

    Instance.closed = False
    monkeypatch.setattr("lai.agent.providers.registry._instantiate", lambda *a, **kw: Instance())
    ok, detail, model = setup_wizard._verify_key("anthropic", "sk-ant-x")
    assert ok and "OK" in detail and model
    assert Instance.closed, "the probe client must be closed"


def test_verify_key_reports_a_rejected_key(monkeypatch):
    from lai import setup_wizard

    def explode(*a, **kw):
        raise RuntimeError("invalid x-api-key")

    monkeypatch.setattr("lai.agent.providers.registry._instantiate", explode)
    ok, detail, _model = setup_wizard._verify_key("anthropic", "sk-ant-bad")
    assert not ok and "invalid x-api-key" in detail


def test_verify_key_spends_only_a_tiny_request(monkeypatch):
    """Verification must not cost a real generation."""
    from lai import setup_wizard

    seen: dict = {}

    def capture(name, config, credential):
        seen["max_tokens"] = config.max_tokens
        raise RuntimeError("stop here")

    monkeypatch.setattr("lai.agent.providers.registry._instantiate", capture)
    setup_wizard._verify_key("anthropic", "sk-ant-x")
    assert seen["max_tokens"] <= 32


# -- picking a model during onboarding -----------------------------------


def a_model(identifier, **kwargs):
    from lai.agent.providers.listing import ModelInfo

    return ModelInfo(id=identifier, **kwargs)


def test_a_working_key_leads_straight_into_choosing_a_model(monkeypatch):
    """With hundreds on offer, picking for somebody is worse than showing them
    the free one at the top."""
    from lai.setup_wizard import _offer_models

    monkeypatch.setattr("lai.models.endpoint_for", lambda name: ("https://x.test/v1", ""))
    monkeypatch.setattr(
        "lai.agent.providers.listing.fetch",
        lambda url, key, timeout=8.0: [a_model("free/one", free=True), a_model("paid/two")],
    )
    out, prompt = FakeOut(), ScriptedPrompt(confirms=[True], choices=[1])
    assert _offer_models(out, prompt, "openrouter", "sk-or-x") == "paid/two"


def test_the_default_can_be_kept(monkeypatch):
    from lai.setup_wizard import _offer_models

    monkeypatch.setattr("lai.models.endpoint_for", lambda name: ("https://x.test/v1", ""))
    monkeypatch.setattr(
        "lai.agent.providers.listing.fetch",
        lambda url, key, timeout=8.0: [a_model("a"), a_model("b")],
    )
    out, prompt = FakeOut(), ScriptedPrompt(confirms=[True], choices=[2])  # past the end = keep
    assert _offer_models(out, prompt, "openrouter", "k") == ""


def test_declining_leaves_the_default(monkeypatch):
    from lai.setup_wizard import _offer_models

    monkeypatch.setattr("lai.models.endpoint_for", lambda name: ("https://x.test/v1", ""))
    monkeypatch.setattr(
        "lai.agent.providers.listing.fetch",
        lambda url, key, timeout=8.0: [a_model("a"), a_model("b")],
    )
    out, prompt = FakeOut(), ScriptedPrompt(confirms=[False])
    assert _offer_models(out, prompt, "openrouter", "k") == ""


def test_a_vendor_that_will_not_list_its_models_is_not_a_problem(monkeypatch):
    from lai.setup_wizard import _offer_models

    def explode(name):
        raise RuntimeError("no catalogue endpoint")

    monkeypatch.setattr("lai.models.endpoint_for", explode)
    assert _offer_models(FakeOut(), ScriptedPrompt(confirms=[True]), "weird", "k") == ""


def test_a_single_model_is_not_worth_a_question(monkeypatch):
    from lai.setup_wizard import _offer_models

    monkeypatch.setattr("lai.models.endpoint_for", lambda name: ("https://x.test/v1", ""))
    monkeypatch.setattr(
        "lai.agent.providers.listing.fetch", lambda url, key, timeout=8.0: [a_model("only")]
    )
    assert _offer_models(FakeOut(), ScriptedPrompt(confirms=[True]), "x", "k") == ""


def test_a_scripted_install_is_never_asked(monkeypatch):
    """`--yes` must not stop to ask which model somebody wants."""
    from lai.setup_wizard import _offer_models

    assert _offer_models(FakeOut(), Prompt(interactive=False), "openrouter", "k") == ""


def test_openrouter_is_offered_during_onboarding():
    """Somebody who has installed this and has no key needs to see the option
    that gives them hundreds of models for one signup."""
    from lai.setup_wizard import BACKENDS

    names = {entry[0] for entry in BACKENDS}
    assert "openrouter" in names
    signup = next(entry[2] for entry in BACKENDS if entry[0] == "openrouter")
    assert signup.startswith("https://")
