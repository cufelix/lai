"""CLI: argument parsing, command dispatch, and output shape."""

from __future__ import annotations

import json

import pytest

from lai.cli import Out, _apply_overrides, _make_reporter, _strip_markup, build_parser, main
from lai.config import load_config


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("LAI_HOME", str(tmp_path / "home"))
    for name in ("LAI_MODE", "LAI_MODEL", "LAI_PROVIDER", "LAI_MAX_STEPS", "LAI_DRY_RUN"):
        monkeypatch.delenv(name, raising=False)


# -- parser --------------------------------------------------------------


def test_every_command_is_registered():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    commands = set(actions[0].choices)
    assert {"do", "repl", "doctor", "observe", "tools", "skills", "sessions", "serve", "mcp",
            "tui", "channels", "schedule"} <= commands


# -- help ----------------------------------------------------------------


def test_the_command_table_covers_every_registered_command():
    """COMMANDS drives both the parser and the overview; drift would orphan one."""
    from lai.cli import COMMANDS

    parser = build_parser()
    registered = set(next(a for a in parser._actions if a.dest == "command").choices)
    assert registered == set(COMMANDS), (
        "every subcommand needs an entry in cli.COMMANDS or it disappears from --help"
    )


def test_the_overview_names_every_command(capsys):
    """`lai --help` must mention every command — a help screen that omits one lies."""
    from lai.cli import COMMANDS

    assert main(["--help"]) == 0
    text = capsys.readouterr().out
    for name in COMMANDS:
        assert name in text, f"`lai --help` does not mention {name}"
    assert "lai setup" in text
    assert "lai doctor" in text


def test_help_for_a_subcommand_shows_its_examples(capsys):
    assert main(["help", "do"]) == 0
    text = capsys.readouterr().out
    assert "examples" in text
    assert "task" in text  # the positional argument is still explained


def test_a_spinner_runs_while_the_model_thinks():
    """The reporter must show life between one tool ending and the model speaking."""

    class FakeStatus:
        stopped = False

        def stop(self):
            self.stopped = True

    class FakeOut:
        def __init__(self):
            self.statuses: list[FakeStatus] = []

        def spinner(self, text):
            status = FakeStatus()
            self.statuses.append(status)
            return status

        def write(self, *args, **kwargs):
            pass

        def rule(self, title=""):
            pass

        def raw(self, text):
            pass

    out = FakeOut()
    report = _make_reporter(out)
    report("start", {"provider": "zai", "model": "glm", "task": "do a thing"})
    assert len(out.statuses) == 1 and not out.statuses[0].stopped
    report("tool_call", {"name": "screenshot", "input": {}})
    assert out.statuses[0].stopped, "output must clear the spinner before printing"
    report("tool_result", {"ok": True, "summary": "done"})
    assert len(out.statuses) == 2, "the next model call starts a fresh spinner"
    report("done", {})
    assert out.statuses[1].stopped


def test_no_spinner_when_output_is_muted():
    assert Out(quiet=True).spinner("thinking…") is None


# -- schedule ------------------------------------------------------------


def test_schedule_list_is_empty_and_explains_itself(capsys):
    assert main(["schedule", "list"]) == 0
    text = capsys.readouterr().out
    assert "No scheduled tasks" in text and "schedule add" in text


def test_schedule_add_list_remove_roundtrip(capsys):
    assert main(["schedule", "add", "nightly", "@daily", "summarise my notes"]) == 0
    assert "added" in capsys.readouterr().out

    assert main(["schedule", "list", "--json"]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert len(tasks) == 1
    task = tasks[0]
    assert task["name"] == "nightly" and task["schedule"] == "@daily"
    assert task["next_run"] > 0 and task["enabled"]

    assert main(["schedule", "remove", task["id"]]) == 0
    capsys.readouterr()
    assert main(["schedule", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_schedule_add_validates_its_arguments(capsys):
    assert main(["schedule", "add", "incomplete"]) == 2
    assert "usage" in capsys.readouterr().err


def test_schedule_add_rejects_an_impossible_cron(capsys):
    # 30 February never arrives; it must fail at add time, not at fire time.
    assert main(["schedule", "add", "broken", "0 0 30 2 *", "never runs"]) == 2
    assert capsys.readouterr().err


def test_schedule_disable_clears_the_next_run(capsys):
    main(["schedule", "add", "pinger", "every:900", "check the build"])
    capsys.readouterr()
    main(["schedule", "list", "--json"])
    task_id = json.loads(capsys.readouterr().out)[0]["id"]

    assert main(["schedule", "disable", task_id]) == 0
    assert main(["schedule", "list"]) == 0
    assert "off" in capsys.readouterr().out

    assert main(["schedule", "enable", task_id]) == 0
    assert main(["schedule", "list"]) == 0
    assert "on" in capsys.readouterr().out


def test_schedule_actions_report_a_missing_id(capsys):
    assert main(["schedule", "remove", "nosuchid"]) == 1
    assert main(["schedule", "enable", "nosuchid"]) == 1
    assert "no task" in capsys.readouterr().err


def test_do_requires_a_task():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["do"])


def test_do_flags_parse():
    args = build_parser().parse_args([
        "do", "open the editor", "--mode", "auto", "--steps", "5",
        "--model", "glm-4.6", "--dry-run", "--json",
    ])
    assert args.task == "open the editor"
    assert args.mode == "auto" and args.steps == 5
    assert args.model == "glm-4.6" and args.dry_run and args.json


def test_invalid_mode_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["do", "x", "--mode", "banana"])


def test_serve_defaults_to_loopback():
    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1" and args.port == 8787 and args.allow_remote is False


def test_skills_action_choices():
    assert build_parser().parse_args(["skills"]).action == "list"
    assert build_parser().parse_args(["skills", "install", "owner/repo"]).query == "owner/repo"


# -- overrides -----------------------------------------------------------


def test_apply_overrides_maps_flags_onto_config():
    args = build_parser().parse_args([
        "do", "t", "--mode", "yolo", "--model", "m1", "--provider", "openai",
        "--steps", "7", "--timeout", "30", "--thinking", "1024", "--dry-run",
    ])
    config = _apply_overrides(load_config(), args)
    assert config.safety.mode == "yolo" and config.safety.dry_run is True
    assert config.provider.model == "m1" and config.provider.name == "openai"
    assert config.provider.thinking_budget == 1024
    assert config.limits.max_steps == 7 and config.limits.max_seconds == 30


def test_apply_overrides_leaves_unset_flags_alone():
    original = load_config()
    args = build_parser().parse_args(["do", "t"])
    config = _apply_overrides(original, args)
    assert config.safety.mode == original.safety.mode
    assert config.provider.model == original.provider.model


def test_apply_overrides_does_not_mutate_the_original():
    original = load_config()
    args = build_parser().parse_args(["do", "t", "--mode", "yolo"])
    _apply_overrides(original, args)
    assert original.safety.mode != "yolo"


# -- output helpers ------------------------------------------------------


def test_strip_markup_removes_rich_tags():
    assert _strip_markup("[bold]hi[/bold] [dim]there[/dim]") == "hi there"


def test_quiet_output_writes_nothing(capsys):
    out = Out(quiet=True)
    out.write("hello")
    out.raw("raw")
    out.rule("title")
    assert capsys.readouterr().out == ""


def test_output_without_color_is_plain(capsys):
    Out(color=False).write("[bold]plain[/bold]")
    assert "plain" in capsys.readouterr().out


def test_errors_go_to_stderr(capsys):
    Out().error("something broke")
    captured = capsys.readouterr()
    assert "something broke" in captured.err
    assert captured.out == ""


def test_reporter_renders_the_event_stream(capsys):
    report = _make_reporter(Out(color=False), stream=False, verbose=True)
    report("start", {"provider": "zai", "model": "glm", "task": "do a thing"})
    report("step", {"step": 1, "of": 5})
    report("tool_call", {"name": "ui_click", "input": {"ref": 3}})
    report("tool_result", {"name": "ui_click", "ok": True, "summary": "clicked\nmore", "images": 1})
    report("assistant", {"text": "all set"})
    report("compacting", {"estimated_tokens": 50000})
    report("error", {"error": "hiccup"})
    report("done", {})
    text = capsys.readouterr().out
    assert "do a thing" in text
    assert "ui_click" in text
    assert "clicked" in text and "+1 lines" in text
    assert "all set" in text
    assert "compacting" in text and "hiccup" in text


def test_reporter_streams_text_deltas(capsys):
    report = _make_reporter(Out(color=False), stream=True)
    report("text", {"delta": "hel"})
    report("text", {"delta": "lo"})
    report("tool_call", {"name": "x", "input": {}})
    assert "hello" in capsys.readouterr().out


# -- commands ------------------------------------------------------------


def test_tools_json_emits_valid_schemas(capsys):
    assert main(["tools", "--json", "--no-mcp"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload
    names = {entry["name"] for entry in payload}
    assert {"ui_snapshot", "app_open", "task_complete"} <= names
    for entry in payload:
        assert entry["input_schema"]["type"] == "object"


def test_tools_human_output_is_grouped(capsys):
    assert main(["tools", "--no-mcp"]) == 0
    text = capsys.readouterr().out
    assert "ui_snapshot" in text and "tool(s)" in text


def test_tools_filter(capsys):
    assert main(["tools", "window", "--no-mcp"]) == 0
    text = capsys.readouterr().out
    assert "window_list" in text and "ui_snapshot" not in text


def test_skills_list_runs(capsys):
    assert main(["skills", "list"]) == 0
    assert "skill(s)" in capsys.readouterr().out


def test_skills_show_requires_a_name(capsys):
    assert main(["skills", "show"]) == 2
    assert "usage" in capsys.readouterr().err


def test_skills_show_unknown_reports_an_error(capsys):
    assert main(["skills", "show", "definitely-not-a-real-skill"]) == 1
    assert "no skill named" in capsys.readouterr().err


def test_skills_install_requires_a_source(capsys):
    assert main(["skills", "install"]) == 2


def test_skills_install_and_show_roundtrip(tmp_path, capsys):
    source = tmp_path / "src" / "demo"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a demo skill\n---\nBody text here.\n", encoding="utf-8"
    )
    assert main(["skills", "install", str(source)]) == 0
    assert "demo" in capsys.readouterr().out
    assert main(["skills", "show", "demo"]) == 0
    assert "Body text here." in capsys.readouterr().out
    assert main(["skills", "remove", "demo"]) == 0


def test_sessions_empty_json(capsys):
    assert main(["sessions", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_sessions_lists_a_written_session(capsys):
    from lai.agent.session import Session

    config = load_config()
    config.ensure_dirs()
    session = Session(task="a recorded task")
    session.bind(config.sessions_dir)

    assert main(["sessions", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["task"] == "a recorded task"

    assert main(["sessions", session.id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == session.id


def test_do_without_a_provider_exits_two(monkeypatch, capsys):
    from lai.errors import ProviderError

    monkeypatch.setattr(
        "lai.runtime.build_provider",
        lambda config: (_ for _ in ()).throw(ProviderError("no backend here")),
    )
    assert main(["do", "anything", "--no-mcp"]) == 2
    assert "no backend here" in capsys.readouterr().err


def test_unknown_command_exits():
    assert main(["nonsense"]) == 2


def test_help_exits_zero():
    assert main(["--help"]) == 0


@pytest.mark.x11
def test_observe_json(capsys):
    assert main(["observe", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "windows" in payload and "monitors" in payload


@pytest.mark.x11
def test_observe_human_readable(capsys):
    assert main(["observe"]) == 0
    assert "FOCUSED" in capsys.readouterr().out


@pytest.mark.x11
def test_observe_saves_a_screenshot(tmp_path, capsys):
    target = tmp_path / "shot.png"
    assert main(["observe", "--screenshot", "--save", str(target)]) == 0
    assert target.is_file() and target.stat().st_size > 1000
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.x11
@pytest.mark.slow
def test_doctor_reports_every_subsystem(capsys):
    code = main(["doctor", "--no-mcp"])
    text = capsys.readouterr().out
    for label in ("platform", "display server", "screen capture", "input", "accessibility", "tools", "skills"):
        assert label in text
    assert code in (0, 1)


# -- onboarding ----------------------------------------------------------


def test_version_flag(capsys):
    assert main(["--version"]) == 0
    assert "lai" in capsys.readouterr().out


def test_help_points_a_newcomer_at_setup(capsys):
    assert main(["--help"]) == 0
    assert "lai setup" in capsys.readouterr().out


def test_a_missing_backend_names_the_command_that_fixes_it(monkeypatch, capsys):
    """The most common wall for a new user must not be a bare error string."""
    from lai import cli

    class NoProviderRuntime:
        provider = None
        provider_error = "no model backend available"

        def close(self):
            pass

    monkeypatch.setattr(cli, "build_runtime", lambda *a, **kw: NoProviderRuntime())
    code = main(["do", "anything", "--no-mcp"])
    captured = capsys.readouterr()
    assert code == 2
    assert "lai setup" in captured.out
    assert "ANTHROPIC_API_KEY" in captured.out
    assert "no model backend available" in captured.err


def test_default_command_is_setup_on_a_fresh_machine(monkeypatch):
    from lai import cli

    monkeypatch.setattr(cli, "build_parser", cli.build_parser)
    monkeypatch.setattr("lai.setup_wizard.needs_setup", lambda *a, **kw: True)
    assert cli._default_command([]) == ["setup"]


def test_default_command_is_the_interface_once_configured(monkeypatch):
    from lai import cli

    monkeypatch.setattr("lai.setup_wizard.needs_setup", lambda *a, **kw: False)
    assert cli._default_command([])[0] == "chat"


def test_default_command_survives_a_broken_setup_probe(monkeypatch):
    """A failure deciding what to do must not stop `lai` from doing anything."""
    from lai import cli

    def explode(*a, **kw):
        raise RuntimeError("cannot read config")

    monkeypatch.setattr("lai.setup_wizard.needs_setup", explode)
    assert cli._default_command([])[0] == "chat"


def test_doctor_json_is_machine_readable(capsys):
    code = main(["doctor", "--json", "--no-mcp"])
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload["ready"], bool)
    assert payload["checks"], "doctor must report at least one check"
    assert {"key", "label", "status", "detail"} <= set(payload["checks"][0])
    assert code in (0, 1)


def test_doctor_lists_the_fix_for_every_failure(monkeypatch, capsys):
    from lai.checks import FAIL, Check, Fix, Report

    monkeypatch.setattr(
        "lai.checks.run_checks",
        lambda *a, **kw: Report([
            Check("xdotool", "input", FAIL, "missing",
                  fix=Fix("install xdotool", command=("sudo", "apt-get", "install", "-y", "xdotool"))),
        ]),
    )
    code = main(["doctor", "--no-mcp"])
    out = capsys.readouterr().out
    assert code == 1
    assert "sudo apt-get install -y xdotool" in out
    assert "lai doctor --fix" in out


def test_doctor_fix_applies_automatic_repairs(monkeypatch, capsys):
    from lai.checks import FAIL, Check, Fix, Report

    ran: list[str] = []

    def fake_run(self, *, timeout=300.0):
        ran.append(self.description)
        return True, "installed"

    monkeypatch.setattr(Fix, "run", fake_run)
    monkeypatch.setattr(
        "lai.checks.run_checks",
        lambda *a, **kw: Report([
            Check("ocr", "OCR", FAIL, "missing", fix=Fix("install tesseract", command=("true",))),
        ]),
    )
    main(["doctor", "--fix", "--no-mcp"])
    assert ran == ["install tesseract"], "the fix should have been applied exactly once"
    assert "Applying fixes" in capsys.readouterr().out


def test_setup_command_runs_the_wizard(monkeypatch):
    calls: dict = {}

    def fake_run_setup(out, **kwargs):
        calls.update(kwargs)
        from lai.setup_wizard import Answers

        return 0, Answers(fixed=[], skipped=[])

    monkeypatch.setattr("lai.setup_wizard.run_setup", fake_run_setup)
    assert main(["setup", "--yes", "--no-demo"]) == 0
    assert calls["assume_yes"] is True
    assert calls["skip_demo"] is True


def test_doctor_does_not_connect_mcp_by_default(monkeypatch):
    """It is the command you run when something is wrong; it must be instant."""
    from lai import cli

    seen: dict = {}

    def fake_build(config, **kwargs):
        seen["with_mcp"] = kwargs.get("with_mcp")
        return _StubRuntime(config)

    monkeypatch.setattr(cli, "build_runtime", fake_build)
    main(["doctor"])
    assert seen["with_mcp"] is False

    main(["doctor", "--mcp"])
    assert seen["with_mcp"] is True


def test_doctor_says_when_mcp_was_not_checked(monkeypatch, capsys):
    from lai import cli

    monkeypatch.setattr(cli, "build_runtime", lambda config, **kw: _StubRuntime(config))
    main(["doctor"])
    assert "not checked" in capsys.readouterr().out


class _StubRuntime:
    """Enough of a Runtime for doctor to render."""

    def __init__(self, config):
        self.config = config
        self.desktop = None
        self.provider = None
        self.provider_error = "none configured"
        self.registry = []
        self.skills = []
        self.mcp_tools = []
        self.mcp_errors = {}

    def close(self):
        pass


# -- models --------------------------------------------------------------


def test_models_lists_what_is_usable(capsys, monkeypatch):
    from lai.models import KIND_API, KIND_CLI, KNOWN, READY, Backend

    monkeypatch.setattr("lai.models.discover", lambda **kw: [
        Backend("zai", "GLM", KIND_API, READY, "via ZAI_API_KEY", "glm-5"),
        Backend("cli:claude", "Claude Code", KIND_CLI, READY, "on PATH", "claude", vision=False),
        Backend("groq", "Groq", KIND_API, KNOWN, "set GROQ_API_KEY", "llama"),
    ])
    assert main(["models"]) == 0
    text = capsys.readouterr().out
    assert "Ready now" in text
    assert "zai" in text and "cli:claude" in text
    assert "no vision" in text, "a backend that cannot see screenshots must say so"
    assert "2 usable now, 3 known in total" in text


def test_models_hides_the_long_tail_until_asked(capsys, monkeypatch):
    from lai.models import KIND_API, KNOWN, Backend

    monkeypatch.setattr("lai.models.discover", lambda **kw: [
        Backend(f"v{i}", f"V{i}", KIND_API, KNOWN, "set KEY", "m") for i in range(9)
    ])
    main(["models"])
    assert "and 9 more" in capsys.readouterr().out

    main(["models", "--all"])
    assert "v8" in capsys.readouterr().out


def test_models_json_is_machine_readable(capsys, monkeypatch):
    from lai.models import KIND_API, READY, Backend

    monkeypatch.setattr("lai.models.discover", lambda **kw: [
        Backend("zai", "GLM", KIND_API, READY, "via ZAI_API_KEY", "glm-5"),
    ])
    assert main(["models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "zai" and payload[0]["status"] == "ready"


def test_models_test_reports_success_and_failure(capsys, monkeypatch):
    monkeypatch.setattr("lai.models.check", lambda name, **kw: (True, "zai/glm-5 replied 'OK'"))
    assert main(["models", "test", "zai"]) == 0
    assert "works" in capsys.readouterr().out

    monkeypatch.setattr("lai.models.check", lambda name, **kw: (False, "401 Unauthorized"))
    assert main(["models", "test", "zai"]) == 1
    assert "401" in capsys.readouterr().out


def test_models_test_needs_a_name(capsys):
    assert main(["models", "test"]) == 2
    assert "usage" in capsys.readouterr().err


def test_models_use_writes_the_default(capsys, monkeypatch, tmp_path):
    from lai import config_file

    monkeypatch.setattr("lai.models.check", lambda name, **kw: (True, "ok"))
    assert main(["models", "use", "cli:claude"]) == 0
    home = load_config().home
    assert config_file.read(home)["provider"]["name"] == "cli:claude"
    assert "cli:claude" in capsys.readouterr().out


def test_models_use_refuses_a_backend_that_does_not_work(capsys, monkeypatch):
    from lai import config_file

    monkeypatch.setattr("lai.models.check", lambda name, **kw: (False, "no API key"))
    assert main(["models", "use", "groq"]) == 1
    assert "--force" in capsys.readouterr().out
    assert config_file.read(load_config().home) == {}, "nothing may be written"


def test_models_use_force_saves_anyway(monkeypatch):
    from lai import config_file

    monkeypatch.setattr("lai.models.check", lambda name, **kw: (False, "no API key"))
    assert main(["models", "use", "groq", "--force"]) == 0
    assert config_file.read(load_config().home)["provider"]["name"] == "groq"
