"""Environment checks and the repairs attached to them.

The contract that matters: a check never raises, a failure that can be fixed
always carries the fix, and `Report.ready` reflects only what would actually
stop LAI from working — an absent tesseract must not read as "broken".
"""

from __future__ import annotations

import pytest

from lai.checks import (
    FAIL,
    OK,
    WARN,
    Check,
    Fix,
    Report,
    check_a11y,
    check_apps,
    check_clipboard,
    check_config,
    check_display,
    check_ocr,
    check_platform,
    check_provider,
    check_screen,
    check_windows,
    check_xdotool,
    package_manager,
    run_checks,
)

# -- Check / Fix primitives ----------------------------------------------


def test_blocking_is_required_and_failed():
    assert Check("k", "l", FAIL, "d").blocking
    assert not Check("k", "l", FAIL, "d", required=False).blocking, "optional pieces never block"
    assert not Check("k", "l", WARN, "d").blocking
    assert not Check("k", "l", OK, "d").blocking


def test_fix_knows_whether_it_can_act():
    assert Fix("x", command=("true",)).automatic
    assert Fix("x", apply=lambda: "done").automatic
    assert not Fix("x", manual="do it yourself").automatic


def test_fix_runs_a_command_and_reports_success():
    ok, output = Fix("say hi", command=("echo", "hello")).run()
    assert ok and "hello" in output


def test_fix_reports_a_failing_command_without_raising():
    ok, output = Fix("fail", command=("false",)).run()
    assert not ok
    assert isinstance(output, str)


def test_fix_reports_a_missing_binary():
    ok, output = Fix("nope", command=("definitely-not-a-real-binary-xyz",)).run()
    assert not ok and output


def test_fix_contains_an_exception_from_apply():
    def explode() -> str:
        raise RuntimeError("boom")

    ok, output = Fix("x", apply=explode).run()
    assert not ok and "boom" in output


def test_fix_apply_returns_its_message():
    ok, output = Fix("x", apply=lambda: "changed 3 things").run()
    assert ok and output == "changed 3 things"


def test_fix_shell_renders_the_command():
    assert Fix("x", command=("sudo", "apt-get", "install", "-y", "xdotool")).shell() == (
        "sudo apt-get install -y xdotool"
    )
    assert Fix("x", manual="…").shell() == ""


def test_fix_with_nothing_to_run():
    ok, output = Fix("x", manual="by hand").run()
    assert not ok and "nothing to run" in output


# -- report --------------------------------------------------------------


def test_report_ready_ignores_optional_failures():
    report = Report([
        Check("a", "a", OK, ""),
        Check("b", "b", FAIL, "", required=False),
        Check("c", "c", WARN, ""),
    ])
    assert report.ready, "an optional failure must not make the machine 'not ready'"
    assert report.blockers == []


def test_report_ready_is_false_for_a_required_failure():
    report = Report([Check("a", "a", OK, ""), Check("x", "x", FAIL, "broken")])
    assert not report.ready
    assert [c.key for c in report.blockers] == ["x"]


def test_report_fixable_lists_only_automatic_repairs():
    report = Report([
        Check("a", "a", FAIL, "", fix=Fix("run", command=("true",))),
        Check("b", "b", FAIL, "", fix=Fix("manual", manual="by hand")),
        Check("c", "c", FAIL, ""),
        Check("d", "d", OK, "", fix=Fix("run", command=("true",))),
    ])
    assert [c.key for c in report.fixable] == ["a"]


def test_report_get_and_len():
    report = Report([Check("a", "a", OK, "")])
    assert len(report) == 1
    assert report.get("a") is not None
    assert report.get("missing") is None


def test_report_to_dict_is_serialisable():
    import json

    report = Report([Check("a", "a", FAIL, "d", fix=Fix("fix it", command=("echo", "x")))])
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["ready"] is False
    assert payload["checks"][0]["fix_command"] == "echo x"


# -- individual probes ---------------------------------------------------


def test_platform_check_runs():
    check = check_platform()
    assert check.key == "platform"
    assert check.status in (OK, FAIL)


def test_display_check_reports_wayland_with_a_way_out(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("DISPLAY", ":0")
    check = check_display()
    assert check.status == FAIL
    assert check.fix is not None and "Xorg" in check.fix.manual


def test_display_check_explains_a_missing_display(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "")
    monkeypatch.delenv("DISPLAY", raising=False)
    check = check_display()
    assert check.status == FAIL
    assert check.fix is not None and "ssh" in check.fix.manual.lower()


def test_display_check_passes_on_x11(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    assert check_display().status == OK


def test_xdotool_check_offers_an_install(monkeypatch):
    monkeypatch.setattr("lai.checks.shutil.which", lambda name: None)
    check = check_xdotool()
    assert check.status == FAIL and check.blocking
    assert check.fix is not None
    assert "xdotool" in (check.fix.shell() or check.fix.manual)


def test_xdotool_check_passes_when_present(monkeypatch):
    monkeypatch.setattr("lai.checks.shutil.which", lambda name: "/usr/bin/xdotool")
    assert check_xdotool().status == OK


def test_ocr_is_optional_when_missing(monkeypatch):
    monkeypatch.setattr("lai.checks.shutil.which", lambda name: None)
    check = check_ocr()
    assert check.status == WARN
    assert not check.required, "OCR missing must never block a run"
    assert not check.blocking


def test_provider_check_fails_with_actionable_guidance(monkeypatch):
    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", list)
    check = check_provider(None)
    assert check.status == FAIL
    assert check.fix is not None
    assert "ANTHROPIC_API_KEY" in check.fix.manual
    assert "lai setup" in check.fix.manual


def test_provider_check_passes_with_a_live_runtime():
    runtime = type("R", (), {"provider": type("P", (), {"name": "zai", "model": "glm-5"})()})()
    check = check_provider(runtime)
    assert check.status == OK and "zai" in check.detail


def test_provider_check_survives_broken_discovery(monkeypatch):
    def explode():
        raise RuntimeError("env is a mess")

    monkeypatch.setattr("lai.agent.providers.registry.discover_credentials", explode)
    check = check_provider(None)
    assert check.status == FAIL and "env is a mess" in check.detail


# -- probes against a fake desktop ---------------------------------------


class FakeDesktop:
    class _Screen:
        def monitors(self):
            bounds = type("B", (), {"width": 1920, "height": 1080})()
            return [type("M", (), {"name": "HDMI-1", "bounds": bounds})()]

    class _Windows:
        def list_windows(self):
            return [object(), object()]

    class _A11y:
        available = True

        def applications(self):
            return [("Xed", 123, None), ("Firefox", 456, None)]

    class _Clipboard:
        available = True

    class _Apps:
        def apps(self):
            return [object()] * 157

    screen = _Screen()
    windows = _Windows()
    a11y = _A11y()
    clipboard = _Clipboard()
    apps = _Apps()

    def snapshot(self, max_elements=40):
        return [object()] * 12


class BrokenDesktop(FakeDesktop):
    class _Screen:
        def monitors(self):
            raise RuntimeError("no X connection")

    class _Windows:
        def list_windows(self):
            raise RuntimeError("no window manager")

    screen = _Screen()
    windows = _Windows()


def test_probes_read_a_working_desktop():
    assert check_screen(FakeDesktop()).status == OK
    assert "1920x1080" in check_screen(FakeDesktop()).detail
    assert check_windows(FakeDesktop()).status == OK
    assert check_clipboard(FakeDesktop()).status == OK
    assert "157" in check_apps(FakeDesktop()).detail


def test_probes_report_a_broken_desktop_as_findings_not_crashes():
    screen = check_screen(BrokenDesktop())
    windows = check_windows(BrokenDesktop())
    assert screen.status == FAIL and "no X connection" in screen.detail
    assert windows.status == FAIL and "no window manager" in windows.detail


def test_a11y_check_flags_an_empty_tree_with_the_gsettings_fix():
    class Silent(FakeDesktop):
        def snapshot(self, max_elements=40):
            return []

    check = check_a11y(Silent())
    assert check.status == WARN
    assert check.fix is not None
    assert "toolkit-accessibility" in check.fix.shell()


def test_a11y_check_passes_with_a_populated_tree():
    check = check_a11y(FakeDesktop())
    assert check.status == OK
    assert "12 element" in check.detail


def test_a11y_check_fails_when_the_bus_is_unreachable():
    class NoBus(FakeDesktop):
        class _A11y:
            available = False

        a11y = _A11y()

    check = check_a11y(NoBus())
    assert check.status == FAIL and check.fix is not None


def test_config_check_notices_a_missing_file(tmp_path):
    config = type("C", (), {"home": tmp_path})()
    check = check_config(config)
    assert check.status == WARN and not check.required
    assert "lai setup" in (check.fix.manual if check.fix else "")


def test_config_check_passes_once_written(tmp_path):
    (tmp_path / "config.toml").write_text("[safety]\nmode='ask'\n", encoding="utf-8")
    config = type("C", (), {"home": tmp_path})()
    assert check_config(config).status == OK


# -- the whole run -------------------------------------------------------


def test_run_checks_covers_every_subsystem():
    report = run_checks(None, None)
    keys = {c.key for c in report}
    assert {"platform", "display", "xdotool", "a11y", "provider"} <= keys


def test_run_checks_can_skip_the_optional_ones():
    keys = {c.key for c in run_checks(None, None, include_optional=False)}
    assert "ocr" not in keys and "recorder" not in keys


def test_run_checks_never_raises_on_a_broken_probe(monkeypatch):
    def explode():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr("lai.checks.check_platform", explode)
    report = run_checks(None, None)
    assert any(c.status == FAIL and "probe exploded" in c.detail for c in report)


def test_package_manager_detection(monkeypatch):
    monkeypatch.setattr("lai.checks.shutil.which", lambda name: "/usr/bin/x" if name == "dnf" else None)
    assert package_manager() == "dnf"
    monkeypatch.setattr("lai.checks.shutil.which", lambda name: None)
    assert package_manager() == ""


@pytest.mark.parametrize(
    ("manager", "expected"),
    [("apt-get", "sudo apt-get install -y xdotool"),
     ("dnf", "sudo dnf install -y xdotool"),
     ("pacman", "sudo pacman -S --noconfirm xdotool"),
     ("zypper", "sudo zypper install -y xdotool")],
)
def test_install_command_adapts_to_the_distribution(monkeypatch, manager, expected):
    monkeypatch.setattr("lai.checks.shutil.which", lambda name: "/usr/bin/x" if name == manager else None)
    check = check_xdotool()
    assert check.fix is not None and check.fix.shell() == expected


def test_install_falls_back_to_instructions_without_a_package_manager(monkeypatch):
    monkeypatch.setattr("lai.checks.shutil.which", lambda name: None)
    check = check_xdotool()
    assert check.fix is not None
    assert not check.fix.automatic
    assert "xdotool" in check.fix.manual


# -- coding agents -------------------------------------------------------


def test_installed_coding_agents_are_named(monkeypatch):
    from lai.checks import check_coders

    monkeypatch.setattr("lai.tools.coding.available_coders", lambda: ["claude", "codex"])
    check = check_coders()
    assert check.status == OK
    assert "claude" in check.detail and "code_agent" in check.detail


def test_no_coding_agent_is_a_note_not_a_problem(monkeypatch):
    """LAI writes files perfectly well itself; a coder is an upgrade."""
    from lai.checks import check_coders

    monkeypatch.setattr("lai.tools.coding.available_coders", list)
    check = check_coders()
    assert check.status == WARN and not check.required and not check.blocking
    assert check.fix is not None and "opencode" in check.fix.manual


def test_a_broken_probe_is_a_warning_not_a_crash(monkeypatch):
    from lai.checks import check_coders

    def explode():
        raise RuntimeError("import blew up")

    monkeypatch.setattr("lai.tools.coding.available_coders", explode)
    assert check_coders().status == WARN


def test_doctor_reports_the_coding_agents():
    from lai.checks import run_checks

    assert run_checks().get("coders") is not None
