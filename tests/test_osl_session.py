"""X session discovery — LAI must find the display when nothing hands it one."""

from __future__ import annotations

import os

import pytest

from lai.osl import session as xsession


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("XAUTHORITY", raising=False)


def test_candidates_prefer_the_current_display(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":7")
    monkeypatch.setattr(xsession, "_displays_from_processes", list)
    assert xsession.candidate_displays()[0] == ":7"


def test_candidates_include_x11_sockets(monkeypatch, tmp_path):
    socket_dir = tmp_path / ".X11-unix"
    socket_dir.mkdir()
    for name in ("X0", "X1", "not-a-socket"):
        (socket_dir / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(xsession, "X11_SOCKET_DIR", socket_dir)
    monkeypatch.setattr(xsession, "_displays_from_processes", list)
    assert xsession.candidate_displays() == [":0", ":1"]


def test_candidates_deduplicate(monkeypatch, tmp_path):
    socket_dir = tmp_path / ".X11-unix"
    socket_dir.mkdir()
    (socket_dir / "X0").write_text("", encoding="utf-8")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(xsession, "X11_SOCKET_DIR", socket_dir)
    monkeypatch.setattr(xsession, "_displays_from_processes", lambda: [":0"])
    assert xsession.candidate_displays() == [":0"]


def test_candidates_include_process_environments(monkeypatch, tmp_path):
    monkeypatch.setattr(xsession, "X11_SOCKET_DIR", tmp_path / "absent")
    monkeypatch.setattr(xsession, "_displays_from_processes", lambda: [":3"])
    assert ":3" in xsession.candidate_displays()


def test_missing_socket_directory_is_survivable(monkeypatch, tmp_path):
    monkeypatch.setattr(xsession, "X11_SOCKET_DIR", tmp_path / "nope")
    monkeypatch.setattr(xsession, "_displays_from_processes", list)
    assert xsession.candidate_displays() == []


def test_ensure_display_sets_the_first_working_candidate(monkeypatch):
    monkeypatch.setattr(xsession, "candidate_displays", lambda: [":9", ":0"])
    monkeypatch.setattr(xsession, "probe_display", lambda name, **kw: name == ":0")
    assert xsession.ensure_display() == ":0"
    assert os.environ["DISPLAY"] == ":0"


def test_ensure_display_honours_a_preference(monkeypatch):
    monkeypatch.setattr(xsession, "candidate_displays", lambda: [":0"])
    monkeypatch.setattr(xsession, "probe_display", lambda name, **kw: True)
    assert xsession.ensure_display(":5") == ":5"
    assert os.environ["DISPLAY"] == ":5"


def test_ensure_display_falls_back_to_a_guess_when_nothing_probes(monkeypatch):
    monkeypatch.setattr(xsession, "candidate_displays", lambda: [":2"])
    monkeypatch.setattr(xsession, "probe_display", lambda name, **kw: False)
    assert xsession.ensure_display() == ""
    # A named display makes the eventual error message actionable.
    assert os.environ.get("DISPLAY") == ":2"


def test_ensure_display_with_no_candidates_at_all(monkeypatch):
    monkeypatch.setattr(xsession, "candidate_displays", list)
    assert xsession.ensure_display() == ""


def test_xauthority_is_filled_in_when_the_file_exists(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".Xauthority").write_text("cookie", encoding="utf-8")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)
    monkeypatch.setattr(xsession, "candidate_displays", list)
    xsession.ensure_display()
    assert os.environ.get("XAUTHORITY") == str(home / ".Xauthority")


def test_existing_xauthority_is_left_alone(monkeypatch):
    monkeypatch.setenv("XAUTHORITY", "/custom/auth")
    monkeypatch.setattr(xsession, "candidate_displays", list)
    xsession.ensure_display()
    assert os.environ["XAUTHORITY"] == "/custom/auth"


def test_probe_of_a_bogus_display_is_false():
    assert xsession.probe_display(":99") is False


def test_session_info_shape(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(xsession, "_displays_from_processes", list)
    info = xsession.session_info()
    assert set(info) == {"display", "xauthority", "session_type", "candidates"}
    assert info["display"] == ":0"


@pytest.mark.x11
def test_real_display_is_discoverable():
    assert xsession.candidate_displays()
    assert xsession.ensure_display()
    assert xsession.probe_display(os.environ["DISPLAY"])


@pytest.mark.x11
def test_display_survives_being_stripped(monkeypatch):
    """The MCP SDK strips DISPLAY from a child's env; LAI must recover it."""
    monkeypatch.delenv("DISPLAY", raising=False)
    resolved = xsession.ensure_display()
    assert resolved, "LAI should rediscover the X session with no DISPLAY set"

    from lai.osl import Desktop

    desktop = Desktop()
    try:
        assert desktop.windows.list_windows() is not None
    finally:
        desktop.close()
