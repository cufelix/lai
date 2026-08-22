"""Putting LAI in the applications menu.

Everything else in this project assumes a terminal — a reasonable assumption
for whoever ran the install, and a fatal one for anybody they hand it to.
"""

from __future__ import annotations

import pytest

from lai import launcher


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    return tmp_path


def test_installing_writes_an_entry_and_an_icon(data_home):
    entry = launcher.install()
    assert entry == data_home / "applications" / "lai.desktop"
    assert entry.is_file()
    assert launcher.icon_path().is_file()


def test_the_entry_starts_without_a_terminal(data_home):
    text = launcher.install().read_text(encoding="utf-8")
    assert "Terminal=false" in text
    assert "Exec=" in text and " open" in text
    assert "Name=LAI" in text


def test_the_entry_is_executable(data_home):
    assert launcher.install().stat().st_mode & 0o111


def test_it_uses_an_absolute_path_when_there_is_one(data_home, monkeypatch):
    """A desktop session's PATH is whatever the display manager decided at
    login, and frequently does not include ~/.local/bin."""
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/home/x/.local/bin/lai")
    assert launcher.command() == "/home/x/.local/bin/lai open"


def test_it_falls_back_to_the_bare_name(data_home):
    assert launcher.command() == "lai open"


def test_installing_twice_is_the_same_as_once(data_home):
    first = launcher.install().read_text(encoding="utf-8")
    assert launcher.install().read_text(encoding="utf-8") == first


def test_removing_takes_both_files(data_home):
    launcher.install()
    assert launcher.uninstall() is True
    assert not launcher.entry_path().exists()
    assert not launcher.icon_path().exists()
    assert launcher.uninstall() is False, "nothing left to remove"


def test_it_can_say_whether_it_is_there(data_home):
    assert launcher.installed() is False
    launcher.install()
    assert launcher.installed() is True


def test_a_missing_desktop_database_tool_is_not_a_failure(data_home):
    """Plenty of systems do not ship it, and the entry works regardless."""
    assert launcher.install().is_file()


def test_the_icon_is_a_real_svg(data_home):
    launcher.install()
    text = launcher.icon_path().read_text(encoding="utf-8")
    assert text.startswith("<svg") and "</svg>" in text
