"""Tests for lai.osl.apps.

_parse_desktop_file and AppEntry are pure, file-based logic and are tested
against real .desktop files written to tmp_path. The @pytest.mark.x11 test
only reads the system's installed .desktop files — it never launches anything.
"""

from __future__ import annotations

import pytest

from lai.osl.apps import AppEntry, AppLauncher, _parse_desktop_file
from lai.osl.windows import WindowManager

# -- _parse_desktop_file ------------------------------------------------------


def test_parse_normal_entry(tmp_desktop_file):
    path = tmp_desktop_file(
        "foo.desktop",
        """[Desktop Entry]
Type=Application
Name=Foo Editor
Exec=/usr/bin/foo-editor %U
Comment=Edits foos
Categories=Utility;TextEditor;
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.id == "foo"
    assert entry.name == "Foo Editor"
    assert entry.exec_line == "/usr/bin/foo-editor %U"
    assert entry.comment == "Edits foos"
    assert entry.categories == ("Utility", "TextEditor")
    assert entry.path == str(path)


def test_parse_ignores_localized_name_keys(tmp_desktop_file):
    path = tmp_desktop_file(
        "foo.desktop",
        """[Desktop Entry]
Type=Application
Name=Foo Editor
Name[cs]=Editor Foo
Exec=/usr/bin/foo-editor
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.name == "Foo Editor"


def test_parse_type_link_returns_none(tmp_desktop_file):
    path = tmp_desktop_file(
        "link.desktop",
        """[Desktop Entry]
Type=Link
Name=Some Website
URL=https://example.com
""",
    )
    assert _parse_desktop_file(path) is None


def test_parse_missing_exec_returns_none(tmp_desktop_file):
    path = tmp_desktop_file(
        "noexec.desktop",
        """[Desktop Entry]
Type=Application
Name=No Exec App
""",
    )
    assert _parse_desktop_file(path) is None


def test_parse_no_display_true_sets_no_display(tmp_desktop_file):
    path = tmp_desktop_file(
        "hidden1.desktop",
        """[Desktop Entry]
Type=Application
Name=Hidden App
Exec=/usr/bin/hidden
NoDisplay=true
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.no_display is True


def test_parse_hidden_true_sets_no_display(tmp_desktop_file):
    path = tmp_desktop_file(
        "hidden2.desktop",
        """[Desktop Entry]
Type=Application
Name=Hidden App 2
Exec=/usr/bin/hidden2
Hidden=true
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.no_display is True


def test_parse_no_display_and_hidden_default_false(tmp_desktop_file):
    path = tmp_desktop_file(
        "visible.desktop",
        """[Desktop Entry]
Type=Application
Name=Visible App
Exec=/usr/bin/visible
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.no_display is False


def test_parse_startup_wm_class_captured(tmp_desktop_file):
    path = tmp_desktop_file(
        "wm.desktop",
        """[Desktop Entry]
Type=Application
Name=WM App
Exec=/usr/bin/wm-app
StartupWMClass=WmAppClass
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.wm_class == "WmAppClass"


def test_parse_categories_split(tmp_desktop_file):
    path = tmp_desktop_file(
        "cats.desktop",
        """[Desktop Entry]
Type=Application
Name=Cats App
Exec=/usr/bin/cats-app
Categories=Graphics;Photography;2DGraphics;
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.categories == ("Graphics", "Photography", "2DGraphics")


def test_parse_tolerates_comments_and_blank_lines(tmp_desktop_file):
    path = tmp_desktop_file(
        "commented.desktop",
        """# this is a leading comment
[Desktop Entry]
# another comment
Type=Application

Name=Commented App
# comment between keys
Exec=/usr/bin/commented-app

Comment=Has comments and blanks
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.name == "Commented App"
    assert entry.exec_line == "/usr/bin/commented-app"
    assert entry.comment == "Has comments and blanks"


def test_parse_second_group_terminates_parsing(tmp_desktop_file):
    path = tmp_desktop_file(
        "twogroups.desktop",
        """[Desktop Entry]
Type=Application
Name=Main App
Exec=/usr/bin/main-app
[Some Other Group]
Comment=Should never be read
Exec=/bin/should-not-be-used
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.name == "Main App"
    assert entry.exec_line == "/usr/bin/main-app"
    # Comment only appears after the second [group] header -> must be ignored.
    assert entry.comment == ""


def test_parse_missing_name_falls_back_to_stem(tmp_desktop_file):
    path = tmp_desktop_file(
        "stemname.desktop",
        """[Desktop Entry]
Type=Application
Exec=/usr/bin/stemname
""",
    )
    entry = _parse_desktop_file(path)
    assert entry is not None
    assert entry.name == "stemname"


def test_parse_unreadable_path_returns_none(tmp_path):
    missing = tmp_path / "does-not-exist.desktop"
    assert _parse_desktop_file(missing) is None


# -- AppEntry.command ------------------------------------------------------


@pytest.mark.parametrize(
    "exec_line",
    [
        '/usr/bin/app --flag "quoted value" %U',
        '/usr/bin/app --flag "quoted value" %f',
        '/usr/bin/app --flag "quoted value" %F',
        '/usr/bin/app --flag "quoted value" %i',
        '/usr/bin/app --flag "quoted value" %c',
    ],
)
def test_app_entry_command_strips_field_codes_and_splits_quoted_args(exec_line):
    entry = AppEntry(id="app", name="App", exec_line=exec_line, path="/x/app.desktop")
    assert entry.command == ["/usr/bin/app", "--flag", "quoted value"]


def test_app_entry_command_strips_multiple_field_codes():
    entry = AppEntry(
        id="app",
        name="App",
        exec_line="/usr/bin/app %U %f %F %i %c --verbose",
        path="/x/app.desktop",
    )
    assert entry.command == ["/usr/bin/app", "--verbose"]


def test_app_entry_command_no_field_codes():
    entry = AppEntry(id="app", name="App", exec_line="/usr/bin/app --flag", path="/x/app.desktop")
    assert entry.command == ["/usr/bin/app", "--flag"]


def test_app_entry_command_falls_back_to_plain_split_on_unbalanced_quotes():
    entry = AppEntry(id="app", name="App", exec_line='/usr/bin/app "unterminated', path="/x")
    # shlex.split raises ValueError on unbalanced quotes; command() must not raise.
    assert entry.command == ["/usr/bin/app", '"unterminated']


# -- AppEntry.score ------------------------------------------------------


@pytest.fixture
def firefox_entry():
    return AppEntry(
        id="firefox",
        name="Firefox",
        exec_line="/usr/bin/firefox %u",
        path="/x/firefox.desktop",
        comment="Browse the web",
        wm_class="Firefox",
    )


def test_score_exact_name_match_is_100(firefox_entry):
    assert firefox_entry.score("firefox") == 100.0
    assert firefox_entry.score("Firefox") == 100.0  # case-insensitive


def test_score_exact_wm_class_match_is_100(firefox_entry):
    assert firefox_entry.score("Firefox") == 100.0


def test_score_prefix_match(firefox_entry):
    assert firefox_entry.score("fire") == 80.0


def test_score_substring_match(firefox_entry):
    # "refox" is a substring of "firefox" but not a prefix.
    assert firefox_entry.score("refox") == 60.0


def test_score_no_match_is_zero(firefox_entry):
    assert firefox_entry.score("zzzznotfound") == 0.0


def test_score_empty_query_is_zero(firefox_entry):
    assert firefox_entry.score("") == 0.0
    assert firefox_entry.score("   ") == 0.0


# -- AppLauncher.find ordering (monkeypatched app list) ------------------------------------------------------


def test_find_orders_by_score_descending(monkeypatch):
    launcher = AppLauncher(window_manager=WindowManager())
    exact = AppEntry(id="a-exact", name="Terminal", exec_line="/bin/a", path="/x/a")
    prefix = AppEntry(id="b-prefix", name="Terminal Emulator", exec_line="/bin/b", path="/x/b")
    substring = AppEntry(id="c-substring", name="My Terminal App", exec_line="/bin/c", path="/x/c")
    no_match = AppEntry(id="d-none", name="Calculator", exec_line="/bin/d", path="/x/d")

    monkeypatch.setattr(
        launcher, "apps", lambda **kwargs: [no_match, substring, prefix, exact]
    )

    results = launcher.find("terminal")
    assert [e.id for e in results] == ["a-exact", "b-prefix", "c-substring"]


def test_find_respects_limit(monkeypatch):
    launcher = AppLauncher(window_manager=WindowManager())
    entries = [
        AppEntry(id=f"term-{i}", name=f"Terminal {i}", exec_line=f"/bin/t{i}", path=f"/x/t{i}")
        for i in range(5)
    ]
    monkeypatch.setattr(launcher, "apps", lambda **kwargs: entries)
    results = launcher.find("terminal", limit=2)
    assert len(results) == 2


def test_find_one_raises_app_not_found_when_nothing_matches(monkeypatch):
    from lai.errors import AppNotFound

    launcher = AppLauncher(window_manager=WindowManager())
    monkeypatch.setattr(launcher, "apps", lambda **kwargs: [])
    with pytest.raises(AppNotFound):
        launcher.find_one("nonexistent")


# -- x11: real installed apps ------------------------------------------------------


@pytest.mark.x11
def test_apps_returns_non_empty_list_with_valid_entries():
    launcher = AppLauncher(window_manager=WindowManager())
    entries = launcher.apps()
    assert len(entries) > 0
    for entry in entries:
        assert entry.name.strip() != ""
        assert entry.exec_line.strip() != ""


# -- a browser that is actually a second browser -------------------------


def test_a_browser_is_given_its_own_profile(tmp_path):
    """Browsers are single-instance: start one while another is running and it
    hands your request to that one and exits. On the agent's own display the
    window then appears on *your* desktop, or nowhere, and the launcher reports
    "no window appeared" — which is what it did."""
    from lai.osl.apps import isolate_browser

    command = isolate_browser(["/usr/bin/firefox", "https://example.test"], tmp_path)
    assert command[0] == "/usr/bin/firefox"
    assert "--no-remote" in command
    assert str(tmp_path / "firefox") in command
    assert command[-1] == "https://example.test", "flags must precede the URL"


def test_chromium_family_gets_a_user_data_dir(tmp_path):
    from lai.osl.apps import isolate_browser

    command = isolate_browser(["/usr/bin/google-chrome-stable"], tmp_path)
    assert f"--user-data-dir={tmp_path / 'google-chrome-stable'}" in command
    assert "--no-first-run" in command


def test_the_profile_directory_is_created(tmp_path):
    from lai.osl.apps import isolate_browser

    isolate_browser(["/usr/bin/firefox"], tmp_path)
    assert (tmp_path / "firefox").is_dir()


def test_anything_that_is_not_a_browser_is_left_alone(tmp_path):
    from lai.osl.apps import isolate_browser

    assert isolate_browser(["/usr/bin/xed", "a.txt"], tmp_path) == ["/usr/bin/xed", "a.txt"]
    assert isolate_browser([], tmp_path) == []


def test_an_explicit_profile_wins(tmp_path):
    """Somebody who said which profile to use meant it."""
    from lai.osl.apps import isolate_browser

    command = ["/usr/bin/firefox", "--profile", "/home/me/work"]
    assert isolate_browser(command, tmp_path) == command


def test_the_launcher_leaves_browsers_alone_on_your_own_desktop(tmp_path):
    """Sharing your desktop is the one case where handing the URL to the
    browser you already have open is exactly right."""
    from lai.osl.apps import AppLauncher

    assert AppLauncher(window_manager=object()).browser_profile is None


def test_a_second_firefox_on_this_screen_reuses_the_first(tmp_path):
    """`--no-remote` is what stops Firefox handing the URL to the copy on your
    desktop — and it also refuses to start a second copy on this profile. Once
    one is up on the agent's screen, the URL belongs to that one."""
    from lai.osl.apps import isolate_browser

    command = isolate_browser(
        ["/usr/bin/firefox", "https://x.test"], tmp_path, already_running=True
    )
    assert "--no-remote" not in command
    assert str(tmp_path / "firefox") in command, "still its own profile, not yours"


def test_chromium_needs_no_such_care(tmp_path):
    """The same --user-data-dir reaches the same instance either way."""
    from lai.osl.apps import isolate_browser

    fresh = isolate_browser(["/usr/bin/chromium"], tmp_path)
    again = isolate_browser(["/usr/bin/chromium"], tmp_path, already_running=True)
    assert fresh == again


def test_the_launcher_asks_whether_a_window_is_already_there(tmp_path):
    from lai.osl.apps import AppLauncher

    class Windows:
        def __init__(self, classes):
            self.classes = classes

        def list_windows(self):
            return [type("W", (), {"wm_class": c})() for c in self.classes]

    launcher = AppLauncher(Windows(["firefox", "Gnome-terminal"]), browser_profile=tmp_path)
    assert launcher._has_window_for("/usr/bin/firefox") is True
    assert launcher._has_window_for("/usr/bin/google-chrome-stable") is False


def test_a_broken_window_list_does_not_stop_a_launch(tmp_path):
    from lai.osl.apps import AppLauncher

    class Broken:
        def list_windows(self):
            raise RuntimeError("x server gone")

    assert AppLauncher(Broken(), browser_profile=tmp_path)._has_window_for("firefox") is False


def test_chromium_is_launched_with_its_accessibility_tree_on(tmp_path):
    """Without it a Chromium window is one opaque rectangle to AT-SPI — not
    the page, not even the address bar. The agent is reduced to reading pixels,
    and the handover cannot tell what page was open."""
    from lai.osl.apps import isolate_browser

    for binary in ("google-chrome-stable", "chromium", "brave-browser", "microsoft-edge"):
        command = isolate_browser([f"/usr/bin/{binary}"], tmp_path)
        assert "--force-renderer-accessibility" in command, binary


def test_firefox_needs_no_such_flag(tmp_path):
    from lai.osl.apps import isolate_browser

    assert "--force-renderer-accessibility" not in isolate_browser(["/usr/bin/firefox"], tmp_path)


# -- Electron apps are opaque until asked not to be -----------------------


def make_electron(tmp_path, name="cursor"):
    """The layout every Electron app on Linux actually has."""
    root = tmp_path / "share" / name
    (root / "bin").mkdir(parents=True)
    binary = root / "bin" / name
    binary.write_text("#!/bin/sh\nexec real\n")
    binary.chmod(0o755)
    (root / "chrome_crashpad_handler").write_text("")
    return binary


def test_an_electron_app_is_recognised(tmp_path):
    from lai.osl.apps import is_chromium_app

    assert is_chromium_app(make_electron(tmp_path)) is True


def test_an_ordinary_program_is_not(tmp_path):
    from lai.osl.apps import is_chromium_app

    plain = tmp_path / "bin" / "xed"
    plain.parent.mkdir(parents=True)
    plain.write_text("#!/bin/sh\n")
    assert is_chromium_app(plain) is False


def test_a_missing_binary_is_not(tmp_path):
    from lai.osl.apps import is_chromium_app

    assert is_chromium_app(tmp_path / "nothing" / "here") is False


def test_launching_one_turns_its_accessibility_tree_on(tmp_path):
    """Without the flag, Cursor, VS Code, Slack and Discord are one opaque
    rectangle to AT-SPI, and the agent is reduced to clicking pixels. It spent
    a whole run doing exactly that."""
    from lai.osl.apps import accessible_command

    binary = make_electron(tmp_path)
    command = accessible_command([str(binary), "/tmp/notes.txt"])
    assert "--force-renderer-accessibility" in command
    assert command[0] == str(binary)
    assert command[-1] == "/tmp/notes.txt", "flags belong before the positional"


def test_an_ordinary_program_is_launched_untouched(tmp_path):
    from lai.osl.apps import accessible_command

    plain = tmp_path / "xed"
    plain.write_text("#!/bin/sh\n")
    assert accessible_command([str(plain), "a.txt"]) == [str(plain), "a.txt"]


def test_the_flag_is_not_added_twice(tmp_path):
    from lai.osl.apps import accessible_command

    binary = make_electron(tmp_path)
    once = accessible_command([str(binary), "--force-renderer-accessibility"])
    assert once.count("--force-renderer-accessibility") == 1


def test_an_empty_command_is_left_alone():
    from lai.osl.apps import accessible_command

    assert accessible_command([]) == []
