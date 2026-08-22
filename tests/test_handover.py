"""Giving the work back when the run ends.

An X window belongs to the display its client connected to, so the agent's
browser cannot cross to yours — it dies with the server. What was *in* it can
come across, and that is the part that mattered.
"""

from __future__ import annotations

import pytest

from lai.osl.handover import Handoff, collect, deliver


class FakeElement:
    def __init__(self, role, value="", name=""):
        self.role, self.value, self.name = role, value, name


class FakeWindow:
    def __init__(self, wm_class, window_id=1, title=""):
        self.wm_class, self.id, self.title = wm_class, window_id, title


class FakeDesktop:
    def __init__(self, windows, elements=()):
        self._windows = list(windows)
        self._elements = list(elements)
        self.focused = []
        self.windows = self

    def list_windows(self):
        return self._windows

    def focus(self, window_id):
        self.focused.append(window_id)

    def snapshot(self, **kwargs):
        return type("S", (), {"elements": self._elements})()


# -- what is worth carrying across ---------------------------------------


def test_the_page_a_browser_was_showing_comes_across():
    desktop = FakeDesktop(
        [FakeWindow("firefox")],
        [FakeElement("entry", "jspaint.app/#local:abc")],
    )
    assert collect(desktop) == [Handoff("url", "https://jspaint.app/#local:abc", "firefox")]


def test_a_full_url_is_kept_as_it_is():
    desktop = FakeDesktop(
        [FakeWindow("chromium")], [FakeElement("entry", "https://example.test/x?y=1")]
    )
    assert collect(desktop)[0].target == "https://example.test/x?y=1"


def test_the_address_bar_is_matched_on_its_value_not_its_label():
    """The label is translated into whatever language the browser is in; the
    value is a URL in every one of them."""
    desktop = FakeDesktop(
        [FakeWindow("firefox")],
        [FakeElement("entry", "example.test", name="Suchen oder Adresse eingeben")],
    )
    assert collect(desktop)[0].target == "https://example.test"


def test_a_search_box_is_not_an_address_bar():
    desktop = FakeDesktop(
        [FakeWindow("firefox")], [FakeElement("entry", "how tall is the eiffel tower")]
    )
    assert collect(desktop) == []


def test_a_browser_showing_nothing_hands_nothing_over():
    desktop = FakeDesktop([FakeWindow("firefox")], [FakeElement("entry", "about:blank")])
    assert collect(desktop) == []
    desktop = FakeDesktop([FakeWindow("firefox")], [FakeElement("entry", "chrome://settings")])
    assert collect(desktop) == []


def test_windows_that_are_not_browsers_are_skipped():
    """An editor's unsaved buffer cannot be reconstructed, and pretending
    otherwise would be worse than saying so."""
    desktop = FakeDesktop([FakeWindow("xed"), FakeWindow("Gimp-2.10")])
    assert collect(desktop) == []
    assert desktop.focused == [], "no need to disturb a window there is nothing to take from"


def test_files_the_run_wrote_come_across(tmp_path):
    written = tmp_path / "report.md"
    written.write_text("done", encoding="utf-8")
    found = collect(FakeDesktop([]), artifacts=[str(written)])
    assert found == [Handoff("file", str(written.resolve()))]


def test_a_file_that_is_not_there_is_not_offered(tmp_path):
    assert collect(FakeDesktop([]), artifacts=[str(tmp_path / "gone.txt")]) == []


def test_the_same_thing_twice_is_handed_over_once(tmp_path):
    written = tmp_path / "a.txt"
    written.write_text("x", encoding="utf-8")
    found = collect(FakeDesktop([]), artifacts=[str(written), str(written)])
    assert len(found) == 1


def test_a_broken_desktop_hands_over_what_it_can(tmp_path):
    class Broken(FakeDesktop):
        def list_windows(self):
            raise RuntimeError("x server gone")

    written = tmp_path / "a.txt"
    written.write_text("x", encoding="utf-8")
    assert collect(Broken([]), artifacts=[str(written)])[0].kind == "file"


# -- delivering it -------------------------------------------------------


def test_each_thing_is_opened_on_the_human_display(monkeypatch):
    launched = []

    def popen(command, **kwargs):
        launched.append((command, kwargs["env"]["DISPLAY"]))
        return object()

    monkeypatch.setattr("lai.osl.handover.subprocess.Popen", popen)
    monkeypatch.setattr("lai.osl.handover._opener", lambda: "/usr/bin/xdg-open")

    opened, problem = deliver(
        [Handoff("url", "https://a.test"), Handoff("file", "/tmp/x.png")], display=":0"
    )
    assert problem == ""
    assert len(opened) == 2
    assert launched[0] == (["/usr/bin/xdg-open", "https://a.test"], ":0")
    assert launched[1][1] == ":0", "the agent's display is about to be shut down"


def test_nothing_to_hand_over_is_not_a_problem():
    assert deliver([]) == ([], "")


def test_no_opener_is_reported_rather_than_silently_skipped(monkeypatch):
    monkeypatch.setattr("lai.osl.handover._opener", lambda: None)
    opened, problem = deliver([Handoff("url", "https://a.test")])
    assert opened == [] and "xdg-open" in problem


def test_one_failure_does_not_stop_the_rest(monkeypatch):
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        if "bad" in command[1]:
            raise OSError("no")
        return object()

    monkeypatch.setattr("lai.osl.handover.subprocess.Popen", popen)
    monkeypatch.setattr("lai.osl.handover._opener", lambda: "/usr/bin/xdg-open")
    opened, problem = deliver([Handoff("url", "bad"), Handoff("url", "https://good.test")])
    assert [h.target for h in opened] == ["https://good.test"]
    assert "bad" in problem


def test_it_says_where_each_thing_came_from():
    assert "from firefox" in Handoff("url", "https://a.test", "firefox").describe()
    assert Handoff("file", "/tmp/x").describe() == "/tmp/x"


# -- only when there was a separate screen -------------------------------


def test_nothing_is_handed_over_when_it_was_already_your_desktop(tmp_path):
    """Reopening a window the person is looking at is a second copy of it."""
    from lai.config import load_config
    from lai.runtime import Runtime

    runtime = Runtime.__new__(Runtime)
    runtime.virtual_display = None
    runtime.config = load_config().with_overrides(home=tmp_path)
    assert runtime.hand_over(["/tmp/x"]) == ([], "")


def test_handover_can_be_switched_off(tmp_path):
    from dataclasses import replace

    from lai.config import load_config
    from lai.runtime import Runtime

    config = load_config().with_overrides(home=tmp_path)
    runtime = Runtime.__new__(Runtime)
    runtime.virtual_display = object()
    runtime.config = config.with_overrides(desktop=replace(config.desktop, handover=False))
    assert runtime.hand_over(["/tmp/x"]) == ([], "")
