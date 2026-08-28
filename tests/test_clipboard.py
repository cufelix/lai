import pytest

# -- GTK present, but with no display to talk to -------------------------


def test_a_clipboard_that_could_not_be_obtained_is_reported_not_crashed(monkeypatch):
    """`Gtk.Clipboard.get()` returns None when GTK cannot reach a display. The
    next line asked it for `set_text` and the tool died with an AttributeError
    the model could do nothing with — twice, in this machine's logs."""
    from lai.errors import BackendUnavailable
    from lai.osl.clipboard import Clipboard

    class NoClipboard:
        SELECTION_CLIPBOARD = object()
        SELECTION_PRIMARY = object()

        class Clipboard:
            @staticmethod
            def get(atom):
                return None

    board = Clipboard()
    monkeypatch.setattr(board, "_api", lambda: (NoClipboard, NoClipboard))

    with pytest.raises(BackendUnavailable, match="clipboard"):
        board.write_text("hello")
    with pytest.raises(BackendUnavailable, match="clipboard"):
        board.read_text()


def test_the_message_says_what_to_do_about_it(monkeypatch):
    from lai.errors import BackendUnavailable
    from lai.osl.clipboard import Clipboard

    class NoClipboard:
        SELECTION_CLIPBOARD = object()
        SELECTION_PRIMARY = object()

        class Clipboard:
            @staticmethod
            def get(atom):
                return None

    board = Clipboard()
    monkeypatch.setattr(board, "_api", lambda: (NoClipboard, NoClipboard))
    try:
        board.write_text("hello")
    except BackendUnavailable as exc:
        assert "DISPLAY" in str(exc) + str(getattr(exc, "detail", ""))


def test_available_is_false_when_the_clipboard_cannot_be_obtained(monkeypatch):
    """Reporting available when every operation fails is worse than useless —
    `lai doctor` said the clipboard was fine while the agent could not use it."""
    from lai.osl.clipboard import Clipboard

    class NoClipboard:
        SELECTION_CLIPBOARD = object()
        SELECTION_PRIMARY = object()

        class Clipboard:
            @staticmethod
            def get(atom):
                return None

    board = Clipboard()
    monkeypatch.setattr(board, "_api", lambda: (NoClipboard, NoClipboard))
    assert board.available is False
