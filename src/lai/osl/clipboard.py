"""Clipboard access via GTK (works without xclip/xsel being installed)."""

from __future__ import annotations

from ..errors import BackendUnavailable


class Clipboard:
    """Read/write the X11 CLIPBOARD and PRIMARY selections."""

    def __init__(self) -> None:
        self._gtk = None
        self._gdk = None

    def _api(self):
        if self._gtk is None:
            try:
                import gi  # noqa: PLC0415

                gi.require_version("Gtk", "3.0")
                gi.require_version("Gdk", "3.0")
                from gi.repository import Gdk, Gtk  # noqa: PLC0415
            except Exception as exc:  # pragma: no cover - env dependent
                raise BackendUnavailable(
                    "GTK clipboard unavailable",
                    detail="install with: sudo apt install python3-gi gir1.2-gtk-3.0",
                ) from exc
            self._gtk, self._gdk = Gtk, Gdk
        return self._gtk, self._gdk

    @property
    def available(self) -> bool:
        """Whether the clipboard can actually be used, not merely imported.

        Saying yes while every operation fails is worse than saying no: it is
        what let `lai doctor` report a working clipboard on a machine where the
        agent could not read one.
        """
        try:
            self._clipboard("clipboard")
            return True
        except BackendUnavailable:
            return False

    def _clipboard(self, selection: str):
        gtk, gdk = self._api()
        atom = gdk.SELECTION_PRIMARY if selection == "primary" else gdk.SELECTION_CLIPBOARD
        board = gtk.Clipboard.get(atom)
        if board is None:
            # GTK imports fine without a display and hands back None here. The
            # next line then asked None for `set_text`, and the tool died with
            # an AttributeError that told the model nothing it could act on.
            raise BackendUnavailable(
                f"the {selection} clipboard could not be obtained",
                detail="GTK is installed but could not reach a display — check DISPLAY",
            )
        return gtk, board

    def _pump(self, gtk) -> None:
        """Drain pending GTK events so the selection transfer completes."""
        for _ in range(200):
            if not gtk.events_pending():
                break
            gtk.main_iteration_do(False)

    def read_text(self, selection: str = "clipboard") -> str:
        gtk, clipboard = self._clipboard(selection)
        self._pump(gtk)
        return clipboard.wait_for_text() or ""

    def write_text(self, text: str, selection: str = "clipboard") -> int:
        if not isinstance(text, str):
            raise BackendUnavailable("clipboard text must be a string")
        gtk, clipboard = self._clipboard(selection)
        clipboard.set_text(text, -1)
        try:
            clipboard.store()
        except Exception:
            pass  # no clipboard manager running; content still available while we live
        self._pump(gtk)
        return len(text)

    def read_image_png(self, selection: str = "clipboard") -> bytes | None:
        gtk, clipboard = self._clipboard(selection)
        self._pump(gtk)
        pixbuf = clipboard.wait_for_image()
        if pixbuf is None:
            return None
        ok, data = pixbuf.save_to_bufferv("png", [], [])
        return bytes(data) if ok else None

    def has_image(self, selection: str = "clipboard") -> bool:
        gtk, clipboard = self._clipboard(selection)
        self._pump(gtk)
        try:
            return bool(clipboard.wait_is_image_available())
        except Exception:
            return False
