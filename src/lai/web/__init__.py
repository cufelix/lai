"""The browser face of LAI.

The same agent, the same daemon, the same desktop gate — reached from a tab
instead of a terminal. The page is one self-contained file served by the
daemon; there is no build step, no bundler and nothing fetched from the
internet, because a tool that controls your desktop should not be loading code
from anywhere else.

Authentication is the daemon's bearer token, handed to the page in the URL
*fragment*: fragments never leave the browser, so the token cannot end up in a
server log or a proxy, and a page opened without one simply cannot do anything.
"""

from __future__ import annotations

from pathlib import Path

PAGE = Path(__file__).with_name("ui.html")


def page() -> bytes:
    """The single-page app, as bytes ready to serve."""
    return PAGE.read_bytes()


def url(host: str, port: int, token: str) -> str:
    """Where a human should point their browser."""
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host  # noqa: S104 - display only
    return f"http://{shown}:{port}/#{token}" if token else f"http://{shown}:{port}/"


__all__ = ["PAGE", "page", "url"]
