"""Shared pytest fixtures for the LAI OS-layer test suite."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from lai.osl.desktop import Desktop


@pytest.fixture(autouse=True)
def _skip_x11_without_display(request: pytest.FixtureRequest) -> None:
    """Skip any test marked ``x11`` when there is no live X11 display to attach to."""
    if request.node.get_closest_marker("x11") is not None and not os.environ.get("DISPLAY"):
        pytest.skip("no DISPLAY available for an x11 test")


@pytest.fixture
def desktop():
    """A fresh :class:`Desktop` per test, always torn down.

    Teardown swallows errors on purpose: a failure while closing the X11
    connection must not mask the actual assertion that failed in the test.
    """
    instance = Desktop()
    try:
        yield instance
    finally:
        try:
            instance.close()
        except Exception:
            pass


@pytest.fixture
def tmp_desktop_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture: write a ``.desktop`` file with ``content`` under ``name``.

    Usage: ``path = tmp_desktop_file("foo.desktop", "[Desktop Entry]\\n...")``
    """

    def _make(name: str, content: str) -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _make
