"""Sync↔async bridge for the MCP SDK.

The MCP SDK is async; LAI's tool dispatch is synchronous (a model calls a tool
and blocks on the result). Rather than colouring the whole codebase async, one
background event loop runs in a daemon thread and synchronous callers marshal
work onto it.

The subtlety that dictates the design: ``stdio_client`` and ``ClientSession``
are anyio context managers, and anyio cancel scopes **must be exited by the task
that entered them**. So each server gets a single long-lived task that opens its
contexts and then services commands from a queue — never a pair of
enter/exit calls from different tasks.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from typing import Any

_LOOP: asyncio.AbstractEventLoop | None = None
_THREAD: threading.Thread | None = None
_LOCK = threading.Lock()


def get_loop() -> asyncio.AbstractEventLoop:
    """The shared background event loop, started on first use."""
    global _LOOP, _THREAD
    with _LOCK:
        if _LOOP is not None and not _LOOP.is_closed():
            return _LOOP
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=_run_loop, args=(loop,), name="lai-mcp-loop", daemon=True)
        thread.start()
        _LOOP, _THREAD = loop, thread
        atexit.register(shutdown_loop)
        return loop


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        try:
            loop.close()
        except Exception:
            pass


def run_sync(coro: Awaitable[Any], *, timeout: float = 30.0) -> Any:
    """Run a coroutine on the background loop and block for its result."""
    loop = get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise


def spawn(coro: Awaitable[Any]) -> Future:
    """Start a coroutine on the background loop without waiting for it."""
    loop = get_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]


def call_threadsafe(fn: Callable[[], Any]) -> None:
    loop = get_loop()
    loop.call_soon_threadsafe(fn)


def shutdown_loop() -> None:
    """Stop the background loop. Safe to call more than once."""
    global _LOOP, _THREAD
    with _LOCK:
        loop, thread = _LOOP, _THREAD
        _LOOP, _THREAD = None, None
    if loop is None:
        return
    try:
        loop.call_soon_threadsafe(loop.stop)
    except RuntimeError:
        return
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)
