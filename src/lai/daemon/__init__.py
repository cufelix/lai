"""Daemon mode: LAI as a local HTTP service with live event streaming."""

from .server import DEFAULT_HOST, DEFAULT_PORT, DaemonState, serve

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "DaemonState", "serve"]
