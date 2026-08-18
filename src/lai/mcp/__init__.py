"""MCP: consume tools from other servers, and expose the desktop as a server."""

from .client import (
    MCPClient,
    MCPPool,
    MCPServerConfig,
    classify_risk,
    load_mcp_configs,
    register_mcp_tools,
)
from .server import DesktopToolServer, build_mcp_server, run_stdio, serve_stdio

__all__ = [
    "DesktopToolServer",
    "MCPClient",
    "MCPPool",
    "MCPServerConfig",
    "build_mcp_server",
    "classify_risk",
    "load_mcp_configs",
    "register_mcp_tools",
    "run_stdio",
    "serve_stdio",
]
