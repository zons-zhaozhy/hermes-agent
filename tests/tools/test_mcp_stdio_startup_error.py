"""Deterministic stdio MCP startup failures must park instead of retrying."""

import pytest
from contextlib import asynccontextmanager

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_errors import _classify_mcp_failure
from tools.mcp_tool_stdio_startup import (
    StdioStartupFileMissingError,
    stdio_startup_error,
)


def test_missing_child_entrypoint_is_extracted_and_permanent():
    stderr = (
        "/repo/.venv/bin/python: can't open file '/repo/missing/server.py': "
        "[Errno 2] No such file or directory\n"
    )

    error = stdio_startup_error(stderr)

    assert isinstance(error, StdioStartupFileMissingError)
    assert "/repo/missing/server.py" in str(error)
    assert _classify_mcp_failure(error) == "permanent"


def test_unrecognized_startup_stderr_stays_none():
    assert stdio_startup_error("ordinary diagnostics") is None


@pytest.mark.asyncio
async def test_stdio_transport_promotes_missing_entrypoint_from_child_stderr(monkeypatch):
    """A child that exits before MCP wraps only 'Connection closed'; Hermes must preserve the cause."""
    stderr_tail = (
        "/repo/.venv/bin/python: can't open file '/repo/missing/server.py': "
        "[Errno 2] No such file or directory\n"
    )

    class CapturedStderr:
        sink = object()

        def __init__(self, _log_fh):
            pass

        def close(self, timeout: float = 2.0) -> str:
            return stderr_tail

    @asynccontextmanager
    async def failing_stdio_client(_params, errlog):
        raise RuntimeError("Connection closed")
        yield  # pragma: no cover - making this an async context manager matters

    async def preflight(_name, command, args):
        return command, args

    monkeypatch.setattr("tools.mcp_tool._ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr("tools.mcp_tool.StdioServerParameters", lambda **_kwargs: object())
    monkeypatch.setattr("tools.mcp_tool.stdio_client", failing_stdio_client)
    monkeypatch.setattr("tools.mcp_tool_transport._connect_inputs", lambda _name, _config: ([[], set(), None], set()))
    monkeypatch.setattr("tools.mcp_tool_registration._identity_digest", lambda _inputs: "identity")
    monkeypatch.setattr("tools.mcp_tool._preflight_stdio_command", preflight)
    monkeypatch.setattr("tools.mcp_tool_lifecycle._kill_orphaned_mcp_children", lambda: None)
    monkeypatch.setattr("tools.mcp_tool_config._write_stderr_log_header", lambda _name: None)
    monkeypatch.setattr("tools.mcp_tool_config._StderrTee", CapturedStderr)

    with pytest.raises(StdioStartupFileMissingError, match="/repo/missing/server.py"):
        await MCPServerTask("srv")._run_stdio({"command": "python", "args": ["/repo/missing/server.py"]})
