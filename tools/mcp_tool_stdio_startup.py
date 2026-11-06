"""Deterministic stdio MCP startup failures recovered from child stderr."""

import re
from typing import Optional


class StdioStartupFileMissingError(RuntimeError):
    """A stdio child could not open its configured entrypoint.

    Every retry spawns the same command with the same missing path, so the server parks
    until the configuration or checkout is repaired.
    """


_MISSING_ENTRYPOINT_RE = re.compile(
    r"can't open file '([^']+)': \[Errno 2\] No such file or directory")


def stdio_startup_error(stderr_text: str) -> Optional[StdioStartupFileMissingError]:
    """A permanent startup error from *stderr_text*, or ``None`` when unrecognized."""
    if not stderr_text:
        return None
    found = _MISSING_ENTRYPOINT_RE.search(stderr_text)
    if found is None:
        return None
    path = found.group(1)
    return StdioStartupFileMissingError(
        f"stdio child cannot open entrypoint '{path}' ([Errno 2] No such file or directory); "
        "restore the file or update this MCP server's args")
