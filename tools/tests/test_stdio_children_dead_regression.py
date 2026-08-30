"""Regression test: _stdio_children_dead liveness predicate.

Guards against the inverted predicate introduced by the 887-commit merge
(conflict resolution broke the loop so a LIVE child reported 'all dead',
fast-failing every stdio MCP call in ~0.03s). Aligned with upstream fixes
98fce8e52d / 2663117f72 / 8aae2ea539.
"""
import psutil

SRC = "tools/mcp_tool.py"


def _extract_method(source: str) -> str:
    """Pull the _stdio_children_dead method body via str methods (no regex)."""
    start = source.find("    def _stdio_children_dead")
    assert start >= 0, "method _stdio_children_dead not found"
    end_marker = "        return True  # every tracked child has exited\n"
    end = source.find(end_marker, start)
    assert end >= 0, "method end anchor not found — update extraction anchor"
    return source[start : end + len(end_marker)]


code = _extract_method(open(SRC).read())

ns: dict = {}
exec("class _T:\n    def _is_http(self):\n        return False\n" + code, ns)
_T = ns["_T"]

LIVE_PID = next((p.pid for p in psutil.process_iter() if p.pid != 0), None)
assert LIVE_PID, "need at least one live pid for the test"
DEAD_PID = 999_999  # nonexistent


def test_no_pids_unknown_is_not_dead():
    t = _T()
    t._stdio_child_pids = None
    assert t._stdio_children_dead() is False


def test_live_children_not_dead():
    t = _T()
    t._stdio_child_pids = [LIVE_PID]
    assert t._stdio_children_dead() is False, "live children must NOT be 'all dead'"


def test_dead_pid_is_all_dead():
    t = _T()
    t._stdio_child_pids = [DEAD_PID]
    assert t._stdio_children_dead() is True


def test_mixed_live_and_dead_not_dead():
    t = _T()
    t._stdio_child_pids = [DEAD_PID, LIVE_PID]
    assert t._stdio_children_dead() is False


def test_http_server_never_fast_fails():
    class _H(_T):
        def _is_http(self):
            return True

    t = _H()
    t._stdio_child_pids = [DEAD_PID]
    assert t._stdio_children_dead() is False
