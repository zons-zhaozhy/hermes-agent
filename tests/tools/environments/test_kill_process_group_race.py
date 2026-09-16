"""Regression tests for _kill_process_group_posix race tolerance.

Contract (Preconditions): a fake or real Popen-like proc; POSIX only.
Contract (Postconditions): the helper NEVER raises — not on ProcessLookupError
(raced reaper), not on PermissionError (pid/pgid recycled to another owner),
not on a getpgid failure with no cached pgid. The kill path is best-effort;
surfacing its races as [Errno 1]/[Errno 3] to search_files/terminal callers
was the bug (multi-session macOS, 2026-09).
"""

import os
import signal
import subprocess
import sys
from typing import Any, List, Tuple
from unittest.mock import MagicMock

import pytest

from tools.environments.local import _kill_process_group_posix


def _pgid_lookup_error(exc: BaseException) -> Any:
    def _raise(pid: int) -> int:
        raise exc

    return _raise


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
class TestKillProcessGroupPosixRaceTolerance:
    def test_getpgid_process_lookup_without_cache_falls_back_to_pid(self, monkeypatch) -> None:
        # Concurrent reaper wins before getpgid: no cached pgid → must fall
        # back to proc.pid (start_new_session makes pgid == pid), not raise.
        proc = MagicMock()
        proc.pid = 4242
        proc._hermes_pgid = None  # MagicMock would auto-create a truthy attr
        monkeypatch.setattr(os, "getpgid", _pgid_lookup_error(ProcessLookupError()))
        kills: List[Tuple[int, int]] = []
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: kills.append((pgid, sig)))
        monkeypatch.setattr(
            "tools.environments.local._wait_for_group_exit", lambda proc, pgid, t: True
        )
        _kill_process_group_posix(proc)  # must not raise
        assert kills and kills[0][0] == 4242

    def test_getpgid_permission_error_uses_cached_pgid(self, monkeypatch) -> None:
        proc = MagicMock()
        proc.pid = 5151
        proc._hermes_pgid = 9001
        monkeypatch.setattr(os, "getpgid", _pgid_lookup_error(PermissionError()))
        kills: List[Tuple[int, int]] = []
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: kills.append((pgid, sig)))
        monkeypatch.setattr(
            "tools.environments.local._wait_for_group_exit", lambda proc, pgid, t: True
        )
        _kill_process_group_posix(proc)  # must not raise
        assert kills and kills[0][0] == 9001

    def test_killpg_permission_error_is_swallowed(self, monkeypatch) -> None:
        # pgid recycled to another user's process between getpgid and killpg:
        # EPERM from killpg must not surface to the tool result.
        proc = MagicMock()
        proc.pid = 6161
        monkeypatch.setattr(os, "getpgid", lambda pid: 6161)

        def _boom(pgid: int, sig: int) -> None:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(os, "killpg", _boom)
        _kill_process_group_posix(proc)  # must not raise

    def test_killpg_process_lookup_after_term_is_swallowed(self, monkeypatch) -> None:
        # Group exits between TERM and the KILL escalation: ProcessLookupError
        # from the second killpg must not raise.
        proc = MagicMock()
        proc.pid = 7272
        monkeypatch.setattr(os, "getpgid", lambda pid: 7272)
        state = {"n": 0}

        def _killpg(pgid: int, sig: int) -> None:
            state["n"] += 1
            if state["n"] >= 2:
                raise ProcessLookupError()
            return None

        monkeypatch.setattr(os, "killpg", _killpg)
        monkeypatch.setattr(
            "tools.environments.local._wait_for_group_exit", lambda proc, pgid, t: False
        )
        _kill_process_group_posix(proc)  # must not raise

    @pytest.mark.live_system_guard_bypass
    def test_real_short_lived_child_does_not_raise(self) -> None:
        # End-to-end: the child exits on its own; the kill helper races a dead
        # group the whole way and must stay silent.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        proc.wait()
        _kill_process_group_posix(proc)  # must not raise
        assert proc.poll() is not None
