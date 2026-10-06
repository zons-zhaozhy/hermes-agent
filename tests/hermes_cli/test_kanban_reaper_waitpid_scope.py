"""Kanban zombie reaper must not steal exit statuses of unrelated children.

Root cause (2026-10-06, production): the gateway's embedded kanban dispatcher
calls ``reap_worker_zombies()`` every tick, which loops ``os.waitpid(-1,
WNOHANG)`` — reaping EVERY exited child of the gateway process, not just
kanban workers. When it wins the race against a concurrent
``Popen.communicate()`` (cron no_agent scripts, terminal tool subprocesses,
MCP adapters …), the owner's ``waitpid(self.pid)`` raises ``ChildProcessError``
and CPython's ``Popen._try_wait`` then ASSUMES status 0
(python3.14 subprocess.py:2039-2050): a script that exited 1 is recorded as
returncode 0 → the cron ledger writes ``completed`` for a failed run (false
green, observed live at 15:30:56 on job f234e58443b7).

Contract:
  Preconditions: POSIX host (waitpid semantics); a real short-lived child.
  Postconditions: the reaper only ever waits on PIDs it registered itself;
  an unregistered exited child stays reaped-able by its own owner.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb  # noqa: F401  -- kbd re-exports its constants
from hermes_cli import kanban_db_dispatch as kbd

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX waitpid stealing semantics")

BANNER = "=== L0 Watchdog Alerts ==="
EXPECTED_TRUE_EXIT = 1  # 期望: 子进程代码字面即 SystemExit(1)，真实退出码只能是 1


def _spawn_exit_1_with_banner() -> subprocess.Popen:
    """Real child: prints the alert banner then exits 1 (production shape)."""
    return subprocess.Popen(
        [sys.executable, "-c", f"print('{BANNER}'); raise SystemExit({EXPECTED_TRUE_EXIT})"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


@pytest.mark.platforms("macos", "linux")
def test_reaper_never_reaps_unregistered_pid(monkeypatch):
    """The POSIX reaper must only wait on PIDs it registered.

    Red on base: ``waitpid(-1)`` reaps ANY exited child — including cron
    scripts and terminal-tool subprocesses the reaper never spawned. On a
    clean-process base this is the registry-scope contract itself.
    """
    monkeypatch.setattr(kbd, "_live_worker_procs", {})
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})

    # A child the reaper never registered, already exited (its owner is about
    # to communicate()). The dispatcher tick fires reaping here.
    proc = _spawn_exit_1_with_banner()
    stdout = proc.stdout.read()  # drain to EOF: child has now exited
    proc.stderr.close()

    reaped = kbd.reap_worker_zombies()

    # 期望: BANNER 在 stdout——子进程第一行就是 print(BANNER)
    assert BANNER in stdout, "alert banner must be printed"
    # 期望: proc.pid 不在 reaped——该 pid 从未被 reaper 登记，收割它即偷状态
    assert proc.pid not in reaped, (
        "reaper reaped an unregistered child: it will steal that child's "
        "exit status from its owner (cron false-green race)")
    # The owner still gets the TRUE status — the invariant the fix protects.
    # 期望: rc==EXPECTED_TRUE_EXIT(=1)——子进程代码即 SystemExit(1)，未被偷时恒为 1
    assert proc.wait() == EXPECTED_TRUE_EXIT
