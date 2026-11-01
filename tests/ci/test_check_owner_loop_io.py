"""The owner-loop I/O gate blocks direct gateway coroutine I/O but accepts to_thread closures."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ci"))

from check_owner_loop_io import scan_path


def _scan(tmp_path: Path, body: str):
    path = tmp_path / "gateway" / "example.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return scan_path(path)


def test_direct_io_in_the_async_scope_is_blocked(tmp_path: Path) -> None:
    findings = _scan(tmp_path, '''\
import asyncio
import subprocess
import time
from x import discover_plugins, load_gateway_config


async def bad(db, cursor):
    rows = db._read_all("select 1")
    with db._read_ctx() as conn:
        row = conn.execute("select 1").fetchone()
    cursor.execute("select 1")
    data = Path("x").read_text()
    cfg = load_gateway_config()
    discover_plugins()
    proc = subprocess.run(["ls"])
    time.sleep(1)
''')
    rules = sorted(item.rule for item in findings)
    assert rules == sorted([
        "owner-loop.db-helper", "owner-loop.db-helper",
        "owner-loop.db-cursor", "owner-loop.db-cursor", "owner-loop.db-cursor",
        "owner-loop.file-io", "owner-loop.sync-config", "owner-loop.sync-config",
        "owner-loop.subprocess", "owner-loop.sync-sleep",
    ])


def test_nested_offloop_closure_and_marked_exception_are_allowed(tmp_path: Path) -> None:
    findings = _scan(tmp_path, '''\
import asyncio


async def good(db):
    def read():
        return db._read_all("select 1")

    rows = await asyncio.to_thread(read)
    tiny = db._read_all("select 1")  # owner-loop: allow owner-loop.db-helper -- measured one-row probe
    return rows, tiny
''')
    assert findings == []



def test_scheduling_helper_is_never_handed_to_a_worker(tmp_path: Path) -> None:
    findings = _scan(tmp_path, '''\
import asyncio


async def boot(db):
    await asyncio.to_thread(recover_local_sessions, db, schedule=True)
''')
    assert [(item.line, item.rule) for item in findings] == [(5, "owner-loop.async-worker")]
