"""Gateway runtime guard detects SessionDB primitives on the registered owner loop."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from owner_loop_guard import OwnerLoopIOError, register_owner_loop, unregister_owner_loop
from hermes_state import SessionDB


@pytest.fixture
def guard_mode():
    def _set(mode):
        register_owner_loop(mode=mode)
        yield
        unregister_owner_loop()
    return _set


@pytest.mark.asyncio
async def test_warn_mode_reports_loop_thread_db_access(tmp_path: Path, caplog):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    register_owner_loop(mode="warn")
    try:
        with caplog.at_level(logging.WARNING, logger="owner_loop_guard"):
            db._read_all("SELECT 1")
    finally:
        unregister_owner_loop()
    assert any("ran directly on the gateway event loop" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_raise_mode_fails_fast(tmp_path: Path):
    db = SessionDB(tmp_path / "state.db")
    register_owner_loop(mode="raise")
    try:
        with pytest.raises(OwnerLoopIOError, match="SessionDB._read_ctx"):
            db._read_all("SELECT 1")
    finally:
        unregister_owner_loop()


@pytest.mark.asyncio
async def test_off_mode_and_worker_threads_are_allowed(tmp_path: Path):
    db = SessionDB(tmp_path / "state.db")
    register_owner_loop(mode="off")
    assert db._read_all("SELECT 1")
    unregister_owner_loop()

    import asyncio

    register_owner_loop(mode="raise")
    try:
        rows = await asyncio.to_thread(db._read_all, "SELECT 1")
        assert rows
    finally:
        unregister_owner_loop()
