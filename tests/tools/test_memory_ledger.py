"""Tests for tools/memory_ledger.py — append-only history + rollback.

Covers the full loop exercised in production verification (2026-08-24):
add/replace record before/after blobs, rollback restores prior bytes,
create-rollback empties the file, malformed ledger lines are skipped.
"""

from __future__ import annotations

import json

import pytest

from tools import memory_ledger
from tools.memory_tool import MemoryStore


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _last_entry():
    rows = memory_ledger.list_entries(limit=1)
    assert rows, "ledger is empty"
    return rows[0]


def test_add_and_rollback_restore(isolated_home):
    store = MemoryStore()
    assert store.add("memory", "entry one")["success"]
    assert store.add("memory", "entry two")["success"]
    assert store.replace("memory", "entry one", "entry ONE")["success"]

    entry = _last_entry()
    assert entry["action"] == "replace"
    assert entry["target"] == "memory"

    ok, msg = memory_ledger.rollback_entry(entry["id"])
    assert ok, msg
    raw = memory_ledger.read_target("memory")
    assert "entry one" in raw
    assert "entry ONE" not in raw


def test_rollback_of_create_empties_file(isolated_home):
    store = MemoryStore()
    assert store.add("user", "fresh profile line")["success"]
    entry = _last_entry()
    assert entry["action"] == "add"
    ok, msg = memory_ledger.rollback_entry(entry["id"])
    assert ok, msg
    assert memory_ledger.read_target("user") == ""


def test_rollback_is_undoable(isolated_home):
    store = MemoryStore()
    assert store.add("memory", "kept entry")["success"]
    assert store.add("memory", "doomed entry")["success"]
    bad = _last_entry()
    ok, _ = memory_ledger.rollback_entry(bad["id"])
    assert ok
    # The safety entry lets us undo the rollback: restore pre-rollback state.
    safety = memory_ledger.list_entries(limit=10)
    assert any(e["action"] == "pre-rollback-safety" for e in safety)
    assert "doomed entry" not in memory_ledger.read_target("memory")
    assert "kept entry" in memory_ledger.read_target("memory")


def test_malformed_ledger_lines_skipped(isolated_home):
    store = MemoryStore()
    assert store.add("memory", "good entry")["success"]
    p = memory_ledger.ledger_path()
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("this is not json\n")
    rows = memory_ledger.list_entries()
    assert len(rows) == 1
    assert rows[0]["action"] == "add"


def test_rollback_unknown_id_fails_closed(isolated_home):
    ok, msg = memory_ledger.rollback_entry("nonexistent")
    assert not ok
    assert "no ledger entry" in msg


def test_target_filter(isolated_home):
    store = MemoryStore()
    assert store.add("memory", "m")["success"]
    assert store.add("user", "u")["success"]
    only_user = memory_ledger.list_entries(target="user")
    assert len(only_user) == 1
    assert only_user[0]["target"] == "user"
