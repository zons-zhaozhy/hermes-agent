"""Reclaim only unreachable terminal admission results."""

from __future__ import annotations

import json

from hermes_state import SessionDB
from hermes_state_terminal import ADMISSION_PREFIX, RESULT_PREFIX


def _put_result(db: SessionDB, admission_id: str, value: dict) -> None:
    db.set_meta(RESULT_PREFIX + admission_id, json.dumps(value))


def test_prunes_only_results_with_no_live_admission_or_tombstone(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("live", source="cli", model="test")
        db._execute_write(lambda conn: conn.execute(
            """INSERT INTO session_admissions(
                   admission_id, request_id, principal_id, target_session_id, lineage_json,
                   payload_json, payload_digest, intent, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "live-admission", "request-live", "tester", "live", json.dumps([]),
                json.dumps({"text": "hello"}), "digest", "queue", "terminal",
            ),
        ))
        db.set_meta(
            ADMISSION_PREFIX + "tombstone-admission",
            json.dumps({"admission_id": "tombstone-admission"}),
        )
        _put_result(db, "orphan", {"result": {"final_response": "gone"}})
        _put_result(db, "live-admission", {"result": {"final_response": "kept"}})
        _put_result(db, "tombstone-admission", {"result": {"final_response": "kept"}})

        assert db.prune_orphan_admission_results() == 1

        with db._read_ctx() as conn:
            keys = {
                row[0] for row in conn.execute(
                    "SELECT key FROM state_meta WHERE key LIKE ?", (RESULT_PREFIX + "%",)
                )
            }
        assert keys == {
            RESULT_PREFIX + "live-admission",
            RESULT_PREFIX + "tombstone-admission",
        }
