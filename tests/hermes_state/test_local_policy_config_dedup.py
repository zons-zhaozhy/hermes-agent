"""Local launch receipts deduplicate identical frozen configs without changing reads."""
from contextlib import closing
from dataclasses import asdict
import json

from hermes_state import SessionDB
import hermes_state_runtime as rt
from hermes_state_local import commit_local_session


def _receipt(cwd, request_id):
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_local_recovery import local_identity
    from gateway.session_policy import build_policy

    sid = local_identity('profile', 'human', request_id)
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    entry = SessionEntry('local:' + sid, sid, _now(), _now(), origin=source, platform=Platform.LOCAL)
    policy = build_policy(
        {'source': 'cli', 'cwd': str(cwd), 'model': 'm', 'toolsets': []},
        {'platform_toolsets': {'cli': []}, 'auxiliary': {'judge': {'provider': 'fixture'}}},
        private_secrets={},
    )
    return {
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': request_id,
        'session_id': sid, 'route': entry.session_key, 'entry': entry.to_dict(),
        'policy': asdict(policy),
    }


def _create_two(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    first = _receipt(tmp_path, 'one')
    second = _receipt(tmp_path, 'two')
    commit_local_session(db, epoch=epoch, receipt=first)
    commit_local_session(db, epoch=epoch, receipt=second)
    return db, epoch, first, second


def test_identical_config_json_is_deduplicated_and_transparently_restored(tmp_path):
    from hermes_state_local import CONFIG_BLOB_PREFIX, local_receipt

    db, _epoch, first, second = _create_two(tmp_path)
    with closing(db):
        with db._read_ctx() as conn:
            raw_values = [
                json.loads(row[0])['policy']['config_json']
                for row in conn.execute("SELECT value FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'")
            ]
            blobs = conn.execute(
                "SELECT key,value FROM state_meta WHERE key LIKE ?", (CONFIG_BLOB_PREFIX + '%',)
            ).fetchall()

        assert len({value["$hermes_local_config_ref"] for value in raw_values}) == 1
        assert len(blobs) == 1
        assert local_receipt(db, first['session_id'])['policy']['config_json'] == (
            local_receipt(db, second['session_id'])['policy']['config_json']
        )


def test_legacy_receipts_are_compacted_safely(tmp_path):
    from hermes_state_local import CONFIG_BLOB_PREFIX, compact_local_config_receipts, local_receipt

    db, _epoch, first, _second = _create_two(tmp_path)
    with closing(db):
        expected = local_receipt(db, first['session_id'])
        db.set_meta(
            'gateway.local_policy.v1:' + first['session_id'],
            json.dumps(expected, ensure_ascii=False, separators=(',', ':')),
        )

        assert compact_local_config_receipts(db) == 1
        assert local_receipt(db, first['session_id']) == expected
        with db._read_ctx() as conn:
            count = conn.execute(
                "SELECT count(*) FROM state_meta WHERE key LIKE ?", (CONFIG_BLOB_PREFIX + '%',)
            ).fetchone()[0]
        assert count == 1


def test_config_blob_lifecycle_follows_its_last_receipt(tmp_path):
    from hermes_state_local import CONFIG_BLOB_PREFIX, prune_orphan_local_config_blobs

    db, _epoch, first, second = _create_two(tmp_path)
    with closing(db):
        db._execute_write(lambda conn: conn.execute(
            "DELETE FROM state_meta WHERE key=?", ('gateway.local_policy.v1:' + first['session_id'],)
        ))
        assert prune_orphan_local_config_blobs(db) == 0
        db._execute_write(lambda conn: conn.execute(
            "DELETE FROM state_meta WHERE key=?", ('gateway.local_policy.v1:' + second['session_id'],)
        ))
        assert prune_orphan_local_config_blobs(db) == 1


def test_lineage_update_keeps_deduplication_and_transparent_restore(tmp_path):
    from hermes_state_local import CONFIG_BLOB_PREFIX, local_receipt
    from hermes_state_local_lineage import _commit_local_target

    db, _epoch, first, _second = _create_two(tmp_path)
    with closing(db):
        stored = local_receipt(db, first['session_id'])
        changed_entry = dict(stored['entry'], session_id='child')
        receipt = dict(stored, entry=changed_entry, lineage=[stored['session_id'], 'child'])

        def update(conn):
            _commit_local_target(conn, receipt, changed_entry)

        db._execute_write(update)

        assert local_receipt(db, first['session_id']) == receipt
        with db._read_ctx() as conn:
            blobs = conn.execute(
                "SELECT count(*) FROM state_meta WHERE key LIKE ?", (CONFIG_BLOB_PREFIX + '%',)
            ).fetchone()[0]
        assert blobs == 1


def test_auto_maintenance_compacts_receipts_and_prunes_orphan_blobs(tmp_path, monkeypatch):
    from hermes_state_local import local_receipt

    db, _epoch, first, _second = _create_two(tmp_path)
    with closing(db):
        legacy = local_receipt(db, first['session_id'])
        db.set_meta(
            'gateway.local_policy.v1:' + first['session_id'],
            json.dumps(legacy, ensure_ascii=False, separators=(',', ':')),
        )
        monkeypatch.setattr(db, 'prune_sessions', lambda **_kwargs: 0)
        monkeypatch.setattr(db, '_freelist_ratio', lambda: 0.5)
        vacuum_calls = []
        monkeypatch.setattr(db, 'vacuum', lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result['config_receipts_compacted'] == 1
        assert result['config_blobs_pruned'] == 0
        assert result['vacuumed'] is True
        assert vacuum_calls == [True]
        assert local_receipt(db, first['session_id']) == legacy
