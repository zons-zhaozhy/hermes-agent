"""Synchronous admission reads must stay off the gateway owner event loop."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.asyncio
async def test_claim_preflight_reads_admissions_off_loop(tmp_path, monkeypatch):
    from gateway import session_authority

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    loop_thread = threading.get_ident()
    real = session_authority.list_session_admissions
    threads: list[int] = []

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    with db:
        await _submit(authority, 'work')
        monkeypatch.setattr(session_authority, 'list_session_admissions', recorded)
        row, first = await authority._claim_next(REF, authority.sessions['s'])

    assert row['request_id'] == first['request_id'] == 'work'
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_startup_local_receipt_scan_runs_off_loop(tmp_path, monkeypatch):
    from gateway import run_runtime, session_authorities, session_hosted_service
    from gateway.platforms import webhook_ingress
    from gateway.session_local_recovery import local_receipt_ids

    async def noop(*args, **kwargs):
        return None

    async def no_native_rows(*args, **kwargs):
        return {}

    class _EmptyCursor:
        def __iter__(self):
            return iter(())

        def fetchall(self):
            return []

    class _EmptyConnection:
        def execute(self, *args, **kwargs):
            return _EmptyCursor()

    class _EmptyDB:
        def _read_ctx(self):
            return nullcontext(_EmptyConnection())

    monkeypatch.setattr(session_authorities, 'serving_authorities', lambda runner: [
        SimpleNamespace(
            profile_id=tmp_path,
            runner=SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False)),
            db=_EmptyDB(),
        )
    ])
    monkeypatch.setattr(session_hosted_service, 'ensure_hosted_service', noop)
    monkeypatch.setattr(webhook_ingress, 'recover_webhook_finalizations', noop)
    monkeypatch.setattr(run_runtime, 'recover_profile_native_sessions', no_native_rows)
    monkeypatch.setattr(
        'gateway.session_viewer_close.arm_startup_orphan_closes',
        noop,
    )
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def recorded(authority):
        threads.append(threading.get_ident())
        return local_receipt_ids(authority)

    monkeypatch.setattr(
        'gateway.session_local_recovery.local_receipt_ids', recorded, raising=False
    )

    await run_runtime.recover_gateway_native_sessions(SimpleNamespace())

    assert len(threads) == 1
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_startup_detached_recovery_reads_admissions_off_loop(tmp_path, monkeypatch):
    from gateway import run_runtime
    import hermes_state_runtime as state_runtime

    _, authority = _authority(tmp_path, monkeypatch)
    authority.runner.session_store = SimpleNamespace(list_sessions=list)

    async def no_native_rows(bindings):
        return {}

    monkeypatch.setattr(authority, 'recover_native_sessions', no_native_rows)
    monkeypatch.setattr(
        authority.db, '_read_all',
        lambda sql, params=(): [{'target_session_id': 'missing'}],
    )
    real = state_runtime.list_session_admissions
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(state_runtime, 'list_session_admissions', recorded)
    assert state_runtime.list_session_admissions is recorded

    outcome = await run_runtime.recover_profile_native_sessions(authority.runner, authority)

    assert outcome == {}
    assert len(threads) == 1
    assert threads and loop_thread not in threads
