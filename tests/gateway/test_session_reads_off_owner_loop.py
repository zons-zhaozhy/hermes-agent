"""``session.resume``'s resolution reads and the admission receipt read hop off the owner loop.

Both are blocking SQLite reads (row lookup, compression tip, lineage root, admission row) that used
to run directly on the shared gateway event loop. Live incident behind the fix: the loop missed
three consecutive liveness probes at 20:32 and ``shutdown_watchdog`` exited the gateway (code 75)
while a resume read the 2 GB ``state.db`` on the loop; ``owner_loop_guard`` had flagged those sites
~630 times.

The resume test spies the reads themselves (which thread ran them); the receipt test uses the
product's own ``owner_loop_guard`` in ``raise`` mode.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from owner_loop_guard import register_owner_loop, unregister_owner_loop


async def _mounted_session(tmp_path, monkeypatch):
    """A real store, authority and mounted messaging session (the resume path a bot lane walks)."""
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_contract import SessionRef

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False, adapters={},
                             _cached_agent_for=lambda *a, **k: None, _evict_cached_agent=lambda *a, **k: None,
                             _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='replay', instance_id='owner')
    ref = SessionRef('replay', 'shared')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='replay')
    db.create_session(ref.session_id, source='telegram')
    db.append_message(ref.session_id, 'user', 'hello')
    authority.sessions[ref.session_id] = LiveSession(source, ref.session_id)
    return authority, ref


@pytest.mark.asyncio
async def test_a_resume_reads_its_row_off_the_owner_loop(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    authority, ref = await _mounted_session(tmp_path, monkeypatch)
    loop_thread = threading.get_ident()
    reads: list[int] = []
    real = authority.db.get_session

    def spy(*args, **kwargs):
        reads.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(authority.db, 'get_session', spy)
    connection = AuthorityConnection(authority, object(), {'user_id': 'viewer'})
    await connection.resume(ref, {'session_id': ref.session_id})
    assert reads, 'the resume resolved nothing through the store'
    assert loop_thread not in reads, 'the row read ran on the owner loop'


@pytest.mark.asyncio
async def test_a_receipt_reads_its_admission_off_the_owner_loop(tmp_path, monkeypatch):
    from gateway.session_contract import SessionActor
    from hermes_state_runtime import admit_session_input

    authority, ref = await _mounted_session(tmp_path, monkeypatch)
    admitted = admit_session_input(authority.db, epoch=authority.epoch, principal_id='owner',
                                   session_id=ref.session_id, request_id='r1', payload={'text': 'hi'})
    submitter = SessionActor('owner', 'replay',
                             frozenset({'session:submit', 'session:read', 'session:control'}), 'sock')
    register_owner_loop(mode='raise')
    try:
        receipt = await authority.receipt(submitter, ref, admitted['admission_id'])
    finally:
        unregister_owner_loop()
    assert receipt.admission_id == admitted['admission_id']
