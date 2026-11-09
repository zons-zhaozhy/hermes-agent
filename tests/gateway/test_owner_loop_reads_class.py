"""Synchronous admission reads must stay off the gateway owner event loop."""

from __future__ import annotations

import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.asyncio
async def test_terminal_receipt_result_read_off_loop(tmp_path, monkeypatch):
    from gateway import session_controls, session_results
    from gateway.session_contract import AdmissionReceipt

    _db, authority = _authority(tmp_path, monkeypatch)
    connection = session_controls.AuthorityConnection(
        authority, SimpleNamespace(write=lambda frame: None),
        {'user_id': 'human', 'profile_id': authority.profile_id,
         'instance_id': authority.instance_id})
    receipt = AdmissionReceipt(
        'admission', REF, 1, 'terminal', 'completed', authority.epoch, 1)

    async def terminal_receipt(*args, **kwargs):
        return receipt

    saved = {'result': {'final_response': 'done', 'completed': True},
             'usage': {'input_tokens': 3}}
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())
        return saved

    monkeypatch.setattr(authority, 'receipt', terminal_receipt)
    monkeypatch.setattr(session_results, 'admission_result', recorded)
    out = await connection.receipt(
        REF, {'admission_id': 'admission', 'include_result': True})

    assert out['result'] == saved['result']
    assert out['usage'] == saved['usage']
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_busy_redirect_approval_generation_read_off_loop(tmp_path, monkeypatch):
    from gateway import session_busy_controls
    from hermes_state_runtime import admit_session_input, claim_session_input
    from tests.gateway.test_session_authority_cancel_settlement import ACTOR

    db, authority = _authority(tmp_path, monkeypatch)
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = authority.check_approval_generation

    def recorded(session_id, generation):
        threads.append(threading.get_ident())
        return real(session_id, generation)

    with db:
        admit_session_input(
            db, epoch=authority.epoch, principal_id='human', session_id='s',
            request_id='redirect', payload={'text': 'new direction'})
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        monkeypatch.setattr(authority, 'check_approval_generation', recorded)
        monkeypatch.setattr(
            authority, 'steer_agent', lambda ref, generation, text: (None, False))
        connection = SimpleNamespace(authority=authority, actor=ACTOR)
        outcome = await session_busy_controls.correct(
            connection, REF,
            {'session_id': 's', 'text': 'new direction',
             'execution_generation': row['generation']},
            verb='redirect')

    assert outcome['status'] == 'rejected'
    assert outcome['delivered'] is False
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_profile_reconcile_roster_scans_off_loop(tmp_path, monkeypatch):
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin

    runner = GatewayProfileReconcileMixin()
    runner._running = True
    runner._multiplex_on = lambda: True
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._primary_profile_name = 'default'
    launch, profile = tmp_path / 'launch', tmp_path / 'profile'
    runner._served_profile_homes = {'default': launch, 'profile': profile}
    runner._served_profile_signatures = {'default': 'same', 'profile': 'old'}
    runner.session_runtime_descriptor = {}
    runner._drop_stale_platform_verdicts = lambda name: None
    runner._live_resource_claims = lambda active: {}
    runner._record_served_profiles = lambda active, homes: None
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def recorded(**kwargs):
        threads.append(threading.get_ident())
        assert kwargs == {'multiplex': True}
        return [('default', launch), ('profile', profile)]

    async def signatures(homes):
        assert homes == {'profile': profile}
        return {'profile': 'new'}

    async def start_adapters(name, home, claimed):
        return 0

    runner._start_one_profile_adapters = start_adapters

    monkeypatch.setattr('hermes_cli.profiles.profiles_to_serve', recorded)
    monkeypatch.setattr(runner, '_serve_signatures', signatures)
    result = await runner.reconcile_served_profiles(reason='watcher')

    assert result['rescanned'] == ['profile']
    assert result['served_profiles'] == ['default', 'profile']
    assert len(threads) == 2
    assert threads and all(thread != loop_thread for thread in threads)


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
async def test_pause_on_unknown_claim_reads_admissions_off_loop(tmp_path, monkeypatch):
    from gateway import session_authority, session_unattended_unknown
    from hermes_state_runtime import RuntimeStoreError

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = session_authority.list_session_admissions

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    async def unknown_claim(*args, **kwargs):
        raise RuntimeStoreError('unknown_execution')

    async def attended_unknown(*args, **kwargs):
        return False

    with db:
        receipt = await _submit(authority, 'lost')
        waiter = authority.waiters.setdefault(
            receipt.admission_id, asyncio.get_running_loop().create_future())
        authority.native_waiters.add(receipt.admission_id)
        monkeypatch.setattr(authority, '_claim_next', unknown_claim)
        monkeypatch.setattr(
            session_unattended_unknown, 'settle_unattended_unknown', attended_unknown)
        monkeypatch.setattr(session_authority, 'list_session_admissions', recorded)

        assert await authority._claim_with_retry(REF, authority.sessions['s']) is None

    assert threads and loop_thread not in threads
    assert receipt.admission_id not in authority.native_waiters
    assert receipt.admission_id not in authority.waiters
    assert waiter.done()
    with pytest.raises(RuntimeStoreError, match='unknown_execution'):
        waiter.result()


@pytest.mark.asyncio
async def test_admitted_pending_publication_reads_off_loop(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real_handle = authority._handle

    def recorded_handle(ref):
        threads.append(threading.get_ident())
        return real_handle(ref)

    monkeypatch.setattr(authority, '_handle', recorded_handle)
    with db:
        receipt = await _submit(authority, 'queued')

    assert receipt.status == 'queued'
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_committed_settlement_recovery_reads_result_off_loop(tmp_path, monkeypatch):
    from gateway import session_results, session_settlement_recovery

    db, authority = _authority(tmp_path, monkeypatch)
    loop_thread = threading.get_ident()
    threads = []
    real_admission = session_settlement_recovery.get_session_admission
    real_result = session_results.admission_result

    def recorded_admission(*args, **kwargs):
        threads.append(threading.get_ident())
        return real_admission(*args, **kwargs)

    def recorded_result(*args, **kwargs):
        threads.append(threading.get_ident())
        return real_result(*args, **kwargs)

    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {
            'result': {'final_response': 'done', 'completed': True}, 'usage': {}}
        return 'done'

    original_settle = session_settlement_recovery.settle_with_retry

    async def commit_then_lose_receipt(*args, **kwargs):
        await original_settle(*args, **kwargs)
        settlement = kwargs.get('settlement', args[7])
        settlement.clear()  # the owner forgot the commit receipt, as a lost response can
        raise OSError('commit response lost')

    monkeypatch.setattr(
        'gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(
        session_settlement_recovery, 'settle_with_retry', commit_then_lose_receipt)
    monkeypatch.setattr(session_results, 'admission_result', recorded_result)
    monkeypatch.setattr(
        session_settlement_recovery, 'get_session_admission', recorded_admission)
    with db:
        receipt = await _submit(authority, 'committed')
        await asyncio.wait_for(authority.sessions['s'].task, 5)

    row = real_admission(db, admission_id=receipt.admission_id)
    assert row['status'] == 'terminal'
    assert len(threads) == 2
    assert threads and all(thread != loop_thread for thread in threads)


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


@pytest.mark.asyncio
async def test_startup_native_recovery_reads_admissions_off_loop(tmp_path, monkeypatch):
    from gateway import session_authority

    _db, authority = _authority(tmp_path, monkeypatch)
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = session_authority.list_session_admissions

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(session_authority, 'list_session_admissions', recorded)
    outcome = await authority.recover_native_sessions([('s', SimpleNamespace(), None)])

    assert outcome == {'s': 'not_found'}
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_startup_native_recovery_reads_compression_tip_off_loop(tmp_path, monkeypatch):
    from gateway import session_envelope
    from hermes_state_runtime import admit_session_input

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    admit_session_input(
        db, epoch=authority.epoch, principal_id='human', session_id='s', request_id='native',
        payload={'text': 'native', 'native_text_v1': {'source': {}, 'route': 'route'}})
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = authority.physical_target

    def recorded(ref):
        threads.append(threading.get_ident())
        return real(ref)

    async def checked_route(*args, **kwargs):
        return SimpleNamespace(), 'route'

    monkeypatch.setattr(authority, 'physical_target', recorded)
    monkeypatch.setattr(session_envelope, 'check_native_route', checked_route)
    outcome = await authority.recover_native_sessions([('s', SimpleNamespace(), object())])

    assert outcome == {'s': 'ready'}
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_native_route_recovery_lookup_runs_off_loop(monkeypatch):
    from datetime import UTC, datetime
    from types import SimpleNamespace as NS

    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_envelope import check_native_route

    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat', user_id='human')
    adapter = object()
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def lookup(route):
        threads.append(threading.get_ident())
        return NS(session_id='s')

    store = NS(
        _generate_session_key=lambda candidate: 'chat-key',
        lookup_by_session_key=lookup,
    )
    runner = NS(
        config=NS(multiplex_profiles=False),
        session_store=store,
        _adapter_flag=lambda *args, **kwargs: False,
        _adapter_profile_for_source=lambda candidate: None,
        _is_user_authorized_for_source=lambda *args, **kwargs: True,
        _adapter_for_source=lambda candidate: adapter,
    )
    encoded_source = source.to_dict()
    encoded_source['is_bot'] = source.is_bot
    payload = {'text': 'hello', 'native_text_v1': {
        'source': encoded_source, 'route': 'chat-key',
        'timestamp': datetime.now(UTC).isoformat(),
        'event': {'message_id': 'm1'},
    }}

    recovered_source, route = await check_native_route(runner, payload, 's', source, adapter)

    assert recovered_source.to_dict() == source.to_dict()
    assert route == 'chat-key'
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_automation_route_preflight_runs_off_loop(monkeypatch):
    from gateway import session_automation
    from gateway.session_envelope import check_native_route

    loop_thread = threading.get_ident()
    threads: list[int] = []

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())
        return 'source', 'route'

    monkeypatch.setattr(session_automation, 'check_automation_route', recorded)
    outcome = await check_native_route(
        SimpleNamespace(), {'native_text_v1': {'automation': {}}}, 's', SimpleNamespace(), object())

    assert outcome == ('source', 'route')
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_startup_local_automation_preflight_runs_off_loop(tmp_path, monkeypatch):
    from gateway import session_automation, session_envelope
    from hermes_state_runtime import admit_session_input

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    admit_session_input(
        db, epoch=authority.epoch, principal_id='automation:route', session_id='s', request_id='native',
        payload={'text': 'native', 'native_text_v1': {'source': {}, 'route': 'route'}})
    admit_session_input(
        db, epoch=authority.epoch, principal_id='automation:route', session_id='s', request_id='automation',
        payload={'text': 'automation', 'local_automation_v1': {}})
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())

    async def checked_route(*args, **kwargs):
        return SimpleNamespace(), 'route'

    monkeypatch.setattr(session_automation, 'check_local_automation', recorded)
    monkeypatch.setattr(session_envelope, 'check_native_route', checked_route)
    monkeypatch.setattr(authority, 'physical_target', lambda ref: ref.session_id)
    outcome = await authority.recover_native_sessions([('s', SimpleNamespace(), object())])

    assert outcome == {'s': 'ready'}
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_startup_detached_lineage_read_runs_off_loop(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    import hermes_state_runtime as state_runtime
    from gateway import session_detached, session_route_recovery

    _, authority = _authority(tmp_path, monkeypatch)
    source = NS(platform='test', user_id='human')
    authority.runner._restored_source = lambda entry: source
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def recorded_names(*args, **kwargs):
        threads.append(threading.get_ident())
        return False

    monkeypatch.setattr(
        session_route_recovery, '_mount_plan',
        lambda authority, sid: ('agent:main:chat', NS(origin=source), None),
    )
    monkeypatch.setattr(
        session_route_recovery, '_names_another_conversation', recorded_names)
    monkeypatch.setattr(
        state_runtime, 'list_session_admissions',
        lambda *args, **kwargs: [{'status': 'queued'}],
    )

    scheduled = await session_detached.recover_detached_sessions(authority, {'missing'})

    assert scheduled == []
    assert len(threads) == 1
    assert threads and loop_thread not in threads
