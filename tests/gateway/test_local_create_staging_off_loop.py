"""Local session creation leaves the shared gateway loop responsive.

Creation's preflight, staging and post-commit restore all touch SQLite. Running any one of them on
the owner loop lets a slow state.db read freeze every profile and disconnect the client that only
asked for a new chat.
"""

import threading

import pytest


CONFIG = (
    'platform_toolsets: {cli: []}\n'
    'model: {provider: custom, default: frozen, base_url: http://off-loop.invalid/v1}\n'
)


@pytest.mark.asyncio
async def test_local_create_preflight_staging_and_restore_run_off_the_owner_loop(
        tmp_path, monkeypatch):
    from gateway import session_local, session_local_recovery, session_local_title
    from gateway import session_local_migration
    from tests.gateway.test_local_create_env_refs import _owner

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(CONFIG)
    authority, connection = await _owner(home, monkeypatch)
    loop_thread = threading.get_ident()
    call_threads: dict[str, int] = {}

    def wrap(module, name: str) -> None:
        real = getattr(module, name)

        def recorded(*args: object, **kwargs: object) -> object:
            call_threads[name] = threading.get_ident()
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, recorded)

    wrap(session_local_title, 'resolve_titled_session')
    wrap(session_local, 'staged_creation_follows')
    wrap(session_local, 'stage_local_session')
    wrap(session_local_recovery, 'restore_local_session')
    wrap(session_local_recovery, 'transcript_target')
    wrap(session_local_migration, 'resolve_local_target')

    params = {'request_id': 'off-loop', 'source': 'cli', 'cwd': str(home), 'toolsets': [],
              'model': 'frozen', 'title': 'Bot Chat'}
    try:
        result = await connection.create(object(), params)
        assert result['session_id']
        assert set(call_threads) == {
            'resolve_titled_session', 'staged_creation_follows', 'stage_local_session',
            'restore_local_session', 'transcript_target',
        }
        assert loop_thread not in call_threads.values(), (
            f'session.create ran blocking work on the owner loop: {call_threads}'
        )
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_local_history_target_resolution_runs_off_the_owner_loop(tmp_path, monkeypatch):
    from gateway import session_local_recovery
    from gateway.session_local import create_local_session
    from tests.gateway.test_local_create_env_refs import _owner

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(CONFIG)
    authority, connection = await _owner(home, monkeypatch)
    ref = create_local_session(authority, connection.actor, {
        'request_id': 'history-off-loop', 'source': 'cli', 'cwd': str(home), 'toolsets': [],
        'model': 'frozen'})
    loop_thread = threading.get_ident()
    target_threads = []
    real = session_local_recovery.transcript_target

    def transcript_target(*args: object, **kwargs: object) -> object:
        target_threads.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(session_local_recovery, 'transcript_target', transcript_target)
    try:
        target, _rows = await session_local_recovery.prepare_history(authority, ref)
        assert target == ref.session_id
        assert loop_thread not in target_threads, (
            f'transcript target resolution ran on the owner loop: {target_threads}'
        )
    finally:
        await connection.close()
        authority.db.close()
