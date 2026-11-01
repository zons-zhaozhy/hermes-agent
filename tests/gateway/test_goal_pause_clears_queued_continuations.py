"""Pausing a local goal must retire its durable queued wakes, not leave them ahead of user input."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions

ACTOR = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
REF = SessionRef('owned', 's')
GOAL_REQUEST = '["goal-continuation","agent:main:local:dm:s","wake-1"]'


def _authority(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    return db, authority


def _submit(authority, request_id, text):
    return authority.submit(ACTOR, Submission(request_id=request_id, ref=REF,
                                              payload={'text': text}, intent='queue'))


@pytest.mark.asyncio
async def test_goal_pause_cancel_only_removes_goal_continuations(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        await _submit(authority, 'user-follow-up', 'finish the user request')
        await _submit(authority, GOAL_REQUEST, '[Continuing toward your standing goal]\nGoal: ship')

        removed = await authority.cancel_goal_continuations('s')

    statuses = {row['request_id']: (row['status'], row['outcome'])
                for row in list_session_admissions(db, session_id='s', pending_only=False)}
    assert removed == 1
    assert statuses == {
        'user-follow-up': ('queued', None),
        GOAL_REQUEST: ('terminal', 'cancelled'),
    }
