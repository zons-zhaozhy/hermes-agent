"""Shared clarify registration must not read SQLite on the gateway owner loop."""
from types import SimpleNamespace
import threading

import pytest

from gateway.run_turn_runner import TurnRunner


@pytest.mark.asyncio
async def test_shared_clarify_registration_runs_off_owner_loop(monkeypatch):
    registrations = []

    def register_clarify(session_id, generation, entry, *, timeout_s=None):
        registrations.append((threading.get_ident(), session_id, generation, entry, timeout_s))

    async def send_clarify(**kwargs):
        return SimpleNamespace(success=True, kwargs=kwargs)

    from gateway import local_viewer_events
    monkeypatch.setattr(local_viewer_events, 'prompt_timeout_s', lambda turn, kind: 77)

    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(_status_adapter=SimpleNamespace(send_clarify=send_clarify))
    authority = SimpleNamespace(register_clarify=register_clarify)
    turn._approval_owner = (authority, 'owned-session', 7)
    entry = object()

    result = await turn._send_shared_clarify(entry, question='Which?')

    assert result.success is True
    assert registrations == [
        (registrations[0][0], 'owned-session', 7, entry, 77)
    ]
    assert registrations[0][0] != threading.get_ident(), 'clarify registration ran on the owner loop'
