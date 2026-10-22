"""A Stop after the turn adopted its agent reaches that agent even if the cache no longer holds it."""
import asyncio
import threading

import pytest

from tools import interrupt as tool_interrupt
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent
from tests.gateway.test_session_authority_preadopt_stop import Agent


def test_adopt_with_no_delivered_stop_still_clears_a_lingering_thread_interrupt(tmp_path, monkeypatch):
    """A stop signalled on the execution thread without a delivered_stops entry (thread-level
    signal after the finalizer ran) must not survive into the adopting turn: adopt_agent clears
    the interrupt even when nothing was recorded for this session, or the new turn's first tools
    die with exit 130."""
    _, authority = _authority(tmp_path, monkeypatch)

    class ClearRecordingAgent(Agent):
        def __init__(self) -> None:
            super().__init__()
            self.clear_calls = 0
            self._execution_thread_id = threading.current_thread().ident

        def clear_interrupt(self) -> None:
            self.clear_calls += 1
            self.interrupted = False
            # Mirror the real agent's clear_interrupt: drop this thread's interrupt bit.
            tool_interrupt.set_interrupt(False, self._execution_thread_id)

    agent = ClearRecordingAgent()
    tid = threading.current_thread().ident
    tool_interrupt.set_interrupt(True, tid, reason="explicit stop requested")
    try:
        assert tool_interrupt.is_thread_interrupted(tid)  # 期望: 残留位已置上
        authority.adopt_agent('s', 7, agent)
        assert agent.clear_calls == 1  # 期望: 无交付记录也必须清
        assert not tool_interrupt.is_thread_interrupted(tid)  # 期望: 位已清
    finally:
        tool_interrupt.set_interrupt(False, tid)

    # Same generation WITH a delivered stop must NOT clear: the latched fire must survive.
    agent2 = ClearRecordingAgent()
    authority.delivered_stops['s'] = 7
    authority.adopt_agent('s', 7, agent2)
    assert agent2.clear_calls == 0  # 期望: 本回合交付不清


@pytest.mark.asyncio
async def test_stop_between_adoption_and_promotion_reaches_the_adopted_agent_after_cache_eviction(tmp_path, monkeypatch):
    """adopt_agent ran, the route's running slot is still the claim's pending sentinel, and the
    agent cache entry was evicted (pressure / idle sweep): the Stop must still interrupt the agent
    this generation adopted, not resolve nobody through the cache and latch nothing."""
    from gateway import session_finite

    db, authority = _authority(tmp_path, monkeypatch)
    fresh = Agent()
    cache = {}
    authority.runner._cached_agent_for = cache.get
    adopted, stopped = asyncio.Event(), asyncio.Event()
    ran = {}

    async def execute(authority, ref, row):
        cache['route'] = fresh
        _wire_turn_agent(authority, row['generation'], fresh)
        cache.pop('route')  # evicted before the running slot is promoted
        adopted.set()
        await asyncio.wait_for(stopped.wait(), 5)
        ran[row['request_id']] = fresh.interrupted
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'adopted')
        await asyncio.wait_for(adopted.wait(), 5)
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert ran['adopted'] is True, 'the Stop reached nobody: the adopted agent ran on'
        assert not authority.adopted, 'settlement must drop the adopted agent reference'


@pytest.mark.asyncio
async def test_retire_stop_between_adoption_and_promotion_reaches_the_adopted_agent(tmp_path, monkeypatch):
    """The same window on the retire/shutdown path: stop_authority_turns(in_process=True) used to
    re-resolve the running slot (still the pending sentinel) and latch a generation adopt_agent had
    already consumed, so the adopted agent ran on through the profile stop."""
    from gateway import session_finite
    from gateway.run_runtime import stop_authority_turns

    db, authority = _authority(tmp_path, monkeypatch)
    fresh = Agent()
    authority.runner._cached_agent_for = {}.get
    adopted, stopped = asyncio.Event(), asyncio.Event()
    ran = {}

    async def execute(authority, ref, row):
        _wire_turn_agent(authority, row['generation'], fresh)
        adopted.set()
        await asyncio.wait_for(stopped.wait(), 5)
        ran[row['request_id']] = fresh.interrupted
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'adopted')
        await asyncio.wait_for(adopted.wait(), 5)
        stop_authority_turns(authority, in_process=True)
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert ran['adopted'] is True, 'the profile Stop reached nobody: the adopted agent ran on'
