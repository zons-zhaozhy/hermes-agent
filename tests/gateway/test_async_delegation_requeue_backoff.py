"""Async delegation requeue paths must carry the #140899 backoff stamp.

The official #140899 fix paced only the ``delivered is False`` branch of the
drain loop; two sibling requeue paths requeued bare events (run_notifications
``_deliver_async_delegation_group_scoped`` primary-None branch and the drain
loop's ``except Exception`` branch), so a persistently refused/exception-hit
group re-entered claim/admit every 2 s tick — the exact storm shape the retired
local dead-letter (6b9786e161, 23.7 MB of log over 19 h) used to cap. Both
paths now stamp ``_watch_next_retry_at`` via ``_back_off_refused_completions``
so the drain loop's existing hold check (any event with a future retry
timestamp is held) paces them.

This file locks the sibling branch's behaviour contract: events requeued after
a dropped primary carry a future backoff timestamp. The ``except`` branch is
the same one-line call inside an inline drain loop (covered by the official
refusal-backoff regression family), not separately driven here.

The fake runner subclasses the real mixin so every ``self.*`` helper the
delivery path touches is the production implementation; only the adapter-facing
seams (deliver / identity / readiness) are stubbed.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import time
from typing import Any

import pytest
from gateway.run_notifications import GatewayNotificationsMixin


class _FakeRunner(GatewayNotificationsMixin):
    """Real mixin, stubbed only at the adapter-facing seams."""

    def __init__(self, deliver_result: Any) -> None:
        self.session_authority = None
        self._deliver_result = deliver_result

    @asynccontextmanager
    async def _completion_event_scope(self, evt: dict):
        yield

    async def _deliver_completion_notification(
        self, synth_text: str, evt: dict, *, sibling_claims: Any = ()) -> Any:
        return self._deliver_result

    def _completion_identity_seen(self, identity: Any, *, claim: bool = False) -> bool:
        return False

    async def _completion_delivery_ready(self, evt: dict) -> bool:
        return True


def _stub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import gateway.run as gr
    monkeypatch.setattr(gr, '_format_gateway_process_notification', lambda evt: 'notice')
    import tools.async_delegation as ad
    monkeypatch.setattr(ad, 'claim_event_delivery', lambda *a, **k: 'claim-1')


def _evt(session_key: str) -> dict:
    return {'type': 'async_delegation', 'session_key': session_key}


def test_dropped_primary_requeues_siblings_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Siblings requeued after a dropped primary carry a future retry stamp.

    期望: 每个 sibling 的 _watch_next_retry_at > now（base 4s 退避），
    否则 drain 循环的 hold 检查不生效=每 2s tick 重投。
    """
    _stub_environment(monkeypatch)
    runner = _FakeRunner(deliver_result=None)
    primary, sibling = _evt('s-1'), _evt('s-1')
    delivered = asyncio.get_event_loop().run_until_complete(
        runner._deliver_async_delegation_group([primary, sibling]))
    assert delivered is None  # 期望: primary 丢弃形态返回 None
    assert sibling.get('_watch_next_retry_at', 0) > time.monotonic()  # 期望: 退避时间戳在未来
    assert sibling.get('_watch_retry_step') == 1  # 期望: 步数从 1 起


def test_delivered_group_leaves_no_backoff_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """成功投递的事件不带退避时间戳（正常路径零副作用）。"""
    _stub_environment(monkeypatch)
    runner = _FakeRunner(deliver_result=True)
    evt = _evt('s-2')
    delivered = asyncio.get_event_loop().run_until_complete(
        runner._deliver_async_delegation_group([evt]))
    assert delivered is True  # 期望: 投递成功原样返回
    assert '_watch_next_retry_at' not in evt  # 期望: 成功路径不打退避标记
