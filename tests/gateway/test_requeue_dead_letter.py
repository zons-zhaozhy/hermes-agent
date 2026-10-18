"""Bounded requeue for undeliverable queue events — a persistent refusal must dead-letter, not spin.

Regression for the invalid_params watch storm: `_drain_watch_notifications` requeued a rejected
event unconditionally, so one persistently-refused event re-injected every drain tick (2 s) and
saturated the gateway. The watcher path already had `_MAX_COMPLETION_INJECTION_ATTEMPTS`; the
queue paths now share the same bounded-retry contract via `_requeue_with_dead_letter`.
"""
import queue as queue_mod

import pytest

from gateway.run_notifications import _QUEUE_EVENT_REQUEUE_LIMIT, _requeue_with_dead_letter


def test_first_rejections_requeue():
    q = queue_mod.Queue()
    evt = {"type": "process_complete", "origin_session_id": "s-1"}
    for expected_rejects in range(1, _QUEUE_EVENT_REQUEUE_LIMIT):
        _requeue_with_dead_letter(q, evt, context="watch")
        assert q.qsize() == expected_rejects  # 期望: 每次未超限拒绝都重新入队
        assert evt["_wake_rejects"] == expected_rejects  # 期望: 计数逐次保留


def test_persistent_rejection_dead_letters(caplog):
    q = queue_mod.Queue()
    evt = {"type": "process_complete", "origin_session_id": "s-1"}
    for _ in range(_QUEUE_EVENT_REQUEUE_LIMIT - 1):
        _requeue_with_dead_letter(q, evt, context="watch")
    drained = q.qsize()
    with caplog.at_level("ERROR", logger="gateway.run"):
        _requeue_with_dead_letter(q, evt, context="watch")  # 第 5 次：持续拒绝 → 死信
    assert q.qsize() == drained  # 期望: 死信后不再入队
    assert any("Dead-lettering watch event" in r.message for r in caplog.records)  # 期望: error 留痕
    assert evt["_wake_rejects"] == _QUEUE_EVENT_REQUEUE_LIMIT


def test_dead_letter_is_final_even_if_called_again(caplog):
    q = queue_mod.Queue()
    evt = {"type": "async_delegation"}
    for _ in range(_QUEUE_EVENT_REQUEUE_LIMIT + 3):
        _requeue_with_dead_letter(q, evt, context="async delegation group")
    assert q.qsize() == _QUEUE_EVENT_REQUEUE_LIMIT - 1  # 期望: 只有前 4 次入队，之后全部死信
    assert evt.get("session_id") is None or isinstance(evt.get("session_id"), str)  # evt 字段不被破坏


@pytest.mark.parametrize("context", ["watch", "unadmitted sibling", "async delegation group error"])
def test_context_reaches_log(caplog, context):
    evt = {"type": "heartbeat"}
    with caplog.at_level("ERROR", logger="gateway.run"):
        for _ in range(_QUEUE_EVENT_REQUEUE_LIMIT):
            _requeue_with_dead_letter(queue_mod.Queue(), evt, context=context)
    assert any(f"Dead-lettering {context}" in r.message for r in caplog.records)  # 期望: 四条调用路径语境进日志
