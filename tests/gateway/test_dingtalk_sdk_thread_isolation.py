"""DingTalk SDK worker-thread isolation (2026-10-29 exit-75 freeze family).

The SDK's ``start()`` performs a synchronous ``requests`` TLS handshake inside its
coroutine; run on the gateway loop it froze the whole gateway past the loop-liveness
watchdog. These tests pin the two structural guarantees of the thread-isolation fix:

A. a blocking SDK ``start()`` must not stall the adapter loop (the freeze itself);
B. incoming messages processed on the SDK worker loop must be bounced back to the
   adapter loop via ``call_soon_threadsafe`` (they would otherwise die with the
   worker loop on every reconnect);
C. ``_stop_sdk_thread`` reports detach when the worker is stuck in a synchronous
   call, and the queued loop stop still ends the thread once the call returns.
"""

import asyncio
import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest


def _adapter() -> "DingTalkAdapter":
    from gateway.config import Platform
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    adapter = DingTalkAdapter.__new__(DingTalkAdapter)
    adapter.platform = Platform.DINGTALK
    adapter._running = True
    adapter._stream_client = MagicMock()
    adapter._stream_task = None
    adapter._fatal_error_handler = None
    adapter._sdk_thread = None
    adapter._sdk_thread_loop = None
    adapter._sdk_thread_exc = None
    adapter._sdk_thread_wakeup = None
    adapter._fatal_pending_notify = False
    adapter._loop = asyncio.get_running_loop()
    return adapter


@pytest.mark.asyncio
async def test_blocking_sdk_start_does_not_stall_adapter_loop(caplog):
    import logging as _logging

    adapter = _adapter()
    released = threading.Event()

    async def blocking_start() -> None:
        # simulates the SDK's synchronous open_connection(): a plain blocking call
        # inside the coroutine that must run on the worker thread, not this loop
        if released.is_set():
            adapter._running = False  # second round: stop the reconnect loop cleanly
            return
        released.wait(timeout=10.0)
        raise ConnectionError("after block")

    adapter._stream_client.start = blocking_start
    task = asyncio.create_task(adapter._run_stream())
    try:
        t0 = asyncio.get_running_loop().time()
        await asyncio.sleep(0.2)
        elapsed = asyncio.get_running_loop().time() - t0
        assert elapsed < 0.4, elapsed  # 期望: <0.4s — SDK 阻塞期间主 loop 0.2s sleep 不被拉长(旧实现此处冻结)
    finally:
        released.set()
    await asyncio.wait_for(task, timeout=5.0)
    errors = [r for r in caplog.records if r.levelno >= _logging.WARNING and "after block" in r.getMessage()]
    assert len(errors) >= 1, caplog.text  # 期望: >=1 — worker 异常经 _sdk_thread_exc 浮出并由主 loop 记录


@pytest.mark.asyncio
async def test_process_bounces_dispatch_to_adapter_loop(monkeypatch):
    from types import SimpleNamespace

    import plugins.platforms.dingtalk.adapter as adapter_mod
    from plugins.platforms.dingtalk.adapter import _IncomingHandler

    adapter = MagicMock()

    def _consume(coro: Any) -> None:
        coro.close()  # a MagicMock _spawn_bg would leave the coroutine un-awaited

    adapter._spawn_bg.side_effect = _consume
    handler = _IncomingHandler(adapter, asyncio.get_running_loop())
    if getattr(adapter_mod, "ChatbotMessage", None) is None:
        # test env lacks the SDK's ChatbotMessage symbol; stand in a shape-equivalent type
        monkeypatch.setattr(
            adapter_mod, "ChatbotMessage",
            SimpleNamespace(from_dict=lambda data: SimpleNamespace(
                message_id=data.get("messageId"), conversation_id=data.get("conversationId"),
                session_webhook=None, is_in_at_list=False)),
            raising=False)
    payload = {
        "messageId": "m1",
        "conversationId": "c1",
        "sessionWebhook": "https://example.invalid/hook",
        "senderId": "s1",
        "senderStaffId": "staff1",
        "isInAtList": True,
        "text": {"content": "hi"},
    }
    message = MagicMock()
    message.data = json.dumps(payload)

    def run_process_on_worker_loop() -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(handler.process(message))
        finally:
            loop.close()

    worker = threading.Thread(target=run_process_on_worker_loop, name="fake-sdk-worker")
    worker.start()
    try:
        for _ in range(200):  # <=2s: wait for the call_soon_threadsafe handoff
            await asyncio.sleep(0.01)
            if adapter._spawn_bg.call_count >= 1:
                break
        assert adapter._spawn_bg.call_count >= 1, adapter._spawn_bg.call_count  # 期望: >=1 — SDK 线程 loop 上的分派经主 loop _spawn_bg 落地
    finally:
        worker.join(timeout=5.0)
    assert not worker.is_alive()  # 期望: True(非存活) — process 所在一次性 worker loop 正常退出


@pytest.mark.asyncio
async def test_stop_sdk_thread_detaches_then_worker_exits_after_block():
    adapter = _adapter()
    released = threading.Event()

    async def stuck_start() -> None:
        released.wait(timeout=10.0)  # simulates a hung synchronous SDK call

    adapter._stream_client.start = stuck_start
    assert adapter._start_sdk_thread()  # 期望: True — 无前置线程时启动成功
    await asyncio.sleep(0.2)  # let the worker enter the blocking call

    stopped = adapter._stop_sdk_thread(join_timeout=0.3)
    assert stopped is False  # 期望: False — 线程卡同步调用内, 0.3s join 必超时, 须如实报 detach
    assert adapter._sdk_thread is not None and adapter._sdk_thread.is_alive()

    released.set()
    adapter._sdk_thread.join(timeout=5.0)
    assert not adapter._sdk_thread.is_alive()  # 期望: True(非存活) — detach 时排队的 loop.stop 在阻塞返回后生效
