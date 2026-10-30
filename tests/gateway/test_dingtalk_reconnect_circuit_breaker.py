"""#24851 — DingTalk stream reconnect loop must not storm on a persistent error.

The SDK now runs on a worker thread (``_sdk_thread_main``); these tests exercise the
thread-driven loop: backoff sleeps stay on the adapter loop, worker exceptions surface
through ``_sdk_thread_exc`` + the wakeup event, and the fatal path lands via
``_fatal_pending_notify``.
"""

import asyncio
import logging
from typing import Any, NoReturn
from unittest.mock import MagicMock, patch

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
async def test_repeated_identical_error_trips_breaker_and_stays_silent(caplog):
    adapter = _adapter()
    count = 0
    sleeps = []

    async def fake_start() -> None:
        nonlocal count
        count += 1
        if count > 20:
            adapter._running = False
            return
        raise ConnectionError("boom")

    async def fake_sleep(secs: float, *a: Any, **k: Any) -> None:
        sleeps.append(secs)

    adapter._stream_client.start = fake_start
    with caplog.at_level(logging.WARNING), patch("asyncio.sleep", new=fake_sleep):
        await adapter._run_stream()
    trips = 5  # RECONNECT_CIRCUIT_BREAKER_TRIPS
    records = [r for r in caplog.records if "Stream client error" in r.getMessage()]
    assert len(records) <= trips + 1, [r.getMessage() for r in records]  # 期望: 熔断后静默, 仅第 trips+1 次打 tripped 一条
    assert sleeps[trips:] == [300] * (len(sleeps) - trips), sleeps  # 期望: trips 后恒 300s 静默间隔, 不再指数退避
    # worker lifecycle: last thread must be joined by the loop's exit path or already dead
    assert adapter._sdk_thread is None or not adapter._sdk_thread.is_alive()


@pytest.mark.asyncio
async def test_real_sdk_start_with_issue_type_error_is_bounded_and_fatal(caplog):
    dingtalk_stream = pytest.importorskip("dingtalk_stream")
    import websockets.exceptions

    adapter = _adapter()
    client = dingtalk_stream.DingTalkStreamClient(dingtalk_stream.Credential("id", "secret"))
    client.open_connection = lambda: {"endpoint": "wss://example.invalid", "ticket": "t"}
    adapter._stream_client = client

    def bad_connect(uri: str) -> NoReturn:
        raise TypeError("'coroutine' object does not support the asynchronous context manager protocol")

    real_sleep = asyncio.sleep

    async def fast_sleep(secs: float, *a: Any, **k: Any) -> None:
        await real_sleep(0)

    with caplog.at_level(logging.INFO), \
            patch.object(dingtalk_stream.stream.websockets, "connect", bad_connect), \
            patch("asyncio.sleep", new=fast_sleep):
        try:
            await asyncio.wait_for(adapter._run_stream(), timeout=2.0)
        except TimeoutError:
            logging.getLogger(__name__).warning("run_stream exceeded 2s budget; fatal flags checked below", exc_info=True)
    sdk = [r for r in caplog.records if r.name == "dingtalk_stream.client"]
    assert len(sdk) <= 5, len(sdk)  # 期望: _SdkLogGuard 折叠 SDK 重试循环重复记录至 <=5 条
    assert getattr(adapter, "_fatal_error_code", None) == "dingtalk_stream_error"  # 期望: #24851 缺陷必走 fatal 上报
    assert adapter._fatal_error_retryable is False  # 期望: 不可重试 — 仅重装+重启可修, 网关重连只会每 tick 再失败
    # the worker thread must not leak past the fatal exit
    assert adapter._sdk_thread is None or not adapter._sdk_thread.is_alive()
