"""Weixin pending-delivery redelivery: texts parked while iLink refuses bot-initiated
sends must survive on disk, stay bounded, and flush exactly once on session recovery."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.weixin import adapter as weixin
from plugins.platforms.weixin.adapter import PendingDeliveryStore, WeixinAdapter


def _make_adapter(home: Path) -> WeixinAdapter:
    with patch.object(weixin, "get_hermes_home", return_value=str(home)):
        adapter = WeixinAdapter(
            PlatformConfig(enabled=True, token="test-token", extra={"account_id": "test-account"})
        )
    adapter._poll_session = Mock()
    adapter._send_session = Mock()
    return adapter


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path


class TestPendingDeliveryStore:
    @pytest.mark.asyncio
    async def test_enqueue_then_take_due_returns_oldest_first(self, home: Path):
        store = PendingDeliveryStore(str(home))
        await store.enqueue("acct", "peer", "first")
        await store.enqueue("acct", "peer", "second")
        first_take = await store.take_due("acct", "peer")
        assert first_take == [("first", 0), ("second", 0)]  # 期望: 按入队序返回且移除
        second_take = await store.take_due("acct", "peer")
        assert second_take == []  # 期望: 取走后队列清空

    @pytest.mark.asyncio
    async def test_ttl_expiry_drops_aged_entries(self, home: Path):
        store = PendingDeliveryStore(str(home), ttl_seconds=10.0)
        await store.enqueue("acct", "peer", "stale")
        key = store._key("acct", "peer")
        store._cache[key][0]["queued_at"] = time.time() - 11.0
        taken = await store.take_due("acct", "peer")
        assert taken == []  # 期望: 超过 ttl_seconds 的条目被丢弃

    @pytest.mark.asyncio
    async def test_per_peer_cap_drops_oldest(self, home: Path):
        store = PendingDeliveryStore(str(home), max_per_peer=2)
        for text in ("a", "b", "c"):
            await store.enqueue("acct", "peer", text)
        taken = await store.take_due("acct", "peer")
        assert taken == [("b", 0), ("c", 0)]  # 期望: 超限丢最旧，保留最近 max_per_peer 条

    @pytest.mark.asyncio
    async def test_requeue_attempts_increment_then_drop(self, home: Path):
        store = PendingDeliveryStore(str(home), max_attempts=2)
        await store.enqueue("acct", "peer", "text")
        taken = await store.take_due("acct", "peer")
        await store.requeue_failed("acct", "peer", taken)
        taken2 = await store.take_due("acct", "peer")
        assert taken2 == [("text", 1)]  # 期望: 失败重排队列 attempts 递增为 1
        await store.requeue_failed("acct", "peer", taken2)
        taken3 = await store.take_due("acct", "peer")
        assert taken3 == []  # 期望: attempts 达 max_attempts 后被丢弃

    @pytest.mark.asyncio
    async def test_persist_and_restore_roundtrip(self, home: Path):
        store = PendingDeliveryStore(str(home))
        await store.enqueue("acct", "peer", "survives restart")
        restored = PendingDeliveryStore(str(home))
        restored.restore("acct")
        taken = await restored.take_due("acct", "peer")
        assert taken == [("survives restart", 0)]  # 期望: 磁盘持久化后新实例可恢复

    @pytest.mark.asyncio
    async def test_take_due_removes_persisted_file_when_empty(self, home: Path):
        store = PendingDeliveryStore(str(home))
        await store.enqueue("acct", "peer", "gone")
        await store.take_due("acct", "peer")
        path = home / "platforms" / "weixin" / "acct.pending-deliveries.json"
        assert not path.exists()  # 期望: 队列清空后磁盘文件移除


class TestSendFailureParking:
    @pytest.mark.asyncio
    async def test_session_not_ready_failure_parks_text(self, home: Path):
        adapter = _make_adapter(home)
        stale_resp = AsyncMock(return_value={"ret": -2, "errmsg": "prepare failed"})
        with patch.object(weixin, "_send_message", new=stale_resp):
            result = await adapter.send("peer", "reply that iLink refused")
        assert result.success is False  # 期望: 协议性失声下 send 仍报失败
        parked = await adapter._pending_store.take_due("test-account", "peer")
        assert parked == [("reply that iLink refused", 0)]  # 期望: session-not-ready 失败的文本被暂存待补投


class TestFlushPending:
    @pytest.mark.asyncio
    async def test_flush_delivers_all_parked_texts_in_order(self, home: Path):
        adapter = _make_adapter(home)
        await adapter._pending_store.enqueue("test-account", "peer", "first")
        await adapter._pending_store.enqueue("test-account", "peer", "second")
        ok = AsyncMock(return_value=SendResult(success=True))
        with patch.object(WeixinAdapter, "send", ok):
            await adapter._flush_pending("peer")
        sent = [c.args[1] for c in ok.await_args_list]
        assert sent == ["first", "second"]  # 期望: 补投按序送达全部暂存文本
        leftover = await adapter._pending_store.take_due("test-account", "peer")
        assert leftover == []  # 期望: 补投后队列清空

    @pytest.mark.asyncio
    async def test_flush_failure_requeues_with_attempts(self, home: Path):
        adapter = _make_adapter(home)
        await adapter._pending_store.enqueue("test-account", "peer", "still failing")
        fail = AsyncMock(return_value=SendResult(success=False, error="boom"))
        with patch.object(WeixinAdapter, "send", fail):
            await adapter._flush_pending("peer")
        requeued = await adapter._pending_store.take_due("test-account", "peer")
        assert requeued == [("still failing", 1)]  # 期望: 补投失败重排 attempts=1

    @pytest.mark.asyncio
    async def test_concurrent_flush_is_single_flight(self, home: Path):
        adapter = _make_adapter(home)
        await adapter._pending_store.enqueue("test-account", "peer", "once")
        send_calls = []

        async def slow_send(*args: object) -> SendResult:
            send_calls.append(args[-1])  # bound call shape: (instance, user_id, text) — text is last
            await asyncio.sleep(0.05)
            return SendResult(success=True)

        with patch.object(WeixinAdapter, "send", slow_send):
            await asyncio.gather(adapter._flush_pending("peer"), adapter._flush_pending("peer"))
        assert send_calls == ["once"]  # 期望: 同 peer 并发 flush 只投递一次


class TestInboundTriggersFlush:
    @pytest.mark.asyncio
    async def test_fresh_context_token_schedules_flush(self, home: Path):
        adapter = _make_adapter(home)
        adapter._is_dm_intake_allowed = lambda sender_id: True
        flush = AsyncMock()
        with patch.object(adapter, "_flush_pending", flush), patch.object(
            adapter, "_fetch_typing_ticket", AsyncMock()
        ):
            await adapter._process_message(
                {
                    "from_user_id": "peer",
                    "message_id": "m1",
                    "context_token": "fresh-token",
                    "item_list": [{"content": '{"text":"hi"}'}],
                }
            )
            await asyncio.sleep(0.01)
        assert flush.await_args_list[0].args == ("peer",)  # 期望: 带新 context_token 的入站消息触发该 peer 的补投

    @pytest.mark.asyncio
    async def test_message_without_context_token_does_not_flush(self, home: Path):
        adapter = _make_adapter(home)
        adapter._is_dm_intake_allowed = lambda sender_id: True
        flush = AsyncMock()
        with patch.object(adapter, "_flush_pending", flush), patch.object(
            adapter, "_fetch_typing_ticket", AsyncMock()
        ):
            await adapter._process_message(
                {
                    "from_user_id": "peer",
                    "message_id": "m2",
                    "context_token": "",
                    "item_list": [{"content": '{"text":"hi"}'}],
                }
            )
            await asyncio.sleep(0.01)
        assert flush.await_count == 0  # 期望: 无 context_token 的消息不触发补投
