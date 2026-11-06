"""The DingTalk stream socket keeps a stall-tolerant protocol ping window.

``dingtalk_stream.stream.start`` opens ``websockets.connect(uri)`` with the library defaults (20 s
ping, 20 s pong timeout) while the SDK also pings at the application level every 60 s. A late pong
then reads as a dead socket (``sent 1011 (internal error): keepalive ping timeout``) and costs a
full reconnect — three times a day on the reference install. The adapter's shim widens the protocol
window inside the SDK's own namespace only.
"""
from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _FakeWebsockets(ModuleType):
    exceptions = 'sdk-websockets-exceptions'

    def __init__(self):
        super().__init__('websockets')
        self.calls: list[dict] = []

    def connect(self, uri, **kwargs):  # the SDK calls this as an async context manager
        self.calls.append({'uri': uri, **kwargs})
        return object()


@pytest.fixture
def sdk_stream(monkeypatch):
    """A stand-in ``dingtalk_stream`` package whose ``stream.websockets.connect`` records kwargs."""
    from plugins.platforms.dingtalk import adapter

    fake_ws = _FakeWebsockets()
    stream = ModuleType('dingtalk_stream.stream')
    stream.websockets = fake_ws
    stream.requests = SimpleNamespace(post=lambda *a, **k: None)
    package = ModuleType('dingtalk_stream')
    package.stream = stream
    monkeypatch.setitem(sys.modules, 'dingtalk_stream', package)
    monkeypatch.setitem(sys.modules, 'dingtalk_stream.stream', stream)
    for flag in ('_sdk_ws_shim_installed', '_sdk_http_shim_installed'):
        monkeypatch.setattr(adapter, flag, False)
    return adapter, package, stream, fake_ws


@pytest.mark.asyncio
async def test_connecting_installs_the_ping_window(sdk_stream, monkeypatch):
    """``connect`` wires the shim before the SDK worker opens its socket (a dead installer is no fix)."""
    import asyncio

    adapter, package, stream, fake_ws = sdk_stream
    parked = asyncio.Event()

    class _StreamClient:
        def __init__(self, credential):
            self.credential = credential
            self.logger = logging.getLogger('dingtalk_stream.client')

        async def start(self):
            await parked.wait()

        def register_callback_handler(self, topic, handler):
            return None

    package.Credential = lambda cid, secret: (cid, secret)
    package.DingTalkStreamClient = _StreamClient
    package.ChatbotMessage = SimpleNamespace(TOPIC='chatbot')
    monkeypatch.setattr(adapter, 'dingtalk_stream', package)  # bound at import: None without the SDK
    monkeypatch.setattr(adapter, 'DINGTALK_STREAM_AVAILABLE', True)
    monkeypatch.setattr(adapter, 'HTTPX_AVAILABLE', True)
    monkeypatch.setattr(adapter, 'CARD_SDK_AVAILABLE', False)
    monkeypatch.setenv('DINGTALK_CLIENT_ID', 'id')
    monkeypatch.setenv('DINGTALK_CLIENT_SECRET', 'secret')
    client = adapter.DingTalkAdapter(adapter.PlatformConfig(enabled=True))
    try:
        assert await client.connect() is True
        assert isinstance(stream.websockets, ModuleType), 'the SDK namespace still hands out the raw module'
        stream.websockets.connect('wss://example.invalid/stream')
        assert fake_ws.calls[-1]['ping_timeout'] == adapter._SDK_WS_PING_TIMEOUT_S
    finally:
        await client.disconnect()


def test_the_sdk_stream_socket_gets_a_wider_ping_window(sdk_stream):
    adapter, _package, stream, fake_ws = sdk_stream
    adapter._install_sdk_ws_ping_shim()
    stream.websockets.connect('wss://example.invalid/stream', open_timeout=10)
    assert fake_ws.calls == [{'uri': 'wss://example.invalid/stream', 'open_timeout': 10,
                              'ping_interval': adapter._SDK_WS_PING_INTERVAL_S,
                              'ping_timeout': adapter._SDK_WS_PING_TIMEOUT_S}]
    assert adapter._SDK_WS_PING_TIMEOUT_S > adapter._SDK_WS_PING_INTERVAL_S > 20


def test_the_shim_is_scoped_to_the_sdk_and_idempotent(sdk_stream):
    adapter, _package, stream, _fake_ws = sdk_stream
    global_ws = sys.modules.get('websockets')
    adapter._install_sdk_ws_ping_shim()
    once = stream.websockets
    adapter._install_sdk_ws_ping_shim()
    assert stream.websockets is once, 'a second install wrapped the shim again'
    assert sys.modules.get('websockets') is global_ws, 'the process-global websockets module was patched'
    assert stream.websockets.exceptions == 'sdk-websockets-exceptions', 'other attributes must pass through'
