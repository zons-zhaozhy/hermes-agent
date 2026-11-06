"""The DingTalk stream socket keeps a stall-tolerant protocol ping window.

``dingtalk_stream.stream.start`` opens ``websockets.connect(uri)`` with the library defaults (20 s
ping, 20 s pong timeout) while the SDK also pings at the application level every 60 s. A late pong
then reads as a dead socket (``sent 1011 (internal error): keepalive ping timeout``) and costs a
full reconnect — three times a day on the reference install. The adapter's shim widens the protocol
window inside the SDK's own namespace only.
"""
from __future__ import annotations

import sys
from types import ModuleType

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
    """A stand-in ``dingtalk_stream.stream`` module whose ``websockets.connect`` records kwargs."""
    from plugins.platforms.dingtalk import adapter

    fake_ws = _FakeWebsockets()
    stream = ModuleType('dingtalk_stream.stream')
    stream.websockets = fake_ws
    package = ModuleType('dingtalk_stream')
    package.stream = stream
    monkeypatch.setitem(sys.modules, 'dingtalk_stream', package)
    monkeypatch.setitem(sys.modules, 'dingtalk_stream.stream', stream)
    monkeypatch.setattr(adapter, '_sdk_ws_shim_installed', False)
    return adapter, stream, fake_ws


def test_the_sdk_stream_socket_gets_a_wider_ping_window(sdk_stream):
    adapter, stream, fake_ws = sdk_stream
    adapter._install_sdk_ws_ping_shim()
    stream.websockets.connect('wss://example.invalid/stream', open_timeout=10)
    assert fake_ws.calls == [{'uri': 'wss://example.invalid/stream', 'open_timeout': 10,
                              'ping_interval': adapter._SDK_WS_PING_INTERVAL_S,
                              'ping_timeout': adapter._SDK_WS_PING_TIMEOUT_S}]
    assert adapter._SDK_WS_PING_TIMEOUT_S > adapter._SDK_WS_PING_INTERVAL_S > 20


def test_the_shim_is_scoped_to_the_sdk_and_idempotent(sdk_stream):
    adapter, stream, _ = sdk_stream
    global_ws = sys.modules.get('websockets')
    adapter._install_sdk_ws_ping_shim()
    once = stream.websockets
    adapter._install_sdk_ws_ping_shim()
    assert stream.websockets is once, 'a second install wrapped the shim again'
    assert sys.modules.get('websockets') is global_ws, 'the process-global websockets module was patched'
    assert stream.websockets.exceptions == 'sdk-websockets-exceptions', 'other attributes must pass through'
