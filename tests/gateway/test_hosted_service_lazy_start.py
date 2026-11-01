"""Multiplexed hosted-room supervisors start only when their profile owns work, off the owner loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_readiness_probe_never_blocks_the_owner_loop(monkeypatch):
    """A slow shared-store `bindings()` probe must not starve WebSocket writes; the watcher awaits it."""
    import asyncio
    import threading
    from gateway import run_startup, session_hosted_service
    from gateway import hosted_rooms_gateway_import
    from gateway.run import GatewayRunner

    release, responsive, fallback = threading.Event(), threading.Event(), threading.Event()

    def slow_bindings():
        release.wait(2)
        return ()

    service = SimpleNamespace(_transport_installed=True, _ready_import_started=True,
                              runtime=SimpleNamespace(status=lambda: {"running": False}),
                              bindings=slow_bindings, start=lambda: None)
    runner = SimpleNamespace(
        _running=True, _draining=False, session_runtime_descriptor={"state": "ready"},
        session_authority=object(), hosted_room_service=service)

    async def ensure(_runner):
        return service

    monkeypatch.setattr(session_hosted_service, "serving_authorities", lambda _runner: [object()])
    monkeypatch.setattr(hosted_rooms_gateway_import, "import_after_ready", lambda _authorities: None)
    loop = asyncio.get_running_loop()

    def probe():
        release.wait(0.1)
        loop.call_soon_threadsafe(responsive.set)
        if not responsive.wait(0.2):
            fallback.set()
        release.set()

    threading.Thread(target=probe, name="bindings-probe", daemon=True).start()

    task = asyncio.create_task(GatewayRunner._hosted_room_worker_watcher(runner, interval=0.01))
    await asyncio.to_thread(responsive.wait, 1)
    task.cancel()
    assert not fallback.is_set(), "the hosted-room readiness probe blocked the owner loop"

def test_ready_gateway_leaves_empty_profile_stores_dormant(monkeypatch):
    from gateway import session_hosted_service

    started = []

    def service(bindings, service_id):
        return SimpleNamespace(
            _transport_installed=True,
                _ready_import_started=True,
            bindings=lambda: bindings,
            start=lambda: started.append(service_id),
        )

    empty_id, occupied_id = object(), object()
    empty, occupied = service((), empty_id), service(("room-1",), occupied_id)
    authorities = [
        SimpleNamespace(hosted_room_service=empty),
        SimpleNamespace(hosted_room_service=occupied),
    ]
    runner = SimpleNamespace(
        session_runtime_descriptor={"state": "ready"},
        _draining=False,
    )
    monkeypatch.setattr(session_hosted_service, "serving_authorities", lambda _runner: authorities)

    session_hosted_service.start_ready_hosted_services(runner)

    assert started == [occupied_id]
