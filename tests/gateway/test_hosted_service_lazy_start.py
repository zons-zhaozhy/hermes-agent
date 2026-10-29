"""Multiplexed hosted-room supervisors start only when their profile owns work."""

from __future__ import annotations

from types import SimpleNamespace


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
