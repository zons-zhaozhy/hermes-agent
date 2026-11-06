"""A settled hosted service costs no profile-scope work per watcher tick.

``GatewayRunner._hosted_room_worker_watcher`` re-enters every served authority once a second. Entering
a profile scope reads that profile's ``.env`` and installs its terminal policy from disk, so doing it
for a settled authority multiplied one disk read per profile per second and froze the owner loop often
enough for the liveness watchdog to restart the gateway.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

from gateway import session_hosted_service, session_hosted_transport


def _authority(service):
    return SimpleNamespace(
        profile_id='/home/profile', epoch=1, hosted_room_service=service,
        db=SimpleNamespace(db_path='/home/profile/state.db'))


def _runner():
    return SimpleNamespace(session_control_server=object())


def test_settled_authority_enters_no_profile_scope(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def scope(authority, **_kwargs):
        entered.append(authority)
        yield

    monkeypatch.setattr(session_hosted_service, 'owner_scope', scope)
    authority = _authority(SimpleNamespace(_transport_installed=True))

    asyncio.run(session_hosted_service._ensure_hosted_service(_runner(), authority))

    assert entered == [], "a settled authority re-read its profile scope from the watcher tick"


def test_unsettled_authority_still_enters_profile_scope(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def scope(authority, **_kwargs):
        entered.append(authority)
        yield

    installed = []
    monkeypatch.setattr(session_hosted_service, 'owner_scope', scope)
    monkeypatch.setattr(session_hosted_transport, 'install_hosted_transport',
                        lambda server, authority, loop, *, attest: installed.append(authority))
    service = SimpleNamespace(_transport_installed=False, attest=lambda: None)
    authority = _authority(service)

    asyncio.run(session_hosted_service._ensure_hosted_service(_runner(), authority))

    assert entered == [authority], "preparing the transport must happen inside the owning profile scope"
    assert installed == [authority]
    assert service._transport_installed is True
