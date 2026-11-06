"""A secondary profile's reconnect reads its gateway config off the owner event loop."""
import threading

import pytest

from gateway.config import GatewayConfig
from tests.gateway.test_multiplex_adapter_registry import (
    _SecondaryRecoveryAdapter,
    _install_secondary_reconnect_context,
    _secondary_recovery_runner,
)


@pytest.mark.asyncio
async def test_secondary_reconnect_config_read_runs_in_worker_thread(monkeypatch):
    """Config file loading is synchronous I/O; a slow profile disk must not stall every session."""
    runner = _secondary_recovery_runner()
    _install_secondary_reconnect_context(monkeypatch, runner, _SecondaryRecoveryAdapter())
    loop_thread_id = threading.get_ident()
    config_thread_ids = []

    def slow_config_disk():
        config_thread_ids.append(threading.get_ident())
        return GatewayConfig(multiplex_profiles=True, platforms={})

    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda home: {})
    monkeypatch.setattr("gateway.config.load_gateway_config", slow_config_disk)
    stood_down = []
    monkeypatch.setattr(
        runner,
        "_update_platform_runtime_status",
        lambda *args, **kwargs: stood_down.append((args, kwargs)),
    )

    assert await runner._secondary_reconnect_attempt(
        "reviewer", _SecondaryRecoveryAdapter.platform, None
    ) == (None, None)
    assert config_thread_ids == [config_thread_ids[0]]
    assert config_thread_ids[0] != loop_thread_id
