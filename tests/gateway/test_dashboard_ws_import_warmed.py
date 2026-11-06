"""The native session WebSocket imports its stack during boot warm-up, never on the owner loop.

``GatewayRuntimeAPI._serve`` imports ``tui_gateway.ws`` lazily when an upgrade arrives. Paid inline
that import read hundreds of modules on the owner loop — 7.4 s measured on a loaded host, which froze
every platform heartbeat and lost the upgrade's ``ready`` frame to its own send timeout.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
from types import SimpleNamespace

from gateway import run_api
from gateway.run import GatewayRunner


def test_boot_warmup_imports_the_dashboard_ws_stack_off_the_owner_loop(monkeypatch):
    """Warm-up must import it on an executor thread, so a later client upgrade is a cache hit."""
    import_threads: dict[str, int] = {}
    real_import = importlib.import_module

    def spy(name, *args, **kwargs):
        if name == 'tui_gateway.ws':
            import_threads[name] = threading.get_ident()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, 'import_module', spy)
    loop_threads: list[int] = []

    async def exploit(_fn):
        return await asyncio.get_running_loop().run_in_executor(None, _fn)

    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False),
                             _run_boot_probe_in_launch_scope=exploit)

    async def scenario():
        loop_threads.append(threading.get_ident())
        await GatewayRunner._warm_turn_prerequisites(runner)

    asyncio.run(scenario())

    assert 'tui_gateway.ws' in import_threads, "boot warm-up never imported the session WebSocket stack"
    assert import_threads['tui_gateway.ws'] not in loop_threads, (
        "the WebSocket stack was imported on the owner loop, where it stalls every platform heartbeat")


def test_ws_stack_is_importable_and_cached(monkeypatch):
    """The warm-up leaves the real module in ``sys.modules`` so ``_serve``'s lazy import is a no-op."""
    import sys

    saved = sys.modules.pop('tui_gateway.ws', None)
    try:
        run_api.warm_gateway_ws_stack()
        assert 'tui_gateway.ws' in sys.modules
    finally:
        if saved is not None:
            sys.modules['tui_gateway.ws'] = saved
