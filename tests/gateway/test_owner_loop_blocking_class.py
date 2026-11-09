"""Blocking work seen in live gateway slow-loop stacks stays off the owner loop."""

from __future__ import annotations

import asyncio
import builtins
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_turn_context_approval_read_runs_off_loop(monkeypatch):
    from gateway.run import GatewayRunner

    loop_thread = threading.get_ident()
    threads: list[int] = []
    authority = SimpleNamespace(
        sessions={"chat": object()},
        logical_owner=lambda _session_id: "chat",
        db=SimpleNamespace(get_session=lambda owner: (
            threads.append(threading.get_ident()), {"runtime_generation": 7})[1]),
    )
    runner = object.__new__(GatewayRunner)
    runner.session_authority = authority
    runner._voice_ack_guild_for = lambda _source: None
    runner._delivery_adapter_for = lambda _source: None
    runner._run_still_current_fn = lambda _key, _generation: (lambda: True)
    display_values = {name: None for name in GatewayRunner._DISPLAY_TO_TURN_CTX}
    display_values.update(
        needs_progress_queue=False,
        resolve_display_setting=lambda *_args, **_kwargs: False,
        platform_key="local",
        user_config={},
    )
    display = SimpleNamespace(**display_values)

    _ctx, turn_runner, _cleanup = await GatewayRunner._run_agent_build_turn_context(
        runner, display, object, message="hello",
        source=SessionSource(platform=Platform.LOCAL, chat_id="chat", user_id="user"),
        session_key="key", run_generation=1, history=[], context_prompt=None,
    )

    assert turn_runner._approval_owner == (authority, "chat", 7)
    assert threads and all(thread != loop_thread for thread in threads)


@pytest.mark.asyncio
async def test_launch_profile_scope_files_run_off_loop(monkeypatch, tmp_path):
    from tui_gateway import launch_profile_policy

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: tmp_path)
    loop_thread = threading.get_ident()
    secret_threads: list[int] = []
    terminal_threads: list[int] = []
    monkeypatch.setattr(
        launch_profile_policy, "launch_secret_scope",
        lambda home: (secret_threads.append(threading.get_ident()), {})[1])
    monkeypatch.setattr(
        "tools.terminal_scope.build_profile_terminal_scope",
        lambda *_args, **_kwargs: (terminal_threads.append(threading.get_ident()), {})[1])

    async with launch_profile_policy.async_launch_profile_scope_if_multiplexed():
        pass

    assert secret_threads and secret_threads[0] != loop_thread
    assert terminal_threads and terminal_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_routed_profile_terminal_policy_runs_off_loop(monkeypatch, tmp_path):
    from gateway.run import _async_profile_runtime_scope

    loop_thread = threading.get_ident()
    threads: list[int] = []
    monkeypatch.setattr(
        "tools.terminal_scope.build_profile_terminal_scope",
        lambda *_args, **_kwargs: (threads.append(threading.get_ident()), {})[1])
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda *_args, **_kwargs: None)

    async with _async_profile_runtime_scope(tmp_path):
        pass

    assert threads and threads[0] != loop_thread


@pytest.mark.asyncio
async def test_loop_watcher_cold_import_runs_off_loop(monkeypatch):
    from gateway import run_goals
    from gateway.run import GatewayRunner

    runner = SimpleNamespace(
        _running=True,
        _warm_goals_session_db=_make_async(lambda *args, **kwargs: None),
    )

    async def run_in_executor(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    runner._run_in_executor_with_context = run_in_executor

    async def scopes(_runner):
        return [(None, None)]

    class _NoScope:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    sleep_calls = 0

    async def stop_after_scan_wait(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            runner._running = False

    monkeypatch.setattr("gateway.run._resolve_handoff_watch_scopes", scopes)
    monkeypatch.setattr(
        "tui_gateway.launch_profile_policy.async_launch_profile_scope_if_multiplexed",
        _NoScope)
    monkeypatch.setattr(run_goals.asyncio, "sleep", stop_after_scan_wait)
    loop_thread = threading.get_ident()
    import_threads: list[int] = []
    real_import = builtins.__import__

    def recorded_import(name, *args, **kwargs):
        if name == "hermes_cli.loops":
            import_threads.append(threading.get_ident())
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recorded_import)

    await GatewayRunner._loop_wakeup_watcher(runner, interval=0)

    assert import_threads and all(thread != loop_thread for thread in import_threads)


@pytest.mark.asyncio
async def test_runtime_footer_config_read_runs_off_loop(monkeypatch, tmp_path):
    from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source

    runner = _runner(monkeypatch, tmp_path)
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def footer(*_args, **_kwargs):
        threads.append(threading.get_ident())
        return "usage footer"

    runner._hmwa_runtime_footer_line = footer
    runner._run_agent = _make_async({
        "final_response": "answer", "messages": [], "failed": False})

    await runner._handle_message_with_agent(
        _event(reply_expected=True), _source(), "agent:main:telegram:group:-1001:12345", 1)

    assert threads and threads[0] != loop_thread


@pytest.mark.asyncio
async def test_feishu_client_preparation_runs_off_loop(monkeypatch):
    from plugins.platforms.feishu import adapter as feishu

    adapter = object.__new__(feishu.FeishuAdapter)
    adapter._loop = SimpleNamespace(is_closed=lambda: False)
    adapter._ws_future = None
    adapter._app_id = "app"
    adapter._app_secret = "secret"
    adapter._event_handler = object()
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def prepare():
        threads.append(threading.get_ident())
        return "domain"

    async def hydrate():
        return None

    adapter._prepare_client = prepare
    adapter._hydrate_bot_identity = hydrate
    adapter._get_sdk_executor = lambda: None
    monkeypatch.setattr(feishu, "FEISHU_WEBSOCKET_AVAILABLE", True)
    monkeypatch.setattr(feishu, "lark", SimpleNamespace(LogLevel=SimpleNamespace(INFO=1)))
    monkeypatch.setattr(feishu, "FeishuWSClient", lambda **_kwargs: SimpleNamespace())
    adapter._loop.run_in_executor = lambda executor, callback, *args, **kwargs: SimpleNamespace()

    await adapter._connect_websocket()

    assert threads and threads[0] != loop_thread


def _make_async(value):
    async def async_value(*_args, **_kwargs):
        return value(*_args, **_kwargs) if callable(value) else value
    return async_value
