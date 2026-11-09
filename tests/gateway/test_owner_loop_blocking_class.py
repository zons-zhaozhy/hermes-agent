"""Blocking work seen in live gateway slow-loop stacks stays off the owner loop."""

from __future__ import annotations

import asyncio
import builtins
import threading
from pathlib import Path
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


@pytest.mark.asyncio
async def test_cron_cancel_poll_runs_off_loop():
    from cron.scheduler_authority import cancel_requested_off_loop

    loop_thread = threading.get_ident()
    threads: list[int] = []

    class CancelEvent:
        def is_set(self) -> bool:
            threads.append(threading.get_ident())
            return False

    assert await cancel_requested_off_loop(CancelEvent()) is False
    assert threads and threads[0] != loop_thread


def test_cron_cancel_home_resolution_is_cached(monkeypatch, tmp_path):
    from cron import cancellation
    from hermes_constants import (
        reset_hermes_home_key_cache, reset_hermes_home_override,
        set_hermes_home_override,
    )

    calls: list[Path] = []
    real_resolve = Path.resolve

    def counted_resolve(path: Path, strict: bool = False) -> Path:
        if path == tmp_path:
            calls.append(path)
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", counted_resolve)
    reset_hermes_home_key_cache()
    token = set_hermes_home_override(tmp_path)
    try:
        first = cancellation._cancels_dir()
        second = cancellation._cancels_dir()
    finally:
        reset_hermes_home_override(token)

    assert first == second
    assert len(calls) == 1


def test_runtime_status_home_resolution_is_cached(monkeypatch, tmp_path):
    from gateway.status_identity import canonical_hermes_home
    from hermes_constants import reset_hermes_home_key_cache

    calls: list[Path] = []
    real_resolve = Path.resolve

    def counted_resolve(path: Path, strict: bool = False) -> Path:
        if path == tmp_path:
            calls.append(path)
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", counted_resolve)
    reset_hermes_home_key_cache()
    try:
        first = canonical_hermes_home(tmp_path)
        second = canonical_hermes_home(tmp_path)
    finally:
        reset_hermes_home_key_cache()

    assert first == second
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_api_heartbeat_status_publish_runs_off_loop(monkeypatch):
    from gateway.platforms import api_server

    adapter = object.__new__(api_server.APIServerAdapter)
    adapter._running = True
    adapter._metrics_last_heartbeat_at = None
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def publish(*_args, **_kwargs):
        threads.append(threading.get_ident())

    adapter._write_runtime_status_safe = publish
    adapter._api_server_status_payload = dict

    async def stop_after_first_heartbeat(_delay):
        adapter._running = False

    monkeypatch.setattr(api_server.asyncio, "sleep", stop_after_first_heartbeat)
    await adapter._heartbeat_loop()

    assert threads and threads[0] != loop_thread


@pytest.mark.asyncio
async def test_control_socket_peer_subject_runs_off_loop():
    from gateway.control_socket import GatewayControlServer

    server = object.__new__(GatewayControlServer)
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def subject(_writer):
        threads.append(threading.get_ident())
        return "uid:1"

    class Dispatcher:
        async def dispatch(self, _handler, _line, _subject):
            return b"{}\n"

    class Reader:
        async def readline(self):
            return b"{}\n"

    class Writer:
        def __init__(self):
            self.closed = False

        def write(self, _data):
            return None

        async def drain(self):
            return None

        def close(self):
            self.closed = True

    server._request_dispatch = Dispatcher()
    server._posix_peer_subject = subject
    writer = Writer()
    await server._handle_connection(Reader(), writer)

    assert writer.closed
    assert threads and threads[0] != loop_thread


@pytest.mark.asyncio
async def test_profile_reconcile_avoids_path_hashing_on_loop(monkeypatch, tmp_path):
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin

    class RecordedPath(Path):
        def __hash__(self):
            path_threads.append(threading.get_ident())
            return super().__hash__()

    loop_thread = threading.get_ident()
    path_threads: list[int] = []
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gateway.run_profile_reconcile.Path", RecordedPath)
    monkeypatch.setattr(
        "hermes_constants.get_process_hermes_home", lambda: RecordedPath(tmp_path))

    runner = SimpleNamespace(
        _served_profile_homes={"profile": RecordedPath(tmp_path / "profile")},
        _hook_consent_signatures=None,
    )

    def resync(homes, seen):
        captured["homes"] = homes
        captured["seen"] = seen
        return {}

    runner._resync_changed_hook_consent = resync
    await GatewayProfileReconcileMixin._sync_hooks_on_consent_change(runner)

    assert isinstance(captured["homes"], dict)
    assert captured["seen"] == {}
    assert all(thread != loop_thread for thread in path_threads)


def test_hook_consent_resync_reads_home_values_not_dict_keys(monkeypatch, tmp_path):
    from contextlib import nullcontext

    from agent import shell_hooks
    from gateway import run
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
    from gateway.run_turn import GatewayTurnMixin

    launch_home = tmp_path / "launch-home"
    served_home = tmp_path / "served-home"
    launch_home.mkdir()
    served_home.mkdir()
    loaded: list[object] = []
    synced: list[object] = []
    monkeypatch.setattr(run, "_load_gateway_config", lambda config_path: loaded.append(config_path) or {})
    monkeypatch.setattr(shell_hooks, "sync_from_config", lambda load: synced.append(load()))
    monkeypatch.setattr(GatewayTurnMixin, "_standalone_launch_scope", staticmethod(nullcontext))
    runner = SimpleNamespace(_multiplex_on=lambda: False)

    current = GatewayProfileReconcileMixin._resync_changed_hook_consent(
        runner, {str(launch_home): launch_home, str(served_home): served_home}, {}
    )

    assert [str(home) for home in loaded] == [
        str(launch_home.resolve() / "config.yaml"), str(served_home.resolve() / "config.yaml")]
    assert len(synced) == 2
    assert list(current) == [str(launch_home.resolve()), str(served_home.resolve())]


def _make_async(value):
    async def async_value(*_args, **_kwargs):
        return value(*_args, **_kwargs) if callable(value) else value
    return async_value
