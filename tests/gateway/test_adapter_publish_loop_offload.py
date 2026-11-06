"""API-turn recovery must keep SQLite I/O off the gateway event loop."""
import asyncio
import threading
from types import SimpleNamespace

import gateway.session_api_turn as api_turn
from gateway.session_authorities import owner_scope


def test_api_turn_recovery_io_runs_off_loop_and_schedules_on_loop(monkeypatch) -> None:
    threads: dict[str, int] = {}
    refs = [SimpleNamespace(session_id="queued")]
    authority = SimpleNamespace(profile_id="profile", db=SimpleNamespace())

    def blocking(_adapter, recovered_authority) -> list[SimpleNamespace]:
        assert recovered_authority is authority
        threads["blocking"] = threading.get_ident()
        return refs

    def schedule(ref) -> None:
        assert ref is refs[0]
        threads["schedule"] = threading.get_ident()

    authority._schedule = schedule
    monkeypatch.setattr(api_turn, "_recover_authority_api_turns", blocking)
    monkeypatch.setattr(
        "gateway.session_authorities.serving_authorities",
        lambda _runner: [authority],
    )
    adapter = SimpleNamespace(gateway_runner=SimpleNamespace())

    async def recover() -> None:
        threads["loop"] = threading.get_ident()
        with owner_scope(authority):
            await api_turn.recover_api_turns_off_loop(adapter)

    asyncio.run(recover())
    assert threads["blocking"] != threads["loop"]
    assert threads["schedule"] == threads["loop"]
