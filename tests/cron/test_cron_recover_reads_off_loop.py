"""Cron recovery reads the admission ledger off the gateway owner loop."""
import asyncio
import threading
from types import SimpleNamespace

from gateway import session_cron
import hermes_state_runtime


def test_cron_recover_admission_read_runs_in_worker_thread(tmp_path, monkeypatch):
    """A recovery lookup is SQLite I/O and must not block every gateway callback."""
    authority = SimpleNamespace(
        profile_id="profile", db=SimpleNamespace(db_path=tmp_path / "state.db")
    )
    loop_thread_id = threading.get_ident()
    read_thread_ids = []

    def read_admissions(*args, **kwargs):
        read_thread_ids.append(threading.get_ident())
        return []

    monkeypatch.setattr(hermes_state_runtime, "list_session_admissions", read_admissions)

    async def probe():
        return await session_cron.operation(
            authority,
            "recover",
            {"job_id": "job", "request_id": "fire", "extra_prompt": None},
        )

    assert asyncio.run(probe()) == {"status": "missing", "result": None}
    assert read_thread_ids == [read_thread_ids[0]]
    assert read_thread_ids[0] != loop_thread_id
