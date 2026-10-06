"""misfire catch-up 齐射防线：fire_overdue_jobs 必须经统一并行池分发，禁裸 Thread 齐射。

修前行为：每个 overdue job 起一个裸 threading.Thread 同秒开火，绕过
cron.max_parallel_jobs / HERMES_CRON_MAX_PARALLEL 并发上限——2026-10-06 08:07
gateway 重启后 2h 积压补跑三任务同秒齐发，实测触发 23 条账户级限速(code 1302)。
修后行为：submit 至 _get_parallel_pool(_resolve_max_parallel_workers())，与
scheduler_tick 主路径共用同一并发上限（单一事实源）。
"""

from typing import Any

import cron.scheduler as _sched
import cron.scheduler_provider as sp


class _RecordingProvider:
    """记录 fire_claimed 被调方式的假 provider（非 InProcess 以走 misfire 路径）。"""

    name = "recording"

    def __init__(self) -> None:
        self.fired: list = []
        self._claims: dict = {}

    def claim_fire(self, job_id: str) -> dict:
        return self._claims.setdefault(job_id, {"id": job_id})

    def fire_claimed(self, claimed_job: dict, *, adapters: Any = None, loop: Any = None,
                     cancel_event: Any = None) -> bool:
        self.fired.append(claimed_job["id"])
        return True


def test_fire_overdue_jobs_uses_parallel_pool(monkeypatch, tmp_path) -> None:
    submitted: list = []

    class _FakePool:
        def submit(self, fn: Any, *a: Any, **kw: Any) -> None:
            submitted.append((fn, a, kw))
            fn(*a, **kw)

    monkeypatch.setattr(_sched, "_resolve_max_parallel_workers", lambda: 1)
    monkeypatch.setattr(_sched, "_get_parallel_pool", lambda max_workers: _FakePool())

    provider = _RecordingProvider()
    jobs = [
        {"id": "job-a", "name": "A", "enabled": True, "paused": False,
         "next_run_at": "2026-10-06T06:00:00+08:00", "schedule": {"kind": "cron", "expr": "0 3 * * *"}},
        {"id": "job-b", "name": "B", "enabled": True, "paused": False,
         "next_run_at": "2026-10-06T06:15:00+08:00", "schedule": {"kind": "cron", "expr": "0 4 * * *"}},
    ]
    import cron.jobs as _jobs_mod
    monkeypatch.setattr(_jobs_mod, "load_jobs", lambda: jobs, raising=False)
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 10, 6, 8, 10, 0, tzinfo=timezone(timedelta(hours=8)))

    fired = sp.fire_overdue_jobs(provider, now=now)

    assert fired == 2  # 期望: 两个 overdue job 都被分发(06:00/06:15 距 08:10 均超 grace 10min)
    assert len(submitted) == 2  # 期望: 每 job 恰一次 submit(池通道)
    assert provider.fired == ["job-a", "job-b"]  # 期望: FakePool 同步执行后两 job 都真实 fire
    assert all(fn == provider.fire_claimed for fn, _a, _kw in submitted)  # 期望: 分发目标是 fire_claimed
