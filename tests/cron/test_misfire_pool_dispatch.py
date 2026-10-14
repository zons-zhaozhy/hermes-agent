"""misfire catch-up 齐射防线：fire_overdue_jobs 必须经统一并行池分发，禁裸 Thread 齐射。

修前行为：每个 overdue job 起一个裸 threading.Thread 同秒开火，绕过
cron.max_parallel_jobs / HERMES_CRON_MAX_PARALLEL 并发上限——2026-10-06 08:07
gateway 重启后 2h 积压补跑三任务同秒齐发，实测触发 23 条账户级限速(code 1302)。
修后行为：submit 至 _get_parallel_pool(_resolve_max_parallel_workers())，与
scheduler_tick 主路径共用同一并发上限（单一事实源）；批内首请求密度控制置于
claim_fire 之前（claim 盖章时刻=实际运行起点，claim 不在等待中老化），经
await_first_request_slot 令牌桶按 15s 最小间隔排队、300s 封顶、可被进程关停
中断，中断即放弃开火、不留悬挂 claim；批次前 3 个作业与单 worker 池零等待直发。
"""

from typing import Any

import cron.scheduler as _sched
import cron.scheduler_provider as sp


class _RecordingProvider:
    """记录 claim/fire 调用顺序的假 provider（非 InProcess 以走 misfire 路径）。"""

    name = "recording"

    def __init__(self) -> None:
        self.fired: list = []
        self.claims: list = []
        self.events: list = []
        self._claim_map: dict = {}

    def claim_fire(self, job_id: str) -> dict:
        self.claims.append(job_id)
        self.events.append(f"claim:{job_id}")
        return self._claim_map.setdefault(job_id, {"id": job_id})

    def fire_claimed(self, claimed_job: dict, *, adapters: Any = None, loop: Any = None,
                     cancel_event: Any = None) -> bool:
        self.events.append(f"fire:{claimed_job['id']}")
        self.fired.append(claimed_job["id"])
        return True


def _make_jobs(n: int) -> list:
    return [
        {"id": f"job-{i}", "name": f"J{i}", "enabled": True, "paused": False,
         "next_run_at": "2026-10-13T04:00:00+08:00", "schedule": {"kind": "cron", "expr": "0 3 * * *"}}
        for i in range(n)
    ]


def _dispatch_via_fake_pool(monkeypatch: Any, jobs: list, *, max_workers: int,
                            await_result: bool = True) -> tuple:
    """FakePool 同步执行 submit 的 fn；打点 await_first_request_slot 的调用与返回。"""
    submitted: list = []
    awaits: list = []

    class _FakePool:
        def submit(self, fn: Any, *a: Any, **kw: Any) -> None:
            submitted.append((fn, a))
            fn(*a, **kw)

    def _fake_await() -> bool:
        awaits.append(True)
        return await_result

    _sched._last_first_request_monotonic = float("-inf")
    _sched._first_request_stagger_abort.clear()
    monkeypatch.setattr(_sched, "_resolve_max_parallel_workers", lambda: max_workers)
    monkeypatch.setattr(_sched, "_get_parallel_pool", lambda mw: _FakePool())
    monkeypatch.setattr(_sched, "await_first_request_slot", _fake_await)
    import cron.jobs as _jobs_mod
    monkeypatch.setattr(_jobs_mod, "load_jobs", lambda: jobs, raising=False)
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 10, 13, 6, 18, 0, tzinfo=timezone(timedelta(hours=8)))
    provider = _RecordingProvider()
    fired = sp.fire_overdue_jobs(provider, now=now)
    return provider, fired, submitted, awaits


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
    assert all(fn.__name__ == "_staggered_fire" for fn, _a, _kw in submitted)  # 期望: 分发目标为密度控制包装函数


def test_stagger_applies_from_fourth_job_in_batch(monkeypatch, tmp_path) -> None:
    """灾后恢复批次密度控制: 批内第 4 个起(0 基序号 3)的作业须取首请求槽位。

    10-13 06:18 断网恢复后 6 个 cron 班次 55s 内齐发首请求, 打穿 1302 账户限速——
    池(296ba5a93c)只约束同时运行数, 不约束首请求密度。前 3 个作业同发历史上安全,
    从第 4 个起排队取槽(15s 间隔), 防护面落在风暴形态(≥6 作业)上。
    """
    provider, fired, submitted, awaits = _dispatch_via_fake_pool(
        monkeypatch, _make_jobs(6), max_workers=4)

    assert fired == 6  # 期望: 六个灾后积压作业全部分发
    assert len(submitted) == 6  # 期望: 每作业恰一次 submit
    assert len(awaits) == 3  # 期望: 仅序号 3/4/5 三作业取槽(阈值 4: seq+1>=4)
    assert provider.fired == [f"job-{i}" for i in range(6)]  # 期望: 密度控制不改变分发完整性
    assert provider.events[:6] == ["claim:job-0", "fire:job-0", "claim:job-1", "fire:job-1",
                                   "claim:job-2", "fire:job-2"]  # 期望: 前 3 作业 claim/fire 交替先行
    assert provider.claims == [f"job-{i}" for i in range(6)]  # 期望: 槽位放行后 6 作业全部 claim


def test_small_batch_untouched(monkeypatch, tmp_path) -> None:
    """小批次(3 作业)不触发密度控制: 与修前行为逐位一致, 零额外等待。"""
    provider, fired, _submitted, awaits = _dispatch_via_fake_pool(
        monkeypatch, _make_jobs(3), max_workers=8)

    assert fired == 3  # 期望: 3 作业全部分发
    assert awaits == []  # 期望: 3 作业批次低于阈值 4, 零槽位调用
    assert provider.fired == [f"job-{i}" for i in range(3)]  # 期望: 全部直发


def test_single_worker_skip(monkeypatch, tmp_path) -> None:
    """单 worker 池并发密度恒为 1, 槽位只增拖尾延迟——应零槽位调用直发。"""
    provider, fired, _submitted, awaits = _dispatch_via_fake_pool(
        monkeypatch, _make_jobs(6), max_workers=1)

    assert fired == 6  # 期望: 单 worker 不影响分发完整性
    assert awaits == []  # 期望: max_workers<=1 恒跳过槽位
    assert provider.fired == [f"job-{i}" for i in range(6)]  # 期望: 全部直发


def test_shutdown_abort_abandons_before_claim(monkeypatch, tmp_path) -> None:
    """进程关停中断槽位等待: 放弃开火且该作业 claim_fire 零调用(不留悬挂 claim)。"""
    provider, fired, _submitted, awaits = _dispatch_via_fake_pool(
        monkeypatch, _make_jobs(6), max_workers=4, await_result=False)

    assert fired == 6  # 期望: 分发计数照常(语义=已分发数)
    assert len(awaits) == 3  # 期望: 序号 3/4/5 三作业进入槽位等待
    assert provider.fired == [f"job-{i}" for i in range(3)]  # 期望: 仅前 3 直发作业真实开火
    assert provider.claims == [f"job-{i}" for i in range(3)]  # 期望: 放弃发生在 claim 之前, 后 3 作业 store 零触碰


def test_first_request_stagger_enabled_truth_table() -> None:
    """谓词真值表: 批次阈值(4)×池 worker 数共同决定是否取槽。"""
    assert _sched.first_request_stagger_enabled(0, None) is False  # 期望: 批内第 1 作业永不取槽
    assert _sched.first_request_stagger_enabled(2, 8) is False  # 期望: 第 3 作业仍低于阈值 4
    assert _sched.first_request_stagger_enabled(3, 8) is True  # 期望: 第 4 作业起取槽(多 worker)
    assert _sched.first_request_stagger_enabled(3, None) is True  # 期望: 无上界池同样取槽
    assert _sched.first_request_stagger_enabled(9, 1) is False  # 期望: 单 worker 恒跳过(密度已为 1)


def test_slot_bucket_min_interval(monkeypatch) -> None:
    """令牌桶最小间隔: 连续取槽的第二次须等满间隔(实测时间轴, 间隔缩至 50ms 防慢测)。"""
    import time as _time
    monkeypatch.setattr(_sched, "_FIRST_REQUEST_MIN_INTERVAL_SECONDS", 0.05)
    _sched._last_first_request_monotonic = float("-inf")
    _sched._first_request_stagger_abort.clear()
    try:
        assert _sched.await_first_request_slot() is True  # 期望: 首槽立即可用
        t0 = _time.monotonic()
        assert _sched.await_first_request_slot() is True  # 期望: 第二槽等待后放行
        elapsed = _time.monotonic() - t0
        assert elapsed >= 0.04  # 期望: 实际等待不低于间隔(0.05s 减 20% 计时容差)
    finally:
        _sched._last_first_request_monotonic = float("-inf")


def test_slot_bucket_wait_cap(monkeypatch) -> None:
    """等待封顶: 槽位目标超出最大等待时截断到 _FIRST_REQUEST_MAX_WAIT_SECONDS。"""
    import time as _time
    monkeypatch.setattr(_sched, "_FIRST_REQUEST_MIN_INTERVAL_SECONDS", 5.0)
    monkeypatch.setattr(_sched, "_FIRST_REQUEST_MAX_WAIT_SECONDS", 0.15)
    _sched._last_first_request_monotonic = float("-inf")
    _sched._first_request_stagger_abort.clear()
    try:
        assert _sched.await_first_request_slot() is True  # 期望: 首槽立即可用
        t0 = _time.monotonic()
        assert _sched.await_first_request_slot() is True  # 期望: 封顶等待后放行(非 5s 全等)
        elapsed = _time.monotonic() - t0
        assert 0.1 <= elapsed < 1.0  # 期望: 等待被截断到 ~0.15s 而非完整 5s 间隔
    finally:
        _sched._last_first_request_monotonic = float("-inf")


def test_slot_bucket_abort() -> None:
    """关停中断: abort 事件置位后取槽立即返回 False(调用方须放弃开火)。"""
    import time as _time
    _sched._last_first_request_monotonic = float("-inf")
    _sched._first_request_stagger_abort.clear()
    _sched._first_request_stagger_abort.set()
    try:
        t0 = _time.monotonic()
        assert _sched.await_first_request_slot() is False  # 期望: 已置位 abort → 立即放弃
        assert _time.monotonic() - t0 < 1.0  # 期望: 立即返回, 不等满 15s 间隔
    finally:
        _sched._first_request_stagger_abort.clear()
        _sched._last_first_request_monotonic = float("-inf")
