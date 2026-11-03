"""slow_loop_monitor 空闲 select 等待不得误报为占用（2026-11-04 74.6s 事件定性）。

现场：06:12:55/06:14:56/06:29:01 三起「occupied 74.6s/33.9s/46.8s」报告的栈顶
全部为 selectors.py:548 select——即网关空闲等待形态被当作 callback 占用计
时（签名持续=空闲持续），报告文案把空闲时长写成占用时长，污染 slow_loop
升级线判据。期望（独立于实现推导）：
- 被监控线程停在 selectors.select 空闲等待（时长超阈值）→ 不得产生
  「occupied the event loop」报告（空闲等待非占用）；
- 真同步阻塞（time.sleep，栈顶在用户代码）→ 仍报告（既有不变量）。
"""

from __future__ import annotations

import logging
import selectors
import threading

from gateway.slow_loop_monitor import SlowLoopStackMonitor


def _idle_wait_for_test() -> None:
    # health: allow HX013 -- 模拟网关空闲:阻塞在标准库 select 等待(非用户代码占用)
    sel = selectors.DefaultSelector()
    try:
        sel.select(0.4)
    finally:
        sel.close()


def test_idle_select_wait_is_not_reported_as_occupied(caplog):
    monitor = SlowLoopStackMonitor.start(
        thread_id=threading.get_ident(), threshold_seconds=0.08
    )
    try:
        with caplog.at_level(logging.WARNING, logger="gateway.slow_loop_monitor"):
            _idle_wait_for_test()
    finally:
        assert monitor.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert not any("occupied the event loop" in m for m in messages)  # 期望: 无报告 — 空闲 select 等待(0.4s>阈值)非占用
    assert not any("select" == _topmost_name(m) and "selectors" in m for m in messages if "occupied" in m)  # 期望: 兜底 — 任何报告不得以 select 空闲为据


def _topmost_name(message: str) -> str:
    return message.strip().splitlines()[-1].strip() if message else ""
