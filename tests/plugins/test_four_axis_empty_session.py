"""空 session_id 时四轴双防线的桶一致性回归。

根因（2026-10-27 实测审计）：
  ``read_think_gate/__init__.py::pre_tool_batch`` 用裸 session_id 解析 gate，
  调度面传空时 gate 返回 None → 主防线不扫描 → marker 永不写入；
  ``plugins/guards/four_axis.py`` 读端按空桶（nosession）读 marker → 永远缺失
  → 副防线永久拦截且模型重贴四轴证据无效（four_axis.py 源码注释自认的
  残留形态）。本文件锁定：写读两端经同一 effective_session_id 回退解析，
  空 session_id 不再产生「主防线不扫描 + 副防线照拦」的死锁形态。
"""

from __future__ import annotations

import contextvars
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import pytest

from plugins import read_think_gate as rtg
from plugins.read_think_gate_host import _four_axis_marker_path


def _load_four_axis() -> ModuleType:
    """按 test_four_axis_gate_presence.py 同款 file-location 方式加载副防线模块。"""
    path = Path(__file__).resolve().parents[2] / "plugins" / "guards" / "four_axis.py"
    spec = importlib.util.spec_from_file_location("four_axis_empty_session_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 同仓同存必然可加载
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


four_axis = _load_four_axis()


@pytest.fixture(autouse=True)
def _clean_gate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """隔离 gate 注册表与活跃会话指针，测试间零泄漏。"""
    monkeypatch.setattr(rtg, "_GATES", {})
    # _ACTIVE_SESSION_ID 已 ContextVar 化（多会话隔离）——测试用全新
    # ContextVar(default=None) 整体替换模块属性，等价于「无活跃会话」。
    monkeypatch.setattr(
        rtg, "_ACTIVE_SESSION_ID", contextvars.ContextVar("test_sid", default=None),
    )
    yield


def _make_marker(session_id: str, axes: Optional[list[str]] = None) -> None:
    path = _four_axis_marker_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "verified": True,
        "source": "content_scan",
        "timestamp": time.time(),
        "axes": axes or ["影响面", "原意图", "根因", "风险"],
    }), encoding="utf-8")


def test_pre_tool_batch_empty_session_falls_back_to_active_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """空 session_id + 活跃会话在场 → 主防线仍须扫描（check_batch 被调用）。"""
    rtg.on_session_start("local-active-1")
    gate = rtg._gate_for("local-active-1", "cli")
    calls: list[tuple] = []
    orig = gate.check_batch

    def _spy(content: str, names: list, tool_args: Optional[list] = None) -> Optional[dict]:
        calls.append((content, names))
        return orig(content, names, tool_args=tool_args)

    monkeypatch.setattr(gate, "check_batch", _spy)
    rtg.pre_tool_batch(
        assistant_content="影响面 原意图 根因 风险",
        tool_calls=[{"name": "read_file", "args": {}}],
        session_id="",
        platform="cli",
    )
    assert len(calls) == 1  # 期望: 1次,依据=空session回退活跃gate后check_batch恰好被调用一次(修前0次)


def test_four_axis_reader_empty_session_reads_active_session_bucket() -> None:
    """写端在活跃会话桶写 marker → 读端空 session_id 读同一桶并放行。"""
    rtg.on_session_start("local-active-2")
    _make_marker("local-active-2")
    verdict = four_axis.on_pre_tool_call(
        tool_name="patch",
        args={"path": "/tmp/project/src/a.py"},
        session_id="",
    )
    assert verdict is None  # 期望: None,依据=verified=True四轴齐marker命中即放行(修前读nosession桶恒拦)


def test_repeat_block_within_window_escalates() -> None:
    """同 session+工具+目标 60 秒内二次拦截 → 消息追加升级提示（原样重试无效）。"""
    rtg.on_session_start("local-active-3")
    args = {"path": "/tmp/project/src/b.py"}
    first = four_axis.on_pre_tool_call(tool_name="patch", args=args, session_id="")
    second = four_axis.on_pre_tool_call(tool_name="patch", args=args, session_id="")
    assert first is not None and second is not None  # 期望: 两判都拦,依据=marker缺失(fail-closed)
    assert "已被拦截两次" in second["message"]  # 期望: 命中,依据=60秒窗口内同目标重复拦截触发升级
    assert "已被拦截两次" not in first["message"]  # 期望: 不命中,依据=首次拦截无前科不算重复


def test_block_log_carries_session_for_attribution(caplog: pytest.LogCaptureFixture) -> None:
    """拦截日志必须带 session 归属字段——2026-10-27 审计须查 state.db 才能归因。"""
    rtg.on_session_start("local-active-4")
    with caplog.at_level("INFO", logger="plugins.guards.four_axis"):
        four_axis.on_pre_tool_call(
            tool_name="patch", args={"path": "/tmp/project/src/c.py"}, session_id="",
        )
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "session=local-active-4" in joined  # 期望: 命中,依据=拦截日志必须携带effective会话id供审计归属
