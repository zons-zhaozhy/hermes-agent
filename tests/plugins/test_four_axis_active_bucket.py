"""压缩后会话 id 漂移时四轴副防线活跃桶回退回归。

根因（2026-10-27 实测）：
  上下文压缩后调度面对写端(pre_tool_batch)传新会话 id、对读端(pre_tool_call)
  仍传旧 id——写端把四轴 marker 落在新会话桶，读端按传入旧 id 找不到 →
  四轴证据贴三轮仍被拦。修复=four_axis.py on_pre_tool_call 在传入桶未命中
  marker 时回退 ``_ACTIVE_SESSION_ID`` 活跃桶再读。本文件锁定：
  回退命中放行、两桶皆无照拦（fail-closed 不弱化）、过期活跃桶不借用。
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
    """按 test_four_axis_empty_session.py 同款 file-location 方式加载副防线模块。"""
    path = Path(__file__).resolve().parents[2] / "plugins" / "guards" / "four_axis.py"
    spec = importlib.util.spec_from_file_location("four_axis_active_bucket_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 同仓同存必然可加载
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


four_axis = _load_four_axis()


@pytest.fixture(autouse=True)
def _clean_gate_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 gate 注册表与活跃会话指针，测试间零泄漏。"""
    monkeypatch.setattr(rtg, "_GATES", {})
    # _ACTIVE_SESSION_ID 已 ContextVar 化——测试用全新 ContextVar 整体替换
    # 模块属性；on_session_start 的 .set() 作用于替换后的变量，零泄漏。
    monkeypatch.setattr(
        rtg, "_ACTIVE_SESSION_ID", contextvars.ContextVar("test_sid", default=None),
    )
    yield


def _make_marker(session_id: str, age_seconds: float = 0.0) -> None:
    path = _four_axis_marker_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "verified": True,
        "source": "content_scan",
        "timestamp": time.time() - age_seconds,
        "axes": ["影响面", "原意图", "根因", "风险"],
    }), encoding="utf-8")


def test_reader_falls_back_to_active_bucket_on_id_drift() -> None:
    """传入旧 id 桶无 marker + 活跃桶（写端刚落盘）verified → 放行。

    压缩漂移形态：pre_tool_call 收旧 id，写端 marker 在新 id 桶。
    """
    rtg.on_session_start("local-new-id")
    _make_marker("local-new-id")
    verdict: Optional[dict[str, Any]] = four_axis.on_pre_tool_call(
        tool_name="patch",
        args={"path": "/tmp/project/src/a.py"},
        session_id="local-stale-id",
    )
    assert verdict is None  # 期望: None,依据=活跃桶verified四轴齐即验证过(修前读旧桶恒拦)


def test_reader_still_blocks_when_both_buckets_missing() -> None:
    """负例：传入桶与活跃桶皆无 marker → 照拦（回退不弱化 fail-closed）。"""
    rtg.on_session_start("local-new-id-2")
    verdict = four_axis.on_pre_tool_call(
        tool_name="patch",
        args={"path": "/tmp/project/src/b.py"},
        session_id="local-stale-id-2",
    )
    assert verdict is not None and verdict.get("action") == "block"  # 期望: 拦,依据=两桶皆无验证记录


def test_reader_ignores_expired_active_bucket() -> None:
    """负例：活跃桶 marker 过期(TTL 外) → 不借用,照拦。"""
    rtg.on_session_start("local-new-id-3")
    _make_marker("local-new-id-3", age_seconds=7200.0)
    verdict = four_axis.on_pre_tool_call(
        tool_name="patch",
        args={"path": "/tmp/project/src/c.py"},
        session_id="local-stale-id-3",
    )
    assert verdict is not None  # 期望: 拦,依据=过期marker非有效验证,回退不得绕过TTL
