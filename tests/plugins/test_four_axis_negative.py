"""four_axis 负例契约测试——四轴副防线必须在 marker 缺失时拦写（违规必红）。

Contract:
    Preconditions: plugins/guards/four_axis.py 存在
    Postconditions: 非豁免路径写工具+marker 缺失返回 block；marker 有效放行
"""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "guards" / "four_axis.py"
    spec = importlib.util.spec_from_file_location("four_axis_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 本文件主题=marker 契约负例，前提固定为主防线在场（探测三态契约见
    # test_four_axis_gate_presence.py）。config 读取在测试临时 home 下返回
    # 空节（enabled 默认 True=active），无环境漂移。
    guard: Any = mod
    guard._gate_ctx = SimpleNamespace(has_plugin=lambda pid: True)
    return mod


def test_negative_write_blocked_without_marker(monkeypatch, tmp_path: Path):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    target = str(tmp_path / "victim.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="sess-fa-1")
    assert out is not None and out.get("action") == "block"  # 期望: block=无四轴证据写操作必拦


def test_negative_write_allowed_with_valid_marker(monkeypatch, tmp_path: Path):
    guard = _load_guard()
    seen = []
    monkeypatch.setattr(guard, "_read_marker", lambda sid: seen.append(sid) or {"axes": 4, "ts": 0})
    target = str(tmp_path / "ok.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="sess-fa-2")
    assert out is None  # 期望: None=四轴 marker 有效放行
    assert seen == ["sess-fa-2"]  # 期望: 读端以本会话 ID 查自己的 marker 桶


def test_negative_cron_session_exempt(monkeypatch, tmp_path: Path):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    target = str(tmp_path / "cron.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="cron_job9_20261025")
    assert out is None  # 期望: None=cron 会话豁免（主闸门同样豁免，防结构性死锁）


def test_negative_read_tool_not_gated(monkeypatch):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    out = guard.on_pre_tool_call(tool_name="read_file", args={"path": "x"},
                                 session_id="sess-fa-3")
    assert out is None  # 期望: None=读工具不经四轴闸门


def test_partial_progress_blocked_with_missing_axes_diagnosis(monkeypatch, tmp_path: Path):
    """进度 marker（verified=False）不改变放行语义，拦截消息须诊断缺失轴。"""
    import time as _time

    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)  # 放行判定恒不过
    monkeypatch.setattr(guard, "_read_progress", lambda sid: {
        "verified": False, "axes": ["影响面", "根因", "风险"],
        "timestamp": _time.time(),
    })
    target = str(tmp_path / "partial.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="sess-fa-4")
    assert out is not None and out.get("action") == "block"  # 期望: 3/4 轴仍拦——放行语义零变化
    msg = out["message"]
    assert "原意图" in msg and "影响面" not in msg.split("——")[-1]  # 期望: 只诊断缺失轴（原意图），已命中的轴不进诊断段
    assert "还缺 1 轴" in msg  # 期望: 缺轴计数


def test_no_progress_falls_back_to_plain_block(monkeypatch, tmp_path: Path):
    """无进度记录时拦截消息保持原四项引导（无诊断段）。"""
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    monkeypatch.setattr(guard, "_read_progress", lambda sid: None)
    target = str(tmp_path / "cold.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="sess-fa-5")
    assert out is not None and "诊断" not in out["message"]  # 期望: 冷启动无诊断段
