"""four_axis 负例契约测试——四轴副防线必须在 marker 缺失时拦写（违规必红）。

Contract:
    Preconditions: plugins/guards/four_axis.py 存在
    Postconditions: 非豁免路径写工具+marker 缺失返回 block；marker 有效放行
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "guards" / "four_axis.py"
    spec = importlib.util.spec_from_file_location("four_axis_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_write_blocked_without_marker(monkeypatch, tmp_path: Path):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda: None)
    target = str(tmp_path / "victim.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="sess-fa-1")
    assert out is not None and out.get("action") == "block"  # 期望: block=无四轴证据写操作必拦


def test_negative_write_allowed_with_valid_marker(monkeypatch, tmp_path: Path):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda: {"axes": 4, "ts": 0})
    target = str(tmp_path / "ok.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="sess-fa-2")
    assert out is None  # 期望: None=四轴 marker 有效放行


def test_negative_cron_session_exempt(monkeypatch, tmp_path: Path):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda: None)
    target = str(tmp_path / "cron.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="cron_job9_20261025")
    assert out is None  # 期望: None=cron 会话豁免（主闸门同样豁免，防结构性死锁）


def test_negative_read_tool_not_gated(monkeypatch):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda: None)
    out = guard.on_pre_tool_call(tool_name="read_file", args={"path": "x"},
                                 session_id="sess-fa-3")
    assert out is None  # 期望: None=读工具不经四轴闸门
