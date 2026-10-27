"""four_axis 主防线在场探测契约测试——三态判定与死锁防护（违规必红）。

背景（实测）：2026-10-26~27 hongtai-aml 网关主防线插件 read-think-gate
未注册，副防线按 marker 缺失拦截全部写操作约 19 小时。本文件锁定死锁
防护语义：确认缺席放行+每会话一次告警；探测异常与无 ctx 保持拦截
（unknown 不得折叠为 absent）；marker 有效时不触碰探测。

Contract:
    Preconditions: plugins/guards/four_axis.py 存在
    Postconditions: active/absent/unknown 三态行为逐一可复现
"""
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def _load_guard() -> Any:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "guards" / "four_axis.py"
    spec = importlib.util.spec_from_file_location("four_axis_presence_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(has_plugin: Callable[[str], bool]) -> SimpleNamespace:
    return SimpleNamespace(has_plugin=has_plugin)


def _absence_warnings(caplog: Any) -> int:
    return len([r for r in caplog.records if "absent/disabled" in r.getMessage()])


def test_present_and_no_marker_blocks(monkeypatch, tmp_path: Path):
    guard = _load_guard()
    guard._gate_ctx = _ctx(lambda pid: True)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    target = str(tmp_path / "victim.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="sess-gp-1")
    assert out is not None and out.get("action") == "block"  # 期望: 主防线在场+marker 缺失必拦（原契约不变）


def test_absent_allows_and_warns_once_per_session(monkeypatch, tmp_path: Path, caplog):
    guard = _load_guard()
    guard._gate_ctx = _ctx(lambda pid: False)
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    target = str(tmp_path / "deadlock.py")
    with caplog.at_level(logging.DEBUG):
        out1 = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                      session_id="sess-gp-2")
        out2 = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                      session_id="sess-gp-2")
        out3 = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                      session_id="sess-gp-3")
    assert out1 is None and out2 is None and out3 is None  # 期望: 确认缺席放行（死锁防护）
    assert _absence_warnings(caplog) == 2  # 期望: 每会话恰一次告警（gp-2 与 gp-3 各一条）


def test_config_disabled_allows(monkeypatch, tmp_path: Path, caplog):
    guard = _load_guard()
    guard._gate_ctx = _ctx(lambda pid: True)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda: {"read_think_gate": {"enabled": False}})
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    target = str(tmp_path / "cfgoff.py")
    with caplog.at_level(logging.WARNING):
        out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                     session_id="sess-gp-4")
    assert out is None  # 期望: config 段显式 enabled=false=主防线不写 marker=放行
    assert _absence_warnings(caplog) == 1  # 期望: 放行必须留痕告警


def test_probe_exception_blocks_with_error(monkeypatch, tmp_path: Path, caplog):
    def _boom(pid: str) -> bool:
        raise RuntimeError("registry broken")

    guard = _load_guard()
    guard._gate_ctx = _ctx(_boom)
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    target = str(tmp_path / "probeerr.py")
    with caplog.at_level(logging.ERROR):
        out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                     session_id="sess-gp-5")
    assert out is not None and out.get("action") == "block"  # 期望: 探测异常=unknown 态保持拦截（不得折叠为放行）
    assert any("unknown, keep blocking" in r.getMessage() for r in caplog.records)  # 期望: unknown 态必须 ERROR 留痕


def test_no_ctx_blocks_with_error(monkeypatch, tmp_path: Path, caplog):
    guard = _load_guard()  # 未 register：_gate_ctx 保持模块默认 None
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    target = str(tmp_path / "noctx.py")
    with caplog.at_level(logging.ERROR):
        out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                     session_id="sess-gp-6")
    assert out is not None and out.get("action") == "block"  # 期望: 无 ctx=探测不可用=保持拦截
    assert any("no ctx" in r.getMessage() for r in caplog.records)  # 期望: ERROR 留痕


def test_valid_marker_skips_probe(monkeypatch, tmp_path: Path):
    def _boom(pid: str) -> bool:
        raise AssertionError("probe must not run when marker is valid")

    guard = _load_guard()
    guard._gate_ctx = _ctx(_boom)
    monkeypatch.setattr(guard, "_read_marker",
                        lambda sid: {"verified": True, "ts": 0, "axes": 4})
    target = str(tmp_path / "okmark.py")
    out = guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                                 session_id="sess-gp-7")
    assert out is None  # 期望: marker 有效路径不触碰探测（顺序契约）


def test_recovery_rewarns_after_reactivation(monkeypatch, tmp_path: Path, caplog):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_marker", lambda sid: None)
    target = str(tmp_path / "rewarn.py")
    guard._gate_ctx = _ctx(lambda pid: False)
    with caplog.at_level(logging.DEBUG):
        guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                               session_id="sess-gp-8")  # 缺席→告警#1
    guard._gate_ctx = _ctx(lambda pid: True)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                           session_id="sess-gp-8")  # 在场→拦截+重置告警态
    guard._gate_ctx = _ctx(lambda pid: False)
    with caplog.at_level(logging.DEBUG):
        guard.on_pre_tool_call(tool_name="patch", args={"path": target},
                               session_id="sess-gp-8")  # 再缺席→告警#2
    assert _absence_warnings(caplog) == 2  # 期望: 恢复在场即重置——再次缺席必须重新告警


def test_gate_plugin_id_matches_manifest_name():
    repo = Path(__file__).resolve().parents[2]
    plugin_yaml = repo / "plugins" / "read_think_gate" / "plugin.yaml"
    text = plugin_yaml.read_text(encoding="utf-8")
    guard = _load_guard()
    assert f'name: {guard._GATE_PLUGIN_ID}' in text  # 期望: 探测目标名与主防线 plugin.yaml 声明名零漂移
