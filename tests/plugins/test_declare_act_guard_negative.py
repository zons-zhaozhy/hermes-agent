"""declare_act_guard 负例契约测试——宣告-动手配对守卫的注入闸必须真拦（违规必红）。

Contract:
    Preconditions: plugins/declare_act_guard/__init__.py 存在
    Postconditions: 超封顶/同文本重放被注入闸拒绝；违规注入文本含配对纪律要求
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "declare_act_guard" / "__init__.py"
    spec = importlib.util.spec_from_file_location("declare_act_guard_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_gate_blocks_over_cap():
    guard = _load_guard()
    state = {"injected_count": guard._MAX_INJECTS, "last_text": ""}
    assert guard._gate_ok(state, "some text") is False  # 期望: False=超封顶注入闸拦截


def test_negative_gate_blocks_same_text_replay():
    guard = _load_guard()
    state = {"injected_count": 0, "last_text": "repeat me"}
    assert guard._gate_ok(state, "repeat me") is False  # 期望: False=同文本重放拦截


def test_negative_injection_carries_pairing_rule():
    guard = _load_guard()
    out = guard._build_injection()
    assert "禁止裸宣告" in out.get("context", "")  # 期望: 注入文本含配对纪律（拦截语义）
    assert "必须先确认" in out.get("context", "")  # 期望: 注入文本给出须拍板出路


def test_negative_gate_allows_fresh_text_under_cap():
    guard = _load_guard()
    state = {"injected_count": 0, "last_text": "old"}
    assert guard._gate_ok(state, "brand new text") is True  # 期望: True=未超限新文本放行
