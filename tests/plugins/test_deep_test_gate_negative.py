"""deep_test_gate 负例契约测试——无证据的深度测试完成声明必须被拦（违规必红）。

Contract:
    Preconditions: plugins/deep-test-gate/__init__.py 存在
    Postconditions: 强声明+无证据+pytest 项目+无账本时 _should_block 为真
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "deep-test-gate" / "__init__.py"
    spec = importlib.util.spec_from_file_location("deep_test_gate_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_project(tmp_path: Path) -> str:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    src = tmp_path / "src" / "mod.py"
    src.parent.mkdir()
    src.write_text("x = 1\n", encoding="utf-8")
    return str(src)


def test_negative_bare_deep_claim_blocked(tmp_path: Path):
    guard = _load_guard()
    changed = [_make_project(tmp_path)]
    assert guard._should_block("深度测试已全部通过", changed) is True  # 期望: True=裸强声明被拦


def test_negative_claim_with_evidence_passes(tmp_path: Path):
    guard = _load_guard()
    changed = [_make_project(tmp_path)]
    resp = "深度测试已全部通过\n[deep-test-matrix] 全层覆盖 PASS"
    assert guard._should_block(resp, changed) is False  # 期望: False=三门禁证据放行


def test_negative_claim_with_exemption_passes(tmp_path: Path):
    guard = _load_guard()
    changed = [_make_project(tmp_path)]
    assert guard._should_block("深度测试完成——非 pytest 项目", changed) is False  # 期望: False=显式豁免放行


def test_negative_no_claim_passes(tmp_path: Path):
    guard = _load_guard()
    changed = [_make_project(tmp_path)]
    assert guard._should_block("改完了这个函数", changed) is False  # 期望: False=无强声明不触发
