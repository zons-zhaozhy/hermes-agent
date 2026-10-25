"""rules_selfcheck 负例契约测试——规则自测脚本必须全数通过（违规必红）。

Contract:
    Preconditions: plugins/guards/rules_selfcheck.py 存在（加载即执行内嵌用例）
    Postconditions: 内嵌正反用例零失败（FAIL_COUNT==0）且确有覆盖（PASS_COUNT>0）
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "guards" / "rules_selfcheck.py"
    spec = importlib.util.spec_from_file_location("rules_selfcheck_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rules_selfcheck_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_negative_all_embedded_cases_pass():
    guard = _load_guard()
    assert guard.FAIL_COUNT == 0  # 期望: 0=全部内嵌违规用例被 coding_standards 正确检出（零漏检）
    assert guard.PASS_COUNT > 0  # 期望: 正数=用例集非空（空集自测=无效自证）
