"""coding_standards 负例契约测试——编码规范 AST 检查必须真抓违规（违规必红）。

Contract:
    Preconditions: plugins/guards/coding_standards.py 存在
    Postconditions: 吞异常/裸 except/静默降级均产生对应 rule_id 违规
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "guards" / "coding_standards.py"
    spec = importlib.util.spec_from_file_location("coding_standards_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rule_ids(guard: ModuleType, code: str) -> set:
    return {v.rule_id for v in guard._run_all_checks(code)}


def test_negative_bare_except_pass_detected():
    guard = _load_guard()
    rules = _rule_ids(guard, "try:\n    x = 1\nexcept:\n    pass\n")
    assert "R002" in rules  # 期望: 裸 except:pass 被拦（吞异常检测）


def test_negative_silent_downgrade_detected():
    guard = _load_guard()
    rules = _rule_ids(
        guard,
        "def f():\n"
        "    try:\n"
        "        return compute()\n"
        "    except Exception:\n"
        "        return []\n",
    )
    assert len(rules) > 0  # 期望: 静默降级（异常吞掉返回默认值）被拦


def test_negative_clean_code_passes():
    guard = _load_guard()
    rules = _rule_ids(
        guard,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )
    assert rules == set()  # 期望: 空集=干净代码零违规
