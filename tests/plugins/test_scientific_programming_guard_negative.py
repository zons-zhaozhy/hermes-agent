"""scientific_programming_guard 负例契约测试——科学编程守卫 AST 检查必须真抓违规（违规必红）。

Contract:
    Preconditions: plugins/scientific-programming-guard/__init__.py 存在
    Postconditions: 复杂度超限/缺类型注解/裸 finally close 均产生非空违规列表
"""
import ast
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "scientific-programming-guard" / "__init__.py"
    spec = importlib.util.spec_from_file_location("sci_prog_guard_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_complexity_budget_violation_detected():
    guard = _load_guard()
    code = (
        "def f(a):\n"
        + "".join(f"    if a > {i}: a += {i}\n" for i in range(25))
        + "    return a\n"
    )
    out = guard._check_complexity_budget(ast.parse(code))
    assert len(out) > 0  # 期望: 非空=圈复杂度超预算被拦


def test_negative_missing_type_annotations_detected():
    guard = _load_guard()
    out = guard._check_type_annotations(ast.parse("def add(a, b):\n    return a + b\n"))
    assert len(out) > 0  # 期望: 非空=新函数缺类型注解被拦


def test_negative_unguarded_finally_close_detected():
    guard = _load_guard()
    code = (
        "def f(path):\n"
        "    fh = open(path)\n"
        "    try:\n"
        "        return fh.read()\n"
        "    finally:\n"
        "        fh.close()\n"
    )
    out = guard._check_l3_guarded_cleanup(ast.parse(code))
    assert len(out) > 0  # 期望: 非空=finally 裸 close 未包 try/except 被拦


def test_negative_clean_code_passes_checks():
    guard = _load_guard()
    clean = (
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
    )
    tree = ast.parse(clean)
    assert guard._check_type_annotations(tree) == []  # 期望: 空=带注解函数放行
    assert guard._check_complexity_budget(tree) == []  # 期望: 空=低复杂度放行
