"""Tests for R6 expected-annotation rule (科学编程守卫·测试纪律层).

行为契约（从「期望值独立推导，禁凑绿灯」独立推导，非实现反推）：
- tests/ 路径的断言字面量无「# 期望:」注释 → 违规消息含行号
- 有「# 期望:」注释 → 合规
- truthiness 断言(assert obj is not None) → 豁免（非字面量比较）
- assert x is None → 字面量，无注释 → 违规
- 非 tests/ 路径 → 该规则不启用
- on_pre_tool_call 集成：tests/ 文件裸断言 → block
"""
import ast
import importlib.util
from pathlib import Path

PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "scientific-programming-guard" / "__init__.py"
spec = importlib.util.spec_from_file_location("scientific_programming_guard_r6", PLUGIN_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _issues(code: str) -> list:
    tree = ast.parse(code)
    return mod._check_expected_annotation(tree, code.splitlines())


def test_bare_literal_assert_flagged():
    code = 'def t():\n    assert len(x) == 1\n'
    out = _issues(code)
    assert len(out) == 1 and "line 2" in out[0]  # 期望: 裸字面量断言报行号


def test_annotated_literal_assert_passes():
    code = 'def t():\n    assert len(x) == 2  # 期望: 两个元素由fixture推入\n'
    assert _issues(code) == []  # 期望: 有推导注释即合规


def test_truthiness_assert_exempt():
    code = 'def t():\n    assert out is not None\n'
    assert _issues(code) == []  # 期望: is not None 非字面量比较豁免


def test_none_literal_assert_flagged():
    code = 'def t():\n    assert get() is None\n'
    assert len(_issues(code)) == 1  # 期望: None 属字面量断言需注释


def test_string_literal_flagged():
    code = 'def t():\n    assert msg == "ok"\n'
    assert len(_issues(code)) == 1  # 期望: 字符串字面量需注释


def test_container_literal_flagged():
    code = 'def t():\n    assert rows == [1, 2]\n'
    assert len(_issues(code)) == 1  # 期望: 容器字面量需注释


def test_var_to_var_compare_exempt():
    code = 'def t():\n    assert got == want\n'
    assert _issues(code) == []  # 期望: 变量对变量比较豁免(期望值来自推导变量)


def test_integration_blocks_bare_assert_in_tests():
    args = {
        "path": "tests/foo/test_x.py",
        "content": "def test_a():\n    assert calc() == 42\n",
    }
    out = mod.on_pre_tool_call(tool_name="write_file", args=args)
    assert out.get("action") == "block"  # 期望: tests 路径裸断言被拦


def test_integration_passes_non_tests_path():
    args = {
        "path": "plugins/foo.py",
        "content": "def f(x: int) -> int:\n    assert x == 42\n    return x\n",
    }
    assert mod.on_pre_tool_call(tool_name="write_file", args=args) == {}  # 期望: 非tests路径R6不启用(R3/R4合规)
