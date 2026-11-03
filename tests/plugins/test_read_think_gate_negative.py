"""read_think_gate 负例契约测试——复杂度 fallback 分类必须抓复杂任务（违规必红）。

Contract:
    Preconditions: plugins/read_think_gate/gate.py 存在
    Postconditions: 复杂任务词命中返回 complex；简单任务返回 simple/normal
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "read_think_gate" / "gate.py"
    spec = importlib.util.spec_from_file_location("read_think_gate_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    sys.modules["read_think_gate_under_test"] = mod  # dataclass 装饰器需从 sys.modules 取命名空间
    spec.loader.exec_module(mod)
    return mod


def test_negative_complex_task_detected():
    guard = _load_guard()
    out = guard._fallback_detect("重构整个模块的架构，从零搭建新体系")
    assert out == "complex"  # 期望: complex=复杂任务必须触发 ReadThink 调查门槛（漏放=裸动手）


def test_negative_simple_task_not_escalated():
    guard = _load_guard()
    out = guard._fallback_detect("改一下这个 typo")
    assert out in ("simple", "normal")  # 期望: 非复杂=简单任务不设调查门槛


def test_sql_numeric_comparison_not_file_write():
    """反方审查 D2：SQL 比较符后的数字字面量不是重定向写目标。

    独立期望来源：`WHERE price > 9.99` 中 9.99 是数值字面量；
    2026-11-03 前含 "." 即判写导致只读 SQL 取证被拦。
    """
    guard = _load_guard()
    assert guard._terminal_writes_file('sqlite3 db "SELECT 1 FROM t WHERE price > 9.99"') is False  # 期望: 数字字面量不计写
    assert guard._terminal_writes_file("echo hi > out.txt") is True  # 期望: 真重定向到文件仍计写
    assert guard._terminal_writes_file("psql -c 'SELECT count(*) FROM t WHERE amt > 100.5'") is False  # 期望: psql 数字比较同形态不计写


def test_negative_ordinary_message_normal():
    guard = _load_guard()
    assert guard._fallback_detect("现在几点了") == "normal"  # 期望: normal=常规查询默认档
