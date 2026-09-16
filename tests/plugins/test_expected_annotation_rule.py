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


# ── R5a: import 本地模块前必须读过 ──────────────────────────────────────────

def test_r5a_flags_unread_local_import():
    """tests/ 文件 import 仓库内真实模块(如 tools.file_fingerprint), 本测试
    进程从未 read_file 过它 → R5a 报违规。"""
    code = "from tools.file_fingerprint import content_fingerprint\n\n\ndef test_x():\n    assert content_fingerprint('a') == 'x'  # 期望: 推导注释存在, 只验R5a\n"
    args = {"path": "tests/foo/test_r5a.py", "content": code}
    out = mod.on_pre_tool_call(tool_name="write_file", args=args)
    assert out.get("action") == "block"  # 期望: 未读过的本地模块import被拦
    assert any("read_file" in m for m in out["message"].splitlines()[1:])  # 期望: 消息点名read_file


def test_r5a_passes_stdlib_import():
    """stdlib/第三方 import 解析不到仓库内 .py → 放行。"""
    code = "import json\nimport os\n\n\ndef test_x():\n    assert json.dumps({}) == '{}'  # 期望: json.dumps空dict标准输出\n"
    args = {"path": "tests/foo/test_r5a_std.py", "content": code}
    out = mod.on_pre_tool_call(tool_name="write_file", args=args)
    assert out == {}  # 期望: 非本地模块零违规


def test_r5a_passes_after_session_read(monkeypatch):
    """同一文件, read_history 有记录 → R5a 放行。直接向 _read_tracker 注入
    记录, 验证判定逻辑(不依赖真实 read_file 调用链)。"""
    code = "from tools.file_fingerprint import content_fingerprint\n\n\ndef test_x():\n    assert content_fingerprint('a') == 'x'  # 期望: 推导注释存在, 只验R5a\n"
    from tools.file_tools_read_tracking import _read_tracker, _task_data
    with __import__("tools.file_tools_read_tracking", fromlist=["_read_tracker_lock"])._read_tracker_lock:
        _task_data("r5a-probe")["read_history"].add("tools/file_fingerprint.py")
    try:
        args = {"path": "tests/foo/test_r5a_read.py", "content": code}
        out = mod.on_pre_tool_call(tool_name="write_file", args=args)
        assert out == {}  # 期望: 读记录存在后import放行
    finally:
        with __import__("tools.file_tools_read_tracking", fromlist=["_read_tracker_lock"])._read_tracker_lock:
            _read_tracker.pop("r5a-probe", None)


# ── R5b: 关键字参数必须存在于目标签名 ──────────────────────────────────────

def test_r5b_flags_invented_kwarg():
    """跨模块调用传了签名不存在的键 → 违规。目标=本仓库真实模块真实函数,
    传入键 record_read 签名里没有(精确复刻 true_key 案)。先注入读记录
    满足 R5a, 独立验证 R5b。"""
    from tools.file_tools_read_tracking import _read_tracker, _task_data, _read_tracker_lock
    with _read_tracker_lock:
        _task_data("r5b-probe")["read_history"].add("tools/file_fingerprint.py")
    try:
        code = ("from tools.file_fingerprint import record_read\n"
                "\n"
                "\n"
                "def test_x():\n"
                "    record_read('t', '/x.py', 'aaa', true_key='review')  # 期望: true_key非签名键, 只验R5b拦\n")
        args = {"path": "tests/foo/test_r5b.py", "content": code}
        out = mod.on_pre_tool_call(tool_name="write_file", args=args)
        assert out.get("action") == "block"  # 期望: 瞎编关键字参数被拦
        assert any("true_key" in m for m in out["message"].splitlines()[1:])  # 期望: 消息点名坏参数
    finally:
        with _read_tracker_lock:
            _read_tracker.pop("r5b-probe", None)


def test_r5b_passes_real_kwargs():
    """传真实存在的键(record_read(task_id, resolved_path, fingerprint)) → 放行。
    注入读记录满足 R5a 后独立验证 R5b。"""
    from tools.file_tools_read_tracking import _read_tracker, _task_data, _read_tracker_lock
    with _read_tracker_lock:
        _task_data("r5b-ok")["read_history"].add("tools/file_fingerprint.py")
    try:
        code = ("from tools.file_fingerprint import record_read\n"
                "\n"
                "\n"
                "def test_x():\n"
                "    record_read(task_id='t', resolved_path='/x.py', fingerprint='aaa')  # 期望: 三键皆真实签名键\n")
        args = {"path": "tests/foo/test_r5b_ok.py", "content": code}
        out = mod.on_pre_tool_call(tool_name="write_file", args=args)
        assert out == {}  # 期望: 真实参数零违规
    finally:
        with _read_tracker_lock:
            _read_tracker.pop("r5b-ok", None)
