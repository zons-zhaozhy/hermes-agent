"""patch-first 硬拦截行为测试。

期望值独立推导（先于实现运行）：
  T1 sed -i 写入        → block
  T2 sed -n 只读        → 放行(None)
  T3 非 terminal 工具    → 放行(None)
  T4 普通命令           → 放行(None)
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(HERE, "..", "..", "plugins", "patch-first", "__init__.py")

spec = importlib.util.spec_from_file_location("patch_first_under_test", PLUGIN)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def run(tool, arg):
    return mod._on_pre_tool_call(tool_name=tool, args={"command": arg})


def test_t1_sed_write_blocked():
    r = run("terminal", "sed -i 's/a/b/' sourcefile")
    assert r is not None and r.get("action") == "block", r


def test_t2_sed_readonly_allowed():
    assert run("terminal", "sed -n '10,20p' sourcefile") is None


def test_t3_non_terminal_allowed():
    assert run("execute_code", "sed -i 's/a/b/' sourcefile") is None


def test_t4_plain_command_allowed():
    assert run("terminal", "ls -la") is None
