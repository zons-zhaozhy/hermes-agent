"""scientific-programming-guard E2E 测试 v2——期望值独立推导(先写期望再验证)。

期望值来自科学编程律的语义, 非实现反推:
  - 律11: close 先于哨兵 put → 必拦; finally 裸 close → 必拦;
          哨兵先行+守卫 close → 放行
  - 律3:  cc>10 或 >50 行 → 必拦; 简单函数 → 放行
  - 律4:  参数/返回值缺注解 → 必拦; 注解齐全 → 放行
  - 非网络代码不受 R1/R2 约束; 非 .py 不扫
"""
import importlib.util
from pathlib import Path

PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "scientific-programming-guard" / "__init__.py"
spec = importlib.util.spec_from_file_location("scientific_programming_guard", PLUGIN_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _run(path, content):
    return mod.on_pre_tool_call(tool_name="write_file",
                                args={"path": path, "content": content})


def test_r1_close_before_sentinel_blocked():
    """close 排在哨兵 put 之前 → 必拦(期望来自律11 并发正确性)"""
    bad = """
import oracledb
def worker(q, conn):
    try:
        rows = conn.query("SELECT 1")
        q.put(rows)
    finally:
        conn.close()
        q.put(None)
"""
    r = _run("/proj/x.py", bad)
    assert r.get("action") == "block", f"应拦截, got {r}"


def test_r2_unguarded_close_in_finally_blocked():
    """finally 中裸 close 无 try 包裹 → 必拦(期望来自律11)"""
    bad = """
import oracledb
def f(conn):
    try:
        return conn.query("SELECT 1")
    finally:
        conn.close()
"""
    r = _run("/proj/x.py", bad)
    assert r.get("action") == "block", f"应拦截, got {r}"


def test_guarded_close_after_sentinel_pass():
    """哨兵先行 + close 有守卫 + 注解齐全 → 放行(合规形态)"""
    good = """
import queue
import oracledb
def worker(q: "queue.Queue", conn: "oracledb.Connection") -> list:
    try:
        rows = conn.query("SELECT 1")
        q.put(rows)
        return rows
    finally:
        q.put(None)
        try:
            conn.close()
        except Exception:
            pass
"""
    r = _run("/proj/x.py", good)
    assert r == {}, f"应放行, got {r}"


def test_non_python_pass():
    """非 .py 不扫 → 放行"""
    r = _run("/proj/x.yaml", "finally: conn.close()")
    assert r == {}


def test_r3_complexity_blocked():
    """圈复杂度超预算(>10 分支) → 必拦(期望来自律3)"""
    branches = "\n".join(
        f"    if x == {i}:\n        y += 1" for i in range(12)
    )
    bad = f"""
def calc(x: int) -> int:
    y = 0
{branches}
    return y
"""
    r = _run("/proj/x.py", bad)
    assert r.get("action") == "block", f"应拦截, got {r}"
    assert "复杂度" in r.get("message", "")


def test_r3_simple_function_pass():
    """注解齐全的简单函数 → 放行"""
    good = """
def add(a: int, b: int) -> int:
    return a + b
"""
    r = _run("/proj/x.py", good)
    assert r == {}, f"应放行, got {r}"


def test_r4_missing_annotations_blocked():
    """参数/返回值缺类型注解 → 必拦(期望来自律4)"""
    bad = """
def greet(name):
    return "hi " + name
"""
    r = _run("/proj/x.py", bad)
    assert r.get("action") == "block", f"应拦截, got {r}"
    assert "类型注解" in r.get("message", "")


def test_non_network_no_r1_r2():
    """非网络代码不触发 R1/R2(但 R3/R4 仍生效)——此处注解齐全的
    简单非网络函数应完全放行"""
    good = """
def calc(x: int) -> int:
    return x * 2
"""
    r = _run("/proj/x.py", good)
    assert r == {}


def test_patch_tool_new_string_scanned():
    """patch 工具用 new_string 检查, 违规同样拦截"""
    bad = """
import oracledb
def f(conn):
    try:
        return conn.query("SELECT 1")
    finally:
        conn.close()
"""
    r = mod.on_pre_tool_call(tool_name="patch",
                             args={"path": "/proj/x.py", "new_string": bad,
                                   "old_string": "x"})
    assert r.get("action") == "block"
