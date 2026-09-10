"""six-laws-guard E2E 测试——期望值独立推导(先写期望再验证)。

期望值来自四防线语义, 非实现反推:
  - finally 中 close 先于哨兵 put = 消费方饿死风险 → 必拦
  - finally 中裸 close(无 try 包裹) = 掩盖原始异常 → 必拦
  - close 包了 try/except = L3 合规 → 放行
  - 哨兵先于 close 且 close 有守卫 = L4+L3 合规 → 放行
  - 非网络代码(.py 但无网络痕迹) = 规则不适用 → 放行
  - 非 .py 文件 = 不扫 → 放行
"""
import importlib.util
from pathlib import Path

PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "six-laws-guard" / "__init__.py"
spec = importlib.util.spec_from_file_location("six_laws_guard", PLUGIN_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _run(path, content):
    return mod.on_pre_tool_call(tool_name="write_file",
                                args={"path": path, "content": content})


def test_r1_close_before_sentinel_blocked():
    """close 排在哨兵 put 之前 → 必拦(期望来自 L4 语义)"""
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
    """finally 中裸 close 无 try 包裹 → 必拦(期望来自 L3 语义)"""
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
    """哨兵先行 + close 有守卫 → 放行(即 b810a5e 修复后的正确形态)"""
    good = """
import oracledb
def worker(q, conn):
    try:
        rows = conn.query("SELECT 1")
        q.put(rows)
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


def test_non_network_python_pass():
    """无网络痕迹的 .py → 规则不适用 → 放行"""
    r = _run("/proj/x.py", "def add(a, b):\n    return a + b\n")
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
