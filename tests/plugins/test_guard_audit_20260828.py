"""护栏体系审计回归测试（2026-08-28 审计产出）。

期望值独立推导（业务逻辑，非实现反推）：
  db-safety DDL 表提取：
    DROP TABLE x / DROP TABLE IF EXISTS x / TRUNCATE x / ALTER TABLE x
    均针对具体表 → 未确认 schema 时必须 block
    SELECT 1（无具体表）→ 放行
  tool-safety 灾难删除：
    rm -rf 指向 /、~、~/ → block；指向具体子路径 → 放行
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGINS = os.path.join(HERE, "..", "..", "plugins")


def _load(rel):
    p = os.path.join(PLUGINS, rel, "__init__.py")
    spec = importlib.util.spec_from_file_location(f"guard_{rel.replace('-', '_')}_t", p)
    assert spec is not None and spec.loader is not None, p
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


db = _load("db-safety")
ts = _load("tool-safety")


def _run(mod, tool, cmd):
    return mod.on_pre_tool_call(
        tool_name=tool, args={"command": cmd}, session_id="guard_audit_test"
    )


def _assert_block(mod, tool, cmd):
    r = _run(mod, tool, cmd)
    assert isinstance(r, dict) and r.get("action") == "block", f"{cmd} 未拦截: {r}"


def _assert_pass(mod, tool, cmd):
    r = _run(mod, tool, cmd)
    assert r is None, f"{cmd} 被误拦: {r}"


# db-safety DDL
def test_drop_table_blocked_without_schema():
    _assert_block(db, "terminal", "psql -c 'DROP TABLE foo'")


def test_drop_table_if_exists_blocked():
    _assert_block(db, "terminal", "psql -c 'DROP TABLE IF EXISTS foo'")


def test_truncate_blocked():
    _assert_block(db, "terminal", "psql -c 'TRUNCATE foo'")


def test_alter_table_blocked():
    _assert_block(db, "terminal", "psql -c 'ALTER TABLE foo ADD COLUMN c int'")


def test_select_no_table_allowed():
    _assert_pass(db, "terminal", "psql -c 'SELECT 1'")


def test_select_from_table_blocked():
    _assert_block(db, "terminal", "psql -c 'SELECT * FROM customers'")


# tool-safety catastrophic rm
def test_rm_rf_root_blocked():
    _assert_block(ts, "terminal", "rm -rf /")


def test_rm_rf_home_blocked():
    _assert_block(ts, "terminal", "rm -rf ~/")


def test_rm_rf_tilde_blocked():
    _assert_block(ts, "terminal", "rm -rf ~")


def test_rm_rf_subpath_allowed():
    _assert_pass(ts, "terminal", "rm -rf /tmp/build")


def test_rm_single_file_allowed():
    _assert_pass(ts, "terminal", "rm file.txt")


def test_rm_rf_relative_subpath_allowed():
    _assert_pass(ts, "terminal", "rm -rf ./build")
