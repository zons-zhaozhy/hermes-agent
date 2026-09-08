"""block_escalation 插件行为测试：意图指纹升级终止（SQLite 持久化）。

A/B 对照：不同目标各被拦 1 次 = 不升级；同一目标换通道第 2 次被拦 = 升级
（用户可见提示注入）；窗口外重置；跨进程（重导入模块）计数延续。
"""

import importlib
import sqlite3
import sys

import pytest

from plugins.block_escalation import (
    _get_db_path, _on_post_tool_call, _on_transform_llm_output, _escalated)


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    """隔离 HERMES_HOME，db 落 tmp，进程内标记清空。"""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import plugins.block_escalation as be
    be._db_path = home / "block_escalation.db"
    _escalated.clear()
    yield be
    _escalated.clear()


def _blocked(be, tool, **args):
    be._on_post_tool_call(tool_name=tool, status="blocked",
                          error_type="plugin_block", args=args)


def test_same_target_two_channels_escalates(db_env):
    be = db_env
    _blocked(be, "terminal", command="sed -i 's/a/b/' src/foo.py")
    assert not _escalated
    _blocked(be, "execute_code", code="open('src/foo.py','w').write('x')")
    assert any(any("foo.py" in part for part in fp) for fp in _escalated)
    out = _on_transform_llm_output(response_text="done")
    assert "[拦截升级]" in out


def test_different_targets_do_not_escalate(db_env):
    be = db_env
    _blocked(be, "terminal", command="sed -i 's/a/b/' src/foo.py")
    _blocked(be, "terminal", command="sed -i 's/a/b/' src/bar.py")
    assert not _escalated
    out = _on_transform_llm_output(response_text="ok")
    assert out == "ok"


def test_non_blocked_status_ignored(db_env):
    be = db_env
    be._on_post_tool_call(tool_name="terminal", status="ok", args={"command": "x.py"})
    # 库未被触发创建（零写入）也视为 0 条——文件不存在即无记录
    assert not be._db_path.exists() or _count_rows(be) == 0


def _count_rows(be) -> int:
    conn = sqlite3.connect(str(be._db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM block_streaks").fetchone()[0]
    finally:
        conn.close()


def test_escalation_flag_consumed_once(db_env):
    be = db_env
    _blocked(be, "terminal", command="sed -i 's/a/b/' a.py")
    _blocked(be, "patch", path="a.py", old_string="x", new_string="y")
    first = _on_transform_llm_output(response_text="r1")
    second = _on_transform_llm_output(response_text="r2")
    assert "[拦截升级]" in first
    assert second == "r2"


def test_window_expiry_resets(db_env):
    be = db_env
    _blocked(be, "terminal", command="sed -i 's/a/b/' z.py")
    conn = sqlite3.connect(str(be._db_path))
    conn.execute("UPDATE block_streaks SET last_ts = last_ts - 99999")
    conn.commit()
    conn.close()
    _blocked(be, "terminal", command="sed -i 's/a/b/' z.py")
    assert not _escalated


def test_persistence_across_process_restart(db_env):
    """换会话/换进程规避失效：模块重导入后同指纹计数延续。"""
    be = db_env
    _blocked(be, "terminal", command="sed -i 's/a/b/' src/foo.py")
    assert not _escalated
    del sys.modules["plugins.block_escalation"]
    be2 = importlib.import_module("plugins.block_escalation")
    be2._db_path = be._db_path  # 同一 HERMES_HOME
    be2._escalated.clear()
    _blocked(be2, "execute_code", code="open('src/foo.py','w').write('x')")
    assert any(any("foo.py" in part for part in fp) for fp in be2._escalated)
    conn = sqlite3.connect(str(be2._db_path))
    assert conn.execute(
        "SELECT count FROM block_streaks WHERE count >= 2").fetchone() is not None
    conn.close()
