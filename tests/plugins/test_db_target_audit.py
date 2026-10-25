"""guards.db_target_audit 判定单元测试。

形态对齐 test_source_write_inplace_gate：直接测判定函数+monkeypatch lsof 取证。
预期判定不变量：
  - psql 带 -h/-p + 写动词 + lsof ≥2 活动端点 → block
  - 默认连接（无 -h/-p）→ 放行（单实例无歧义）
  - 豁免标记 → 放行
  - lsof 失败/单实例 → fail-open 放行
"""
# dupcheck-ok: 全库 search_files db_target_audit 命中仅本测试与插件本体，无既有等价测试
from __future__ import annotations

from plugins.guards import db_target_audit as dta


def test_write_segment_psql_with_endpoint_flag():
    # 期望: psql -h -c UPDATE 命中写段判定
    seg = "psql -h localhost -p 5416 -d oms -c 'UPDATE t SET x=1'"
    assert dta._is_db_write_segment(dta._segment_tokens(seg)) is True  # 期望: True——写动词+显式端点


def test_write_segment_default_connection_allowed():
    # 期望: 无 -h/-p 的默认连接不拦（单实例无歧义）
    seg = "psql -d oms -c 'UPDATE t SET x=1'"
    assert dta._is_db_write_segment(dta._segment_tokens(seg)) is False  # 期望: False——无显式端点


def test_write_segment_read_only_select_allowed():
    # 期望: SELECT 只读不拦
    seg = "psql -h localhost -p 5416 -d oms -c 'SELECT count(*) FROM t'"
    assert dta._is_db_write_segment(dta._segment_tokens(seg)) is False  # 期望: False——只读


def test_waiver_mark_allows(monkeypatch):
    # 期望: # dbtarget-ok 豁免后放行，即使多实例写
    monkeypatch.setattr(dta, "_active_pg_endpoints", lambda: ["localhost:5416", "localhost:15416"])
    res = dta.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "psql -h localhost -p 5416 -c 'UPDATE t SET x=1'  # dbtarget-ok: 已对账"},
    )
    assert res is None  # 期望: None——行内豁免放行


def test_multi_instance_write_blocked(monkeypatch):
    # 期望: 双活动端点下写命令被 block，消息含端点证据
    monkeypatch.setattr(dta, "_active_pg_endpoints", lambda: ["localhost:5416", "localhost:15416"])
    res = dta.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "psql -h 127.0.0.1 -p 5416 -d oms -c \"DELETE FROM t WHERE id=1\""},
    )
    assert res is not None and res["action"] == "block"  # 期望: block——目标库歧义未消除
    assert "5416" in res["message"] and "15416" in res["message"]  # 期望: lsof 证据入消息


def test_single_instance_write_allowed(monkeypatch):
    # 期望: 单活动端点（无歧义）fail-open 放行
    monkeypatch.setattr(dta, "_active_pg_endpoints", lambda: ["localhost:5416"])
    res = dta.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "psql -h 127.0.0.1 -p 5416 -d oms -c 'UPDATE t SET x=1'"},
    )
    assert res is None  # 期望: None——单实例不拦


def test_lsof_failure_fails_open(monkeypatch):
    # 期望: lsof 取证失败（空表）fail-open
    monkeypatch.setattr(dta, "_active_pg_endpoints", list)  # 期望: 空表——lsof 取证失败形态
    res = dta.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "psql -h 127.0.0.1 -p 15416 -d oms -c 'DELETE FROM t'"},
    )
    assert res is None  # 期望: None——无法判定时不误拦


def test_non_terminal_tool_ignored(monkeypatch):
    # 期望: 非 terminal 工具直接放行
    monkeypatch.setattr(dta, "_active_pg_endpoints", lambda: ["a:1", "b:2"])
    res = dta.on_pre_tool_call(tool_name="read_file", args={"path": "/tmp/x.sql"})
    assert res is None  # 期望: None——射程仅 terminal
