"""guards.skill_pruned_gate 回归测试（期望值独立推导自守卫设计）。

覆盖契约：
1. skill_view 返回 [SKILL_PRUNED] 占位符 → 技能名入待重载集合 → 写通道 block
2. 再次 skill_view 返回正常全文 → 移出集合 → 写通道放行（重载即解除）
3. 行内豁免标记 ``# skill-pruned-ok`` → 放行
4. execute_code 只读代码 → 放行；含写形态 → 拦
5. 名字提取：args.name 优先；result 切片回退；无锚点 → 不记账
6. 守卫主体异常 → fail-open（return None）
7. __init__ 在岗状态行：全员加载出确认行；模拟 import 失败出缺失名单

Contract:
  Preconditions: 每用例唯一 sid 防会话态互串；零真实仓依赖。
  Postconditions: 全部断言通过 = 上述契约成立。
"""

from __future__ import annotations

import itertools
from typing import Any

import plugins.guards as guards_init
from plugins.guards import skill_pruned_gate as spg

_sid = itertools.count()

PRUNED_RESULT = (
    "[skill_view] name='design-aesthetics-essence' (16,190 chars) "
    "[SKILL_PRUNED: content lost in compression; reload with "
    "skill_view(name='design-aesthetics-essence')]"
)
FULL_RESULT = "---\nname: design-aesthetics-essence\ndescription: 触发:...\n---\n正文"


def _post(sid: str, result: str, name: str = "design-aesthetics-essence") -> None:
    spg.on_post_tool_call(
        session_id=sid, tool_name="skill_view", args={"name": name}, result=result
    )


def _pre(sid: str, tool: str = "write_file", **args: Any) -> dict[str, Any] | None:
    return spg.on_pre_tool_call(session_id=sid, tool_name=tool, args=args)


def test_pruned_then_write_blocks_and_reload_releases() -> None:
    sid = f"t-{next(_sid)}"
    _post(sid, PRUNED_RESULT)
    blocked = _pre(sid, path="/tmp/x.py", content="print(1)")
    assert blocked is not None  # 期望: 占位符未重载即拦(设计:集合非空→block)
    assert blocked["action"] == "block"  # 期望: 拦截动作=block
    assert "design-aesthetics-essence" in blocked["message"]  # 期望: 消息点名待重载技能
    _post(sid, FULL_RESULT)
    released = _pre(sid, path="/tmp/x.py", content="print(1)")
    assert released is None  # 期望: 正常全文移出集合,重载即解除


def test_exempt_marker_passes() -> None:
    sid = f"t-{next(_sid)}"
    _post(sid, PRUNED_RESULT)
    out = _pre(sid, path="/tmp/x.py", content="# skill-pruned-ok: 与该技能无关")
    assert out is None  # 期望: 行内豁免标记放行(设计:_EXEMPT_MARKER)


def test_execute_code_readonly_passes_write_blocks() -> None:
    sid = f"t-{next(_sid)}"
    _post(sid, PRUNED_RESULT)
    assert _pre(sid, tool="execute_code", code="print(sum(range(10)))") is None  # 期望: 只读代码无写形态放行
    blocked = _pre(sid, tool="execute_code", code="open('/tmp/w.txt','w').write('x')")
    assert blocked is not None  # 期望: 含 open( 写形态即拦


def test_name_extraction_priority_and_fallback() -> None:
    assert spg._extract_skill_name(PRUNED_RESULT, {"name": "args-first"}) == "args-first"  # 期望: args.name 优先
    assert spg._extract_skill_name(PRUNED_RESULT, {}) == "design-aesthetics-essence"  # 期望: 无 args 时切片回退取占位符内名字
    assert spg._extract_skill_name("no anchor here", {}) is None  # 期望: 无锚点返回 None 不记账


def test_guard_exception_fail_open(monkeypatch) -> None:
    def _boom(sid: str) -> set[str]:
        raise RuntimeError("state store down")

    monkeypatch.setattr(spg, "_pending", _boom)
    assert spg.on_pre_tool_call(session_id="t-boom", tool_name="write_file", args={}) is None  # 期望: 守卫异常 fail-open


def test_status_line_all_loaded_and_missing() -> None:
    line = guards_init._guards_status_line()
    assert "guards 在岗" in line  # 期望: 全员加载给确认行
    assert "16/16" in line  # 期望: 套件 16 个子护栏全部导入成功
    real = guards_init._LOADED
    try:  # 模拟一个子护栏 import 失败
        guards_init._LOADED = [t for t in real if t[0] != "plain_vocab"]
        degraded = guards_init._guards_status_line()
        assert "15/16" in degraded  # 期望: 缺一个时计数降 1
        assert "plain_vocab" in degraded  # 期望: 缺失名单点名失效守卫
    finally:
        guards_init._LOADED = real


def test_suite_registers_new_subguard() -> None:
    assert "skill_pruned_gate" in guards_init._SUB_GUARD_NAMES  # 期望: 套件常量含新守卫
    names = [n for n, _ in guards_init._SUB_GUARDS]  # skill-pruned-ok: revert to exact 2-tuple shape
    assert "skill_pruned_gate" in names  # 期望: 实际 import 成功进入注册列表


def test_reload_with_marker_in_body_releases() -> None:
    # skill-pruned-ok: gate self-repair regression test
    """回归：技能正文引用 [SKILL_PRUNED 字样（engineering-lessons 教学案例）不得误判为占位符。"""
    sid = f"t-{next(_sid)}"
    _post(sid, PRUNED_RESULT)
    long_body = "x" * 500 + "\n...教学案例: skill_view 返回 [SKILL_PRUNED: content lost] 标记...\n"
    _post(sid, long_body)
    assert _pre(sid, path="/tmp/x.py", content="print(1)") is None  # 期望: 头部窗口外命中不算 PRUNED,重载即解除


def test_head_window_still_catches_real_placeholder() -> None:
    # skill-pruned-ok: gate self-repair regression test
    """回归：真实占位符（标记在头部状态行）仍须入集合拦截。"""
    sid = f"t-{next(_sid)}"
    _post(sid, PRUNED_RESULT)
    assert _pre(sid, path="/tmp/x.py", content="print(1)") is not None  # 期望: 头部窗口内命中仍拦
