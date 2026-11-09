"""guards.watch_pattern 不变式测试。

覆盖场景（正反例对照）：
- 正例: terminal 后台任务 notify 含裸噪声词整词（error/FAILED 等大小写形态）→ block
- 负例: 精确短语 "ERROR in" → 放行
- 负例: notify=True（完成时通知）或非 list 或缺省 → 放行
- 负例: 非 terminal 工具 → 放行
- 豁免: 命令内 `# watch-ok: <理由>` → 放行
- fail-open: args=None 异常形态 → 放行不抛
"""

from __future__ import annotations

from plugins.guards.watch_pattern import on_pre_tool_call


def test_bare_error_word_blocked():
    r = on_pre_tool_call(
        tool_name="terminal",
        args={"command": "npm run dev", "background": True, "notify": ["error", "App running"]},
    )
    assert r.get("action") == "block"  # 期望: 裸 error 整词命中即拦


def test_bare_word_case_insensitive():
    r = on_pre_tool_call(
        tool_name="terminal", args={"command": "pytest -q", "notify": ["FAILED"]}
    )
    assert r.get("action") == "block"  # 期望: 大小写不敏感整词命中即拦


def test_precise_phrase_allowed():
    r = on_pre_tool_call(
        tool_name="terminal", args={"command": "npm run dev", "notify": ["ERROR in", "App running"]}
    )
    assert r == {}  # 期望: 精确短语不误伤


def test_notify_true_or_absent_allowed():
    assert on_pre_tool_call(tool_name="terminal", args={"command": "x", "notify": True}) == {}  # 期望: 完成时通知形态放行
    assert on_pre_tool_call(tool_name="terminal", args={"command": "x"}) == {}  # 期望: 无 notify 放行


def test_non_terminal_tool_allowed():
    r = on_pre_tool_call(tool_name="write_file", args={"path": "a", "notify": ["error"]})
    assert r == {}  # 期望: 只管 terminal


def test_waiver_allows():
    r = on_pre_tool_call(
        tool_name="terminal",
        args={"command": "npm run dev # watch-ok: 该日志无文件名噪声", "notify": ["error"]},
    )
    assert r == {}  # 期望: 行内豁免放行


def test_fail_open_on_bad_args():
    assert on_pre_tool_call(tool_name="terminal", args=None) == {}  # 期望: 异常形态 fail-open
