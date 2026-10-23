"""diff_debt 回显豁免工具范围回归（反方审查 D2/D3）。

D2: skill_manage 的 result 含 _change.description（模型自写 frontmatter
description 前 120 字符）——description 内出现 "@@ -1" 等子串不得静音欠账；
豁免只对 write_file/patch（框架 diff 字段）生效。

期望值独立推导（源自 .hermes-rules.md 编辑纪律）：
- write_file/patch 结果含 diff 形态 → 不记账（回显即履约）
- skill_manage 结果无论含什么子串 → 记账（无框架 diff 回显）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.guards import diff_debt


def _fresh_debt() -> None:
    diff_debt._DIFF_DEBT.clear()
    diff_debt._BATCH_EVIDENCE.clear()


def test_echo_exemption_still_applies_to_write_file() -> None:
    # 期望: write_file 结果含 diff 形态 → 不记账
    _fresh_debt()
    diff_debt.on_post_tool_call(
        tool_name="write_file", args={"path": "/repo/a.py"},
        result="{'success': True, 'diff': 'diff --git a/a.py b/a.py\\n@@ -1 +1,2 @@'}",
        session_id="s1")
    assert "s1" not in diff_debt._DIFF_DEBT  # 期望: 回显豁免生效零欠账


def test_skill_manage_result_with_diff_shaped_text_still_records_debt() -> None:
    # 期望: skill_manage 的 description 含 "@@ -1"（模型可控文本）仍记账
    _fresh_debt()
    diff_debt.on_post_tool_call(
        tool_name="skill_manage", args={"name": "demo-skill"},
        result='{"success": true, "_change": {"description": "patch demo @@ -1 +1 @@ fix"}}',
        session_id="s1")
    assert len(diff_debt._DIFF_DEBT.get("s1", [])) == 1  # 期望: 模型可控子串不豁免
