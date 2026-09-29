"""diff_debt 子防线 + write_file diff 回显 行为验证（红绿测试）。

层1（治本）：WriteResult.diff——write_file 回显自带 old→new unified diff。
层3（兜底）：guards.diff_debt——欠账记账/证据清账/block 执法。

Independent-expectation tests: every assertion derives from the rule text
(.hermes-rules.md 编辑纪律), not from the implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.guards import diff_debt  # noqa: E402


def _fresh_debt() -> None:
    diff_debt._DIFF_DEBT.clear()


# ── 层3：diff_debt 子防线 ─────────────────────────────────────────────────


def test_post_records_debt_for_write_tools() -> None:
    # 期望: write_file/patch/skill_manage 执行后各记一笔欠账
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    diff_debt.on_post_tool_call(tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s1")
    diff_debt.on_post_tool_call(tool_name="skill_manage", args={"name": "x"}, session_id="s1")
    diff_debt.on_post_tool_call(tool_name="read_file", args={"path": "/tmp/c.py"}, session_id="s1")
    assert len(diff_debt._DIFF_DEBT) == 3  # 期望: 读工具不入账


def test_pre_blocks_when_debt_unpaid() -> None:
    # 期望: 欠账未清时下一次写被 block，消息含补救指引
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    verdict = diff_debt.on_pre_tool_call(tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s1")
    assert verdict is not None and verdict.get("action") == "block"
    assert "diff" in verdict["message"]


def test_batch_clears_debt_on_diff_evidence() -> None:
    # 期望: assistant 正文含 ```diff 块 → 欠账清空
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    out = diff_debt.on_pre_tool_batch(
        assistant_content="改动如下\n```diff\n-old\n+new\n```\n", session_id="s1")
    assert out is None and not diff_debt._DIFF_DEBT


def test_batch_clears_debt_on_compensation_keywords() -> None:
    # 期望: skill_manage 无 diff 回显时，「改动前/改动后」文本也算合规代偿
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="skill_manage", args={"name": "x"}, session_id="s1")
    diff_debt.on_pre_tool_batch(
        assistant_content="改动前: 旧规则行\n改动后: 新规则行", session_id="s1")
    assert not diff_debt._DIFF_DEBT


def test_batch_keeps_debt_without_evidence() -> None:
    # 期望: 正文无任何 diff 痕迹 → 欠账保留 → 下一次写仍被拦
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    diff_debt.on_pre_tool_batch(assistant_content="已完成修改，一切正常。", session_id="s1")
    verdict = diff_debt.on_pre_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    assert verdict is not None and verdict["action"] == "block"


def test_cron_session_exempt() -> None:
    # 期望: cron 会话全链路豁免（记账/执法都跳过）
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="cron_job1")
    assert not diff_debt._DIFF_DEBT
    assert diff_debt.on_pre_tool_call(tool_name="write_file", args={}, session_id="cron_job1") is None


def test_git_diff_output_counts_as_evidence() -> None:
    # 期望: 「1 file changed」git diff 输出文本也算证据（terminal 写通道贴 git diff 的形态）
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    diff_debt.on_pre_tool_batch(
        assistant_content=" 1 file changed, 2 insertions(+), 1 deletion(-)", session_id="s1")
    assert not diff_debt._DIFF_DEBT


# ── 层1：WriteResult.diff 回显 ────────────────────────────────────────────


def test_write_result_has_diff_field() -> None:
    # 期望: WriteResult 数据类带 diff 字段且 to_dict 自然带出（None 被过滤不打扰旧契约）
    from tools.file_operations_common import WriteResult
    r = WriteResult(bytes_written=10, verified=True, diff="--- a/x\n+++ b/x\n@@\n-old\n+new\n")
    d = r.to_dict()
    assert d.get("diff", "").startswith("--- a/")
    r2 = WriteResult(bytes_written=10)
    assert "diff" not in r2.to_dict()


def test_bounded_write_diff_new_file_returns_none() -> None:
    # 期望: 新文件（无旧内容）与无变化 → 无 diff（None），不产生噪音
    from tools.file_operations import ShellFileOperations
    ops = ShellFileOperations.__new__(ShellFileOperations)  # 只测纯函数,不初始化 exec 后端
    assert ops._bounded_write_diff(None, "new", "/tmp/x.py") is None
    assert ops._bounded_write_diff("same", "same", "/tmp/x.py") is None


def test_bounded_write_diff_produces_unified_diff() -> None:
    # 期望: 有旧内容时产出标准 unified diff（--- a/ +++ b/ 头）
    from tools.file_operations import ShellFileOperations
    ops = ShellFileOperations.__new__(ShellFileOperations)
    out = ops._bounded_write_diff("line1\nline2\n", "line1\nlineX\n", "/tmp/x.py")
    assert out is not None and out.startswith("--- a/") and "+lineX" in out and "-line2" in out


def test_bounded_write_diff_truncates_huge_delta() -> None:
    # 期望: 超 400 行 diff 被截断为头尾 40 + 统计行
    from tools.file_operations import ShellFileOperations
    ops = ShellFileOperations.__new__(ShellFileOperations)
    old = "".join(f"old{i}\n" for i in range(1000))
    new = "".join(f"new{i}\n" for i in range(1000))
    out = ops._bounded_write_diff(old, new, "/tmp/big.py")
    assert out is not None and "[diff truncated:" in out and "head 40 / tail 40" in out
    assert len(out.splitlines()) <= 90  # 期望: 40+1+40 加头部行,远小于全量


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
