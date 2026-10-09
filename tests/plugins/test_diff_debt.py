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

from plugins.guards import diff_debt


def _fresh_debt() -> None:
    diff_debt._DIFF_DEBT.clear()
    diff_debt._BATCH_EVIDENCE.clear()   # 两个账本都由会话键控，一并清空


# ── 层3：diff_debt 子防线 ─────────────────────────────────────────────────


def test_post_records_debt_for_write_tools() -> None:
    # 期望: write_file/patch/skill_manage 执行后各记一笔欠账
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    diff_debt.on_post_tool_call(tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s1")
    diff_debt.on_post_tool_call(tool_name="skill_manage", args={"name": "x"}, session_id="s1")
    diff_debt.on_post_tool_call(tool_name="read_file", args={"path": "/tmp/c.py"}, session_id="s1")
    assert len(diff_debt._DIFF_DEBT["s1"]) == 3  # 期望: 读工具不入账（同会话 3 笔）


def test_pre_blocks_when_debt_unpaid() -> None:
    # 期望: 欠账未清时下一次写被 block，消息含补救指引
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    verdict = diff_debt.on_pre_tool_call(tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s1")
    assert verdict is not None and verdict.get("action") == "block"  # 期望: 拦
    assert "diff" in verdict["message"]  # 期望: 消息含 diff 字样（补救指引）


def test_batch_clears_debt_on_diff_evidence() -> None:
    # 期望: assistant 正文含 ```diff 块 → 欠账清空
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    out = diff_debt.on_pre_tool_batch(
        assistant_content="改动如下\n```diff\n-old\n+new\n```\n", session_id="s1")
    assert out is None and not diff_debt._DIFF_DEBT  # 期望: 批次层不拦 + 欠账清零


def test_batch_clears_debt_on_compensation_keywords() -> None:
    # 期望: skill_manage 无 diff 回显时，「改动前/改动后」文本也算合规代偿
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="skill_manage", args={"name": "x"}, session_id="s1")
    diff_debt.on_pre_tool_batch(
        assistant_content="改动前: 旧规则行\n改动后: 新规则行", session_id="s1")
    assert not diff_debt._DIFF_DEBT  # 期望: 代偿关键词同样清账


def test_batch_keeps_debt_without_evidence() -> None:
    # 期望: 正文无任何 diff 痕迹 → 欠账保留 → 下一次写仍被拦
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    diff_debt.on_pre_tool_batch(assistant_content="已完成修改，一切正常。", session_id="s1")
    verdict = diff_debt.on_pre_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    assert verdict is not None and verdict["action"] == "block"  # 期望: 无证据仍拦


def test_batch_of_two_writes_with_evidence_shown_once_is_allowed() -> None:
    # 期望: 正文已出示 diff 的批次，批内多笔写不互相记债成阻（规则要的是「写前贴」）
    _fresh_debt()
    diff_debt.on_pre_tool_batch(assistant_content="本批两笔:\n```diff\n-a\n+b\n```\n", session_id="s1")
    diff_debt.on_post_tool_call(tool_name="patch", args={"path": "/tmp/a.py"}, session_id="s1")
    verdict = diff_debt.on_pre_tool_call(tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s1")
    assert verdict is None  # 期望: 同批已出示证据 → 放行


def test_next_batch_without_evidence_still_blocks() -> None:
    # 期望: 新批次正文无 diff 证据 → 旧欠账仍拦（防放宽执法）
    _fresh_debt()
    diff_debt.on_pre_tool_batch(assistant_content="本批:\n```diff\n-a\n+b\n```\n", session_id="s1")
    diff_debt.on_post_tool_call(tool_name="patch", args={"path": "/tmp/a.py"}, session_id="s1")
    diff_debt.on_pre_tool_batch(assistant_content="写完了，一切正常。", session_id="s1")
    verdict = diff_debt.on_pre_tool_call(tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s1")
    assert verdict is not None and verdict["action"] == "block"  # 期望: 跨批无证据仍拦


def test_debt_is_session_scoped() -> None:
    # 期望: 会话隔离——A 会话的欠账不得拦 B 会话的写（网关单进程多会话）
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="patch", args={"path": "/tmp/a.py"}, session_id="s1")
    sid2_verdict = diff_debt.on_pre_tool_call(
        tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s2")
    assert sid2_verdict is None  # 期望: 他会话不受牵连
    verdict = diff_debt.on_pre_tool_call(tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s1")
    assert verdict is not None and verdict["action"] == "block"  # 期望: 本会话仍拦


def test_evidence_does_not_license_other_sessions() -> None:
    # 期望: A 会话正文出示 diff 不得给 B 会话的欠账放行
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="patch", args={"path": "/tmp/a.py"}, session_id="s2")
    diff_debt.on_pre_tool_batch(assistant_content="本批:\n```diff\n-a\n+b\n```\n", session_id="s1")
    verdict = diff_debt.on_pre_tool_call(tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s2")
    assert verdict is not None and verdict["action"] == "block"  # 期望: B 会话证据位未被 A 污染


def test_cron_session_exempt() -> None:
    # 期望: cron 会话全链路豁免（记账/执法都跳过）
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="cron_job1")
    assert not diff_debt._DIFF_DEBT  # 期望: cron 记账被跳过
    cron_verdict = diff_debt.on_pre_tool_call(tool_name="write_file", args={}, session_id="cron_job1")
    assert cron_verdict is None  # 期望: cron 会话不执法


def test_git_diff_output_counts_as_evidence() -> None:
    # 期望: 「1 file changed」git 输出也当证据——terminal 写通道贴 git diff 的形态
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    diff_debt.on_pre_tool_batch(
        assistant_content=" 1 file changed, 2 insertions(+), 1 deletion(-)", session_id="s1")
    assert not diff_debt._DIFF_DEBT  # 期望: git 统计行同样清账


def test_failed_write_records_no_debt() -> None:
    # 期望: status=error 的写（被其他 guard 拦/patch 找不到 match）未改动文件，不产生欠账
    _fresh_debt()
    diff_debt.on_post_tool_call(
        tool_name="patch", args={"path": "/tmp/b.py"}, session_id="s1", status="error")
    assert not diff_debt._DIFF_DEBT  # 期望: 失败写零欠账


def test_cancelled_write_records_no_debt() -> None:
    # 期望: status=cancelled（超时放弃）同样未改动文件，不记账
    _fresh_debt()
    diff_debt.on_post_tool_call(
        tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1", status="cancelled")
    assert not diff_debt._DIFF_DEBT  # 期望: 取消写零欠账


def test_ok_write_records_debt_without_status_kwarg() -> None:
    # 期望: 缺省 status 按成功处理仍记账——旧派发路径未传该字段，保持兼容
    _fresh_debt()
    diff_debt.on_post_tool_call(tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s1")
    assert len(diff_debt._DIFF_DEBT["s1"]) == 1  # 期望: 恰好一笔欠账


# ── 层1：WriteResult.diff 回显 ────────────────────────────────────────────


def test_write_result_has_diff_field() -> None:
    # 期望: WriteResult 数据类带 diff 字段且 to_dict 自然带出（None 被过滤不打扰旧契约）
    from tools.file_operations_common import WriteResult
    r = WriteResult(bytes_written=10, verified=True, diff="--- a/x\n+++ b/x\n@@\n-old\n+new\n")
    d = r.to_dict()
    assert d.get("diff", "").startswith("--- a/")  # 期望: 有 diff 时原样透出
    r2 = WriteResult(bytes_written=10)
    assert "diff" not in r2.to_dict()  # 期望: diff=None 时键不出现（不打扰旧契约）


def test_bounded_write_diff_new_file_returns_none() -> None:
    # 期望: 新文件（无旧内容）与无变化 → 无 diff（None），不产生噪音
    from tools.file_operations import ShellFileOperations
    ops = ShellFileOperations.__new__(ShellFileOperations)  # 只测纯函数,不初始化 exec 后端
    assert ops._bounded_write_diff(None, "new", "/tmp/x.py") is None  # 期望: 新文件无 diff
    assert ops._bounded_write_diff("same", "same", "/tmp/x.py") is None  # 期望: 内容相同无 diff


def test_bounded_write_diff_produces_unified_diff() -> None:
    # 期望: 有旧内容时产出标准 unified diff（--- a/ +++ b/ 头）
    from tools.file_operations import ShellFileOperations
    ops = ShellFileOperations.__new__(ShellFileOperations)
    out = ops._bounded_write_diff("line1\nline2\n", "line1\nlineX\n", "/tmp/x.py")
    diff_ok = out is not None and out.startswith("--- a/") and "+lineX" in out and "-line2" in out
    assert diff_ok  # 期望: 标准 diff 三要素（a/ 头、+新行、-旧行）齐备


def test_bounded_write_diff_truncates_huge_delta() -> None:
    # 期望: 超 400 行 diff 被截断为头尾 40 + 统计行
    from tools.file_operations import ShellFileOperations
    ops = ShellFileOperations.__new__(ShellFileOperations)
    old = "".join(f"old{i}\n" for i in range(1000))
    new = "".join(f"new{i}\n" for i in range(1000))
    out = ops._bounded_write_diff(old, new, "/tmp/big.py")
    truncation_ok = out is not None and "[diff truncated:" in out and "head 40 / tail 40" in out
    assert truncation_ok  # 期望: 截断标记 + 头尾说明
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
