"""dao-recurring-error-guard 行为测试。

覆盖：错误分类计数、阈值触发注入（含五字段固化模板锚点）、
沉淀动作静默、分类保守性。模板五字段=RSIAgent 记忆条目结构的
工程化（arXiv 2609.15364 失败模式三：丢适用范围的规则被无条件复用）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_SPEC = importlib.util.spec_from_file_location(
    "dao_guard", Path(__file__).resolve().parents[2] / "plugins" / "dao-recurring-error-guard" / "__init__.py")
dao = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dao)


def _bump_error(sid: str, tool: str, err: str, n: int = 1) -> None:
    for _ in range(n):
        dao.on_post_tool_call(session_id=sid, tool_name=tool, args={},
                              result={"error": err}, status="error")


def test_reminder_carries_five_field_template() -> None:
    sid = "dao-t1"
    _bump_error(sid, "terminal", "ModuleNotFoundError: No module named 'psycopg2'", 3)
    out = dao.on_pre_llm_call(session_id=sid)
    assert isinstance(out, dict) and "context" in out  # 期望: pre_llm_call 契约=返回 {"context":…}
    ctx = out["context"]
    for field in ("适用条件", "症状", "根因", "修复", "验证命令"):  # 期望: 五字段模板齐备
        assert field in ctx, f"固化模板缺字段 {field}"
    assert "No module named" in ctx or "psycopg2" in ctx  # 期望: 携带真实错误上下文


def test_below_threshold_no_reminder() -> None:
    sid = "dao-t2"
    _bump_error(sid, "terminal", "connection refused", 2)
    assert dao.on_pre_llm_call(session_id=sid) is None  # 期望: 2 次 < 阈值 3，静默


def test_codification_silences() -> None:
    sid = "dao-t3"
    _bump_error(sid, "terminal", "ModuleNotFoundError: foo", 3)
    dao.on_post_tool_call(session_id=sid, tool_name="skill_manage", args={}, result={}, status="ok")
    assert dao.on_pre_llm_call(session_id=sid) is None  # 期望: 沉淀动作后静默


def test_classify_error_known_and_unknown() -> None:
    assert dao.classify_error("ModuleNotFoundError: No module named 'x'") is not None  # 期望: 已知类别命中
    assert dao.classify_error("一切正常无错误") is None  # 期望: 未知文本不归类（宁可漏报）
    assert dao.classify_error("") is None  # 期望: 空串不归类


def test_success_result_with_class_word_not_counted() -> None:
    # 回归：skill_view 成功返回的 SKILL.md 正文含 TIMEOUT/timed out 字样，
    # 旧实现 `status == "error" or err_text` 把成功输出也归类计数，
    # 14 天账面 skill_view×Timeout 2940 次几乎全为误报（outcomes.db 同期真超时 1 次）
    sid = "dao-reg1"
    for _ in range(4):
        dao.on_post_tool_call(
            session_id=sid, tool_name="skill_view", args={"name": "x"},
            result={"output": '{"success": true, "name": "adversarial-review-dispatch", '
                              '"content": "_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S '
                              'Non-streaming API call timed out after 150s"}'},
            status="ok")
    assert dao.on_pre_llm_call(session_id=sid) is None  # 期望: 成功结果不计数，永不到阈值


def test_error_key_result_with_ok_status_not_counted() -> None:
    # 插件只认框架派生的 status，不做二次内容嗅探——result 带 error 键但 status=ok 不计数
    sid = "dao-reg2"
    for _ in range(4):
        dao.on_post_tool_call(session_id=sid, tool_name="skill_view", args={},
                              result={"error": "Request timed out after 30s"}, status="ok")
    assert dao.on_pre_llm_call(session_id=sid) is None  # 期望: status=ok 一律不计


def test_blocked_result_not_counted() -> None:
    # 被护栏拦截（status=blocked）不是调用错误，不进道生法计数
    sid = "dao-reg3"
    for _ in range(4):
        dao.on_post_tool_call(session_id=sid, tool_name="terminal", args={},
                              result={"error": "grep timed out blocked by guard"},
                              status="blocked")
    assert dao.on_pre_llm_call(session_id=sid) is None  # 期望: blocked 不计


def test_real_error_still_counted() -> None:
    # 真错误（status=error 且错误文本含类别词）必须照常计数——修复不得漏报真错误
    sid = "dao-reg4"
    _bump_error(sid, "skill_view", "Non-streaming API call timed out after 150s", 3)
    out = dao.on_pre_llm_call(session_id=sid)
    assert out is not None and "skill_view×Timeout" in out["context"]  # 期望: 真超时错误计数且指纹正确
