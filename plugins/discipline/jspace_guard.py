"""jspace_guard —— J-Space 不变量硬化(原 plugins/jspace-guard 迁入)。

三条不变量全部只观测、只写日志，不改写模型回复：

  不变量 6 (coverage-gate): 声称「验证/完成/通过」但不说覆盖了什么
  不变量 8 (goal-rebase-gate): 声称任务完成但本轮未回读 Goal/TODO
  不变量 9 (tri-state-gate): 事实性断言不带 [实测]/[文档]/[推断]/[未查证]

归属(2026-11-10 收敛)：与 negative_conclusion_guard 同层——都是纯诊断型
输出面不变量，无拦截副作用。迁入 discipline 后由套件统一 owner，插件数 -1。

状态：原实现用模块级单例，一个 gateway 服务多 profile 时会串态；此处按
(hermes_home, session_id) 分桶，与 finish_guard / goal-evidence-gate 一致。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 激活阈值：工具调用 < 此数时跳过检测（短任务不需要此层门控）
ACTIVATION_THRESHOLD = 8

# ── 不变量 6: coverage-gate ────────────────────────────────────

VERIFY_ASSERTION_WORDS = frozenset({
    "验证", "已验证", "已确认", "测试通过", "全绿", "全部通过",
    "verified", "confirmed", "validated", "all tests pass", "all green",
    "✅", "✓",
})

COVERAGE_WORDS = frozenset({
    "覆盖", "覆盖了", "涵盖", "所有", "全部", "完整",
    "covering", "includes", "cases", "bounds", "范围",
    "正反例", "正例", "反例",
})

COVERAGE_IN_CONTEXT = frozenset({
    "9 例", "用例", "测试用例", "test case", "正反",
    "assert", "pytest", "assertion",
})

_COVERAGE_WARNING = (
    "[J-SPACE-GUARD: 不变量6] 回复声称验证/通过（%s），"
    "但未明确说明覆盖了什么范围。"
    "验证声明必须包含覆盖范围：什么场景、多少用例、正反例对照等。"
)

# ── 不变量 8: goal-rebase-gate ────────────────────────────────

COMPLETION_ASSERTION_WORDS = frozenset({
    "完成", "已完成", "搞定", "搞定了", "做完了", "结束",
    "done", "finished", "completed", "all done",
})

GOAL_REBASE_WORDS = frozenset({
    "回读", "重读", "对照", "复核", "确认目标",
    "re-read", "reread", "check goal", "verify against",
})

REBASE_TOOL_NAMES = frozenset({
    "read_file", "search_files", "web_search", "web_extract",
    "session_search", "todo",
})

_GOAL_WARNING = (
    "[J-SPACE-GUARD: 不变量8] 回复声称任务完成（%s），"
    "但本轮未检测到回读目标/TODO 的行为。"
)

# ── 不变量 9: tri-state annotation ──────────────────────────────

FACT_ASSERTION_WORDS = frozenset({
    "修复了", "已修复", "修复完成", "解决了", "已解决",
    "实测", "实际测试", "测试确认",
    "根因是", "原因是", "定位到", "问题出在",
    "结果为", "实测结果",
    "固定了", "修复了 bug",
})

TRI_STATE_TAGS = frozenset({
    "[实测]", "[文档]", "[推断]", "[未查证]",
    "[verified]", "[documented]", "[inferred]", "[unverified]",
})

ANNOTATION_IN_CONTEXT = frozenset({
    "实测验证", "日志显示", "断言", "断言显示",
    "工具输出", "函数返回", "测试输出",
    "git diff", "git log",
})

_TRI_STATE_WARNING = (
    "[J-SPACE-GUARD: 三态标注] 回复包含事实性断言（%s），"
    "但未标注结论来源。"
)


# ── 状态（per-profile + per-session，进程内） ──────────────────

class _SessionState:
    __slots__ = ("current_turn_tools", "tool_call_count")

    def __init__(self) -> None:
        self.tool_call_count = 0
        self.current_turn_tools: set[str] = set()


_SESSIONS: dict[tuple[str, str], _SessionState] = {}
_SESSIONS_CAP = 512


def _session_key(kwargs: Any) -> tuple[str, str]:
    """一个 gateway 服务多 profile：计数必须按 (home, session) 分桶。"""
    from hermes_constants import hermes_home_key

    sid = str(kwargs.get("session_id") or kwargs.get("task_id") or "")
    return hermes_home_key(), sid


def _state_for(kwargs: Any) -> _SessionState:
    key = _session_key(kwargs)
    state = _SESSIONS.get(key)
    if state is None:
        if len(_SESSIONS) >= _SESSIONS_CAP:
            _SESSIONS.pop(next(iter(_SESSIONS)))
        state = _SessionState()
        _SESSIONS[key] = state
    return state


# ── 检测函数 ─────────────────────────────────────────────────

def _contains_any(text: str, words: frozenset[str]) -> bool:
    return any(word in text for word in words)


def _find_assertion_word(text: str, words: frozenset[str]) -> str:
    for word in sorted(words, key=len, reverse=True):
        if word in text:
            return word
    return ""


def _check_coverage_gate(text: str) -> str:
    """不变量 6：声称验证但未说明覆盖范围时返回 warning 文本。"""
    assertion = _find_assertion_word(text, VERIFY_ASSERTION_WORDS)
    if not assertion:
        return ""
    if _contains_any(text, COVERAGE_IN_CONTEXT):
        return ""
    if _contains_any(text, COVERAGE_WORDS):
        return ""
    return _COVERAGE_WARNING % assertion


def _check_goal_rebase_gate(text: str, turn_tools: set[str]) -> str:
    """不变量 8：声称完成但本轮未回读目标时返回 warning 文本。"""
    assertion = _find_assertion_word(text, COMPLETION_ASSERTION_WORDS)
    if not assertion:
        return ""
    if _contains_any(text, GOAL_REBASE_WORDS):
        return ""
    if turn_tools & REBASE_TOOL_NAMES:
        return ""
    return _GOAL_WARNING % assertion


def _check_tri_state_gate(text: str) -> str:
    """不变量 9：事实性断言缺三态标注时返回 warning 文本。"""
    if _contains_any(text, TRI_STATE_TAGS):
        return ""
    if _contains_any(text, ANNOTATION_IN_CONTEXT):
        return ""
    assertion = _find_assertion_word(text, FACT_ASSERTION_WORDS)
    if not assertion:
        return ""
    return _TRI_STATE_WARNING % assertion


# ── 钩子 ──────────────────────────────────────────────────────

def _on_post_tool_call(**kwargs: Any) -> None:
    """累计工具调用数，记录本轮调用过的工具名（供回读判定）。"""
    tool_name = str(kwargs.get("tool_name") or "")
    if not tool_name:
        return
    state = _state_for(kwargs)
    state.tool_call_count += 1
    state.current_turn_tools.add(tool_name)


def _on_transform_llm_output(**kwargs: Any) -> None:
    """扫描回复的三条不变量；只写日志，不改写回答。"""
    # cron 会话豁免：无人值守推送直接投给用户，内部审计噪音禁止混入
    if kwargs.get("platform") == "cron" or str(
        kwargs.get("session_id", "")).startswith("cron_"):
        return

    state = _state_for(kwargs)
    if state.tool_call_count < ACTIVATION_THRESHOLD:
        return

    response_text = str(kwargs.get("response_text") or "")
    if not response_text:
        return

    warnings = [
        warning
        for warning in (
            _check_coverage_gate(response_text),
            _check_goal_rebase_gate(response_text, state.current_turn_tools),
            _check_tri_state_gate(response_text),
        )
        if warning
    ]
    if warnings:
        logger.info(
            "jspace_guard: %d invariant(s) violated (calls=%d): %s",
            len(warnings), state.tool_call_count, " ".join(warnings),
        )


def _on_session_reset(**kwargs: Any) -> None:
    """会话重置时丢弃该会话的计数，避免跨轮串态。"""
    _SESSIONS.pop(_session_key(kwargs), None)


def register(ctx: Any) -> None:
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("on_session_reset", _on_session_reset)
    logger.info("discipline.jspace_guard 已注册(activation=%d)", ACTIVATION_THRESHOLD)
