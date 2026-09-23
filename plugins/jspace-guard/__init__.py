"""
jspace-guard plugin v1.1 — J-Space 不变量硬化。

把结构化思考的关键不变量从软提醒提升到代码拦截：

不变量 6 (coverage-gate):
  声称"验证/完成/通过/已确认"但不说覆盖了什么（范围/用例/cases/bounds）
  → 注入警告：验证必须带覆盖范围说明。
  依据：J-Space COVERAGE 原则——"不能改变动作的监控只是评论"，
  声称验证却不描述覆盖范围 = 空声明。

不变量 8 (goal-rebase-gate):
  声称"任务完成/搞定/结束"但本轮未回读 Goal/TODO/任务列表
  → 注入警告：完成前必须回读目标确认未遗漏。
  依据：J-Space Loop Ledger —— 完成 = 回读 Goal + Verified + 覆盖说明，
  不是"我觉得做完了"。

不变量 9 (tri-state-gate):
  回复包含事实性断言但不标注结论来源
  → 注入提醒：结论应带 [实测]/[文档]/[推断]/[未查证] 三态标注。
  依据：实事求是原则——每个事实性结论标注来源，不编造。

设计原则：
  - transform_llm_output 只追加不替换，不修改模型原始回复
  - 只在 ≥ ACTIVATION_THRESHOLD 次工具调用后激活（短任务不需要此层门控）
  - 纯字符串检测，不用正则
  - 不变量 6: 检测"验证类断言词" + "覆盖范围词"同时/缺失
  - 不变量 8: 检测"完成类断言词" + 本轮是否有 todo/read 回读
  - 不变量 9: 检测"事实性断言词" + "三态标注词"同时/缺失
"""

import logging

logger = logging.getLogger(__name__)

# ── 阈值 ──────────────────────────────────────────────────────

# 激活阈值：工具调用 < 此数时跳过检测（短任务不需要）
ACTIVATION_THRESHOLD = 8

# ── 不变量 6: coverage-gate ────────────────────────────────────

# 验证/确认类断言词——出现在回复中意味着模型声称"这事验证过了"
VERIFY_ASSERTION_WORDS = frozenset({
    "验证", "已验证", "已确认", "测试通过", "全绿", "全部通过",
    "verified", "confirmed", "validated", "all tests pass", "all green",
    "✅", "✓",
})

# 覆盖范围词——必须至少出现一个才构成有效验证声明
COVERAGE_WORDS = frozenset({
    "覆盖", "覆盖了", "涵盖", "所有", "全部", "完整",
    "covering", "includes", "cases", "bounds", "范围",
    "正反例", "正例", "反例",
})

# 免检上下文：回复中同时包含这些词时，说明覆盖已在描述中
COVERAGE_IN_CONTEXT = frozenset({
    "9 例", "用例", "测试用例", "test case", "正反",
    "assert", "pytest", "assertion",
})

_COVERAGE_WARNING = (
    "\n[J-SPACE-GUARD: 不变量6] 回复声称验证/通过（%s），"
    "但未明确说明覆盖了什么范围。\n"
    "验证声明必须包含覆盖范围：什么场景、多少用例、正反例对照等。\n"
    "不说覆盖范围的验证 = 空声明。"
)

# ── 不变量 8: goal-rebase-gate ────────────────────────────────

# 完成/结束类断言词
COMPLETION_ASSERTION_WORDS = frozenset({
    "完成", "已完成", "搞定", "搞定了", "做完了", "结束",
    "done", "finished", "completed", "all done",
})

# 回读/确认目标的行为关键词——出现说明模型在完成前回读了目标
GOAL_REBASE_WORDS = frozenset({
    "回读", "重读", "对照", "复核", "确认目标",
    "re-read", "reread", "check goal", "verify against",
})

# 只读工具名——调用这些工具视为"回读行为"
REBASE_TOOL_NAMES = frozenset({
    "read_file", "search_files", "web_search", "web_extract",
    "session_search", "todo",
})

_GOAL_WARNING = (
    "\n[J-SPACE-GUARD: 不变量8] 回复声称任务完成（%s），"
    "但本轮未检测到回读目标/TODO 的行为。\n"
    "完成前必须：read_file 重读 Goal / read todo 列表 / 对照原始需求确认无遗漏。\n"
    "不回读目标的完成声明 = 猜测式收工。"
)

# ── 不变量 9: tri-state annotation ──────────────────────────────

# 事实性断言词——出现意味着模型在做事实声明
FACT_ASSERTION_WORDS = frozenset({
    "修复了", "已修复", "修复完成", "解决了", "已解决",
    "实测", "实际测试", "测试确认",
    "根因是", "原因是", "定位到", "问题出在",
    "结果为", "实测结果",
    "固定了", "修复了 bug",
})

# 有效三态标注——出现说明模型在标注结论来源
TRI_STATE_TAGS = frozenset({
    "[实测]", "[文档]", "[推断]", "[未查证]",
    "[verified]", "[documented]", "[inferred]", "[unverified]",
})

# 已有标注模式的宽松检测——回复中包含这些模式说明已有来源说明
ANNOTATION_IN_CONTEXT = frozenset({
    "实测验证", "日志显示", "断言", "断言显示",
    "工具输出", "函数返回", "测试输出",
    "git diff", "git log",
})

_TRI_STATE_WARNING = (
    "\n[J-SPACE-GUARD: 三态标注] 回复包含事实性断言（%s），"
    "但未标注结论来源。\n"
    "每个事实性结论应标注来源：[实测] 代码跑出来的 / [文档] 文档写的 / [推断] 推导的 / [未查证] 没查过。\n"
    "例：「修复了 XXX 漏洞 [实测]」「根因是 YYY [推断]」"
)


# ── 状态（per-session，进程内） ────────────────────────────────

class _SessionState:
    def __init__(self):
        self.tool_call_count = 0
        self.current_turn_tools: set = set()  # 本轮调用过的工具名


_state = _SessionState()


# ── 检测函数 ─────────────────────────────────────────────────

def _contains_any(text: str, words: frozenset) -> bool:
    """Check if text contains any word from the set (substring match)."""
    for w in words:
        if w in text:
            return True
    return False


def _find_assertion_word(text: str, words: frozenset) -> str:
    """Return the first matched assertion word, or empty string."""
    for w in sorted(words, key=len, reverse=True):  # longest first
        if w in text:
            return w
    return ""


def _check_coverage_gate(text: str) -> str:
    """Check invariant 6: verification without coverage description.

    Returns warning text if violated, empty string if OK.
    """
    # 先检查是否有验证断言
    assertion = _find_assertion_word(text, VERIFY_ASSERTION_WORDS)
    if not assertion:
        return ""

    # 如果已有覆盖描述（在 COVERAGE_IN_CONTEXT 里），放行
    if _contains_any(text, COVERAGE_IN_CONTEXT):
        return ""

    # 如果有覆盖范围词，放行
    if _contains_any(text, COVERAGE_WORDS):
        return ""

    # 违反：声称验证但无覆盖
    logger.info(
        "jspace-guard: coverage-gate triggered (assertion='%s')",
        assertion,
    )
    return _COVERAGE_WARNING % assertion


def _check_goal_rebase_gate(text: str, turn_tools: set) -> str:
    """Check invariant 8: completion without goal re-read.

    Returns warning text if violated, empty string if OK.
    """
    # 先检查是否有完成断言
    assertion = _find_assertion_word(text, COMPLETION_ASSERTION_WORDS)
    if not assertion:
        return ""

    # 如果回复中有回读目标的描述，放行
    if _contains_any(text, GOAL_REBASE_WORDS):
        return ""

    # 如果本轮调用了回读类工具，放行
    if turn_tools & REBASE_TOOL_NAMES:
        return ""

    # 违反：声称完成但未回读
    logger.info(
        "jspace-guard: goal-rebase-gate triggered (assertion='%s', tools=%s)",
        assertion, turn_tools,
    )
    return _GOAL_WARNING % assertion


def _check_tri_state_gate(text: str) -> str:
    """Check invariant 9: factual assertions without source annotation.

    Returns warning text if violated, empty string if OK.
    """
    # 先检查是否有三态标注——全文有就放行
    if _contains_any(text, TRI_STATE_TAGS):
        return ""

    # 如果已有上下文标注模式，放行
    if _contains_any(text, ANNOTATION_IN_CONTEXT):
        return ""

    # 检查是否有事实性断言
    assertion = _find_assertion_word(text, FACT_ASSERTION_WORDS)
    if not assertion:
        return ""

    # 违反：有断言无来源
    logger.info(
        "jspace-guard: tri-state-gate triggered (assertion='%s')",
        assertion,
    )
    return _TRI_STATE_WARNING % assertion


# ── 钩子 ──────────────────────────────────────────────────────

def _on_post_tool_call(**kwargs):
    """Track tool calls for activation threshold and per-turn rebase detection."""
    tool_name = kwargs.get("tool_name", "")
    if tool_name:
        _state.tool_call_count += 1
        _state.current_turn_tools.add(tool_name)


def _on_transform_llm_output(**kwargs):
    """Scan response for invariant violations; append warnings if found."""
    # cron 会话豁免：无人值守推送直接投给用户，内部审计噪音禁止混入。
    # 两条路径全覆盖：定时调度 platform="cron"；手动 run 的 delegation 会话
    # session_id 形如 "cron_<job_id>_..."。
    if kwargs.get("platform") == "cron" or str(kwargs.get("session_id", "")).startswith("cron_"):
        return ""
    # 未达激活阈值——短任务不需要此门控
    if _state.tool_call_count < ACTIVATION_THRESHOLD:
        return ""

    response_text = kwargs.get("response_text", "")
    if not response_text:
        return ""

    warnings: list = []

    # 不变量 6: 验证须带覆盖
    w = _check_coverage_gate(response_text)
    if w:
        warnings.append(w)

    # 不变量 8: 完成须回读目标
    w = _check_goal_rebase_gate(response_text, _state.current_turn_tools)
    if w:
        warnings.append(w)

    # 不变量 9: 事实性断言须带三态标注
    w = _check_tri_state_gate(response_text)
    if w:
        warnings.append(w)

    if warnings:
        logger.info(
            "jspace-guard: %d invariant(s) violated (calls=%d)",
            len(warnings), _state.tool_call_count,
        )
        return response_text + "".join(warnings)

    return ""


# ── 注册 ──────────────────────────────────────────────────────

def register(ctx):
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    logger.info(
        "jspace-guard v1.1 registered (activation=%d)",
        ACTIVATION_THRESHOLD,
    )
