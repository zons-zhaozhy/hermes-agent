"""
persistence-enforcer plugin v2.0 — 强制复杂任务创建 TODO + 结构化持久化。

四层拦截——从劝导到强制：

Layer 1 (post_tool_call): 追踪工具调用次数和类型。
Layer 2 (pre_llm_call): ≥5 次调用 + 复杂关键词 + 无 TODO → LLM 调用前注入提醒。
Layer 3 (pre_tool_call): ≥10 次调用 + 无 TODO + 无持久化 → 硬拦截 write_file/patch。
  → 只放行：只读工具 + todo + 持久化工具。
  → 一旦创建 TODO → 解除拦截。
Layer 4 (transform_llm_output): LLM 输出含结构化分析 + 未持久化 → 追加提醒。
  原始回复完整保留，只追加。

设计原则：
  - 提醒先于拦截（5 次提醒 → 10 次拦截）
  - 拦截不是目的——让 agent 创建 TODO 后立即放行
  - 只读工具永不禁用（agent 总要能调查）
"""

import logging

logger = logging.getLogger(__name__)

# ── 阈值 ──────────────────────────────────────────────────────
WARN_THRESHOLD = 5     # 提醒阈值
BLOCK_THRESHOLD = 10   # 硬拦截阈值

# write_file 是可持久化工具，但在无 TODO 时会被拦截
# PERSIST_TRACK 用于 post_tool_call 追踪——包含 write_file
PERSIST_TRACK = frozenset({"write_file", "skill_manage", "memory"})
# 拦截目标：只挡代码编辑工具（write_file, patch）
# terminal/delegate/browser/read 等全部放行——agent 需要它们做调查
BLOCKED_TOOLS = frozenset({"write_file", "patch"})

COMPLEX_TASK_KEYWORDS = frozenset({
    "审计", "审查", "全量", "全面", "深度", "重构", "架构",
    "audit", "review", "refactor", "migration", "迁移",
})

_TODO_REMINDER = (
    "\n[PERSISTENCE-ENFORCER] {count} 次工具调用，尚未创建 TODO 列表。\n"
    "大的、复杂的、耗时长的任务必须在动手前用 `todo` 工具创建任务列表。\n"
    "每个子任务完成后必须立即将结果持久化（skill_manage/write_file）。\n"
    "不要等到最后再汇总——上下文压缩会吞掉内存中的结果。"
)

_BLOCK_MESSAGE = (
    "[PERSISTENCE-ENFORCER BLOCK] {count} 次工具调用，无 TODO、无持久化。\n"
    "{tool_name} 已被拦截。\n\n"
    "在用 `write_file`/`patch` 编辑代码之前，你必须：\n"
    "1. 调用 `todo` 创建任务列表\n"
    "2. 将已完成的分析结果用 `skill_manage` 或 `write_file` 持久化\n\n"
    "只读工具不受限制——你仍可以调查。创建 TODO 后立即解封。"
)

_ANALYSIS_PERSIST_REMINDER = (
    "\n\n[PERSISTENCE-ENFORCER] 上一条回复包含结构化分析结果"
    "（{count} 次工具调用）。\n"
    "立即调用 `skill_manage` 或 `write_file` 将分析结果持久化到文件系统。\n"
    "上下文压缩会把没有落盘的内容全部丢弃。"
)


# ── 状态（per-session，进程内） ────────────────────────────────

class _SessionState:
    def __init__(self):
        self.tool_call_count = 0
        self.todo_called = False
        self.persist_called = False
        self.todo_reminded = False
        self.output_reminded = False
        self._last_response_reminded = ""


_state = _SessionState()


def _is_complex_task(conversation_messages: list[dict]) -> bool:
    for msg in conversation_messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            for kw in COMPLEX_TASK_KEYWORDS:
                if kw in content:
                    return True
    return False


def _is_analysis_output(text: str) -> bool:
    if len(text) < 300:
        return False
    lines = text.split("\n")
    heading_count = sum(1 for l in lines if l.strip().startswith("#"))
    evidence_markers = sum(
        1 for l in lines
        if any(m in l for m in ("[实测]", "[文档]", "[推断]", "|", "├", "└", "=="))
    )
    return heading_count >= 2 or evidence_markers >= 3


# ── Layer 1: post_tool_call ────────────────────────────────────

def _on_post_tool_call(**kwargs):
    tool_name = kwargs.get("tool_name", "")
    if not tool_name:
        return

    _state.tool_call_count += 1

    if tool_name == "todo":
        _state.todo_called = True
        logger.info("persistence-enforcer: TODO created, block lifted")

    if tool_name in PERSIST_TRACK:
        _state.persist_called = True


# ── Layer 2: pre_llm_call (提醒) ───────────────────────────────

def _on_pre_llm_call(**kwargs):
    conversation_history = kwargs.get("conversation_history", [])
    if not conversation_history:
        return {}

    if _state.tool_call_count < WARN_THRESHOLD:
        return {}
    if _state.todo_called:
        return {}
    if _state.todo_reminded:
        return {}
    if not _is_complex_task(conversation_history):
        return {}

    _state.todo_reminded = True
    logger.info(
        "persistence-enforcer: pre_llm — TODO reminder (calls=%d)",
        _state.tool_call_count,
    )
    return {"context": _TODO_REMINDER.format(count=_state.tool_call_count)}


# ── Layer 3: pre_tool_call (硬拦截) ────────────────────────────

def _on_pre_tool_call(**kwargs):
    """达到硬拦截阈值 → 阻止 write_file/patch，直到创建 TODO。"""
    tool_name = kwargs.get("tool_name", "")
    if not tool_name:
        return {}

    # 不满足拦截条件：通过
    if _state.tool_call_count < BLOCK_THRESHOLD:
        return {}
    if _state.todo_called:
        return {}
    if _state.persist_called:
        return {}

    # 只拦截代码编辑工具，其他全部放行
    if tool_name not in BLOCKED_TOOLS:
        return {}

    logger.warning(
        "persistence-enforcer: BLOCKING %s (calls=%d, no TODO, no persist)",
        tool_name, _state.tool_call_count,
    )
    return {
        "action": "block",
        "message": _BLOCK_MESSAGE.format(
            count=_state.tool_call_count,
            tool_name=tool_name,
        ),
    }


# ── Layer 4: transform_llm_output (分析提醒) ───────────────────

def _on_transform_llm_output(**kwargs):
    if _state.persist_called:
        return ""
    if _state.tool_call_count < WARN_THRESHOLD:
        return ""
    if _state.output_reminded:
        return ""

    response_text = kwargs.get("response_text", "")
    if not response_text:
        return ""
    if not _is_analysis_output(response_text):
        return ""
    if response_text == _state._last_response_reminded:
        return ""

    _state.output_reminded = True
    _state._last_response_reminded = response_text

    reminder = _ANALYSIS_PERSIST_REMINDER.format(count=_state.tool_call_count)
    logger.info(
        "persistence-enforcer: transform_llm_output — persist reminder (calls=%d)",
        _state.tool_call_count,
    )
    return response_text + reminder


# ── 注册 ──────────────────────────────────────────────────────

def register(ctx):
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    logger.info(
        "persistence-enforcer v2.0 registered (warn=%d, block=%d)",
        WARN_THRESHOLD, BLOCK_THRESHOLD,
    )
