"""
no-pushback plugin — hard enforcement against lazy pushback patterns.

Two-layer interception:
  Layer 1 (transform_llm_output): Scans LLM output text for pushback question
    patterns. When detected, appends a correction directive so the model
    rewrites the offending passage.

  Layer 2 (pre_tool_call): Blocks `clarify` tool calls whose question matches
    pushback patterns. Returns a block message instructing the model to decide
    and act on its own.

Patterns matched (Chinese):
  - 要不要 / 是否需要 / 需不需要
  - 要不要我 / 需要我
  - 请确认 / 请你确认
  - 需要你做 / 需用户做
  - 要不要...吗 / ...吗（在推责上下文中）

Allowed exceptions:
  - Clarify calls for genuinely open-ended questions (no pushback pattern)
  - Output containing these phrases in quoted/code context
"""

import logging
import re

logger = logging.getLogger(__name__)

# Pushback patterns — each is a compiled regex.
# Order matters: longer/more specific patterns first.
_PATTERNS = [
    re.compile(r"要不要我[^。，\n]{0,30}[？?]", re.DOTALL),
    re.compile(r"需不需要[^。，\n]{0,30}[？?]", re.DOTALL),
    re.compile(r"是否需要[^。，\n]{0,30}[？?]", re.DOTALL),
    re.compile(r"要不要[^。，\n]{0,30}[？?]", re.DOTALL),
    re.compile(r"请确认[一下]", re.DOTALL),
    re.compile(r"需要你[^。，\n]{0,20}做", re.DOTALL),
    re.compile(r"需用户[^。，\n]{0,20}", re.DOTALL),
]

# Correction message appended when pushback is detected in output.
_TRANSFORM_SUFFIX = (
    "\n\n[NO-PUSHBACK] 你的回复包含推责式询问（\"要不要/是否需要/请确认\"）。"
    "违反零询问铁律。删掉询问，自己做判断直接执行。"
)

# Block message returned when clarify tool is blocked.
_BLOCK_MESSAGE = (
    "[NO-PUSHBACK BLOCK] clarify 工具调用被拦截：问题内容匹配推责模式。"
    "你自己做判断直接执行，不要问用户。只有不可逆操作（删除/覆盖/发布）"
    "且无回退方案时才允许询问。"
)


def _matches_pushback(text: str) -> bool:
    """Return True if text contains a pushback pattern."""
    if not text:
        return False
    for pat in _PATTERNS:
        if pat.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 1: transform_llm_output
# ---------------------------------------------------------------------------

def _on_transform_llm_output(**kwargs) -> str:
    """Scan LLM output for pushback patterns; append correction if found."""
    response_text = kwargs.get("response_text", "")
    if not response_text:
        return ""  # empty → unchanged

    if _matches_pushback(response_text):
        logger.warning(
            "no-pushback: detected pushback pattern in output (len=%d)",
            len(response_text),
        )
        return response_text + _TRANSFORM_SUFFIX

    return ""  # None/empty → unchanged


# ---------------------------------------------------------------------------
# Layer 2: pre_tool_call (clarify guard)
# ---------------------------------------------------------------------------

def _on_pre_tool_call(**kwargs) -> dict:
    """Block clarify calls whose question matches pushback patterns."""
    tool_name = kwargs.get("tool_name", "")
    if tool_name != "clarify":
        return {}  # not a clarify call, pass through

    args = kwargs.get("args", {})
    if not isinstance(args, dict):
        return {}

    question = args.get("question", "")
    if not question:
        return {}

    if _matches_pushback(question):
        logger.warning(
            "no-pushback: blocked clarify call with pushback question: %s",
            question[:100],
        )
        return {"action": "block", "message": _BLOCK_MESSAGE}

    return {}  # legitimate clarify, pass through


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx):
    """Plugin entry point."""
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    logger.info("no-pushback plugin registered (transform_llm_output + pre_tool_call)")
