"""no-bypass plugin.

Intercepts lazy bypass patterns when the agent discovers a bug or blocker
and chooses to work around it instead of fixing it.

Two-layer defense (same architecture as no-pushback):
  Layer 1: transform_llm_output — scans free-text output for bypass language
           and appends a correction directive.
  Layer 2: pre_tool_call — blocks clarify/send_message if the agent tries to
           tell the user about a workaround instead of fixing the problem.

Trigger patterns:
  - "可以通过 CLI 绕过" / "绕过 API 层" / "跳过 API"
  - "不修这个" / "先用 workaround" / "暂时跳过"
  - "先绕过这个问题" / "后面再修"
  - Discovering a bug then describing an alternative path instead of fixing it
"""

import re
import logging

logger = logging.getLogger("hermes.plugin.no-bypass")

# ── Bypass patterns (Layer 1: free-text scan) ──────────────────────────

_BYPASS_PATTERNS = [
    # Direct "bypass" statements
    (
        re.compile(r"(?:可以|能)通过\s*\S{0,20}(?:绕过|跳过|避开)(?:API|CLI|层|接口|直接)", re.IGNORECASE),
        '发现了bug却选择绕过（"通过CLI绕过/跳过API层"）',
    ),
    (
        re.compile(r"(?:绕过|跳过|避开)(?:这个|该|此)?(?:问题|bug|错误|限制|层)", re.IGNORECASE),
        '选择绕过而非修复（"绕过这个问题"）',
    ),
    # Workaround instead of fix
    (
        re.compile(r"(?:先用|先用|暂用|改用)\s*(?:workaround|替代|绕路|绕过)", re.IGNORECASE),
        '选择workaround而非根因修复',
    ),
    (
        re.compile(r"(?:后面|以后|稍后|回头|下次)(?:再|来)?(?:修|修复|处理|解决)", re.IGNORECASE),
        '推迟修复（"后面再修"）——违反零妥协原则',
    ),
    # "Not fixing this" + describing alternative
    (
        re.compile(r"(?:不修|不修这个|跳过修复)(?:[，。,\n])", re.IGNORECASE),
        '明确拒绝修复（"不修这个"）',
    ),
    # "Can verify via CLI directly" when a code bug was just identified
    (
        re.compile(r"(?:可以|能)\s*(?:直接)?通过\s*CLI\s*(?:直接\s*)?(?:验证|测试|确认|跑)", re.IGNORECASE),
        '发现代码bug后转而描述CLI绕过路径',
    ),
]

# ── Correction directive appended to output ────────────────────────────

_CORRECTION = (
    "\n\n[NO-BYPASS] 你的回复包含绕过模式——发现了 bug/问题却选择替代路径而非修复。"
    "违反零妥协原则。回到刚才发现的问题，修复根因，不要绕过。"
    "绕路成本远高于修根因，绕路是偷懒不是务实。"
)

# ── Layer 1: transform_llm_output ──────────────────────────────────────


def _on_transform_llm_output(**kwargs) -> str:
    text = kwargs.get("response_text", "")
    if not text:
        return ""
    for pattern, reason in _BYPASS_PATTERNS:
        match = pattern.search(text)
        if match:
            logger.info(
                "no-bypass: bypass pattern detected (%s): %s",
                reason, match.group()[:80],
            )
            return text + _CORRECTION
    return ""


# ── Layer 2: pre_tool_call ─────────────────────────────────────────────

_BLOCK_TOOLS = {"clarify", "send_message"}


def _on_pre_tool_call(**kwargs) -> dict:
    tool_name = kwargs.get("tool_name", "")
    if tool_name not in _BLOCK_TOOLS:
        return {}
    args = kwargs.get("args", {})
    # For clarify: check the question text
    # For send_message: check the message text
    text = ""
    if tool_name == "clarify":
        text = args.get("question", "")
        choices = args.get("choices", [])
        if choices:
            text += " " + " ".join(str(c) for c in choices)
    elif tool_name == "send_message":
        text = args.get("message", "")
    if not text:
        return {}
    for pattern, reason in _BYPASS_PATTERNS:
        if pattern.search(text):
            logger.info(
                "no-bypass: blocking %s — bypass detected (%s)",
                tool_name, reason,
            )
            return {
                "action": "block",
                "message": (
                    "[NO-BYPASS BLOCK] {} 调用被拦截：内容包含绕过模式（{}）。"
                    "你发现了 bug 就必须修复根因，不允许绕过或描述替代路径。"
                    "去掉绕过建议，回到问题本身，修复它。"
                ).format(tool_name, reason),
            }
    return {}


# ── Registration ───────────────────────────────────────────────────────


def register(ctx):
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    logger.info("no-bypass plugin registered")
