"""dao-recurring-error-guard plugin — 道生法检查点（心法四：重复即成规）

post_tool_call + pre_llm_call 钩子：
按错误指纹（工具名+错误类别词聚合）计数会话内同类失败。同一指纹
第 3 次出现且尚未见沉淀动作（write_file/patch/skill_manage/commit）时，
pre_llm_call 注入提示：第三次必须成规则/skill/脚本，禁止再手工处理。

设计依据：落实笔记-七心法七功法.md 心法四——"同一问题出现第二次，
第三次必须是规则/skill/plugin，不许再手工处理"（道生法）。

聚合策略（刻意保守，宁可漏报不误报）：
- 只对 error/exception 类工具结果计数
- 指纹 = 工具名 + 从错误文本提取的有限类别词（ImportError/SyntaxError/
  Timeout/ConnectionError/401/403/404/500/1064/exit code 非零），
  不做自由文本哈希——与 db-safety/negative-conclusion-guard 的
  finite-set 思路一致
- 沉淀动作一旦出现即清零该会话提醒（规则已产出）

缓存安全：pre_llm_call 注入 context，不改 system prompt/历史/toolset。

ACTIVATION: 需在 config.yaml plugins.enabled 添加 "dao-recurring-error-guard"。
Set DAO_RECURRING_ERROR_GUARD_DISABLE=1 to turn off.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "dao_recurring_error_guard"

# 有限错误类别词表 → 归一化类别（finite set，非自由枚举）
_ERROR_CLASSES = [
    (re.compile(r"ImportError|ModuleNotFoundError", re.IGNORECASE), "ImportError"),
    (re.compile(r"SyntaxError", re.IGNORECASE), "SyntaxError"),
    (re.compile(r"Timeout|timed?\s*out|ReadTimeout", re.IGNORECASE), "Timeout"),
    (re.compile(r"ConnectionError|connection\s+refused|no\s+such\s+host", re.IGNORECASE), "ConnectionError"),
    (re.compile(r"\b401\b|Unauthorized", re.IGNORECASE), "AuthError-401"),
    (re.compile(r"\b403\b|Forbidden", re.IGNORECASE), "Forbidden-403"),
    (re.compile(r"\b404\b|not\s+found", re.IGNORECASE), "NotFound-404"),
    (re.compile(r"\b500\b|Internal\s+Server\s+Error", re.IGNORECASE), "ServerError-500"),
    (re.compile(r"\b1064\b", re.IGNORECASE), "SQL-Syntax-1064"),
    (re.compile(r"AssertionError", re.IGNORECASE), "AssertionError"),
]

# 沉淀动作：规则/skill/脚本已产出 → 清零
_CODIFICATION_TOOLS = {"write_file", "patch", "skill_manage", "terminal"}
_CODIFICATION_TERMINAL = re.compile(
    r"(\bgit\s+commit\b|>.*\.sh\b|hermes\s+(config|skill|cron)\b)", re.IGNORECASE
)

_THRESHOLD = 3

_REMINDER = (
    "[DaoRecurringErrorGuard] 道生法检查点：本会话「{tool}×{cls}」类错误已出现 {n} 次，"
    "且尚未见沉淀动作。\n"
    "  铁律：同一问题出现第二次，第三次必须成规则——写脚本/skill/配置修复类，"
    "禁止再逐个手工处理。\n"
    "  现在停下来回答：这个错误的共性根因是什么？写成什么规则？"
    "若本次调用已在沉淀，忽略本条。"
)


def _plugin_disabled() -> bool:
    return os.environ.get("DAO_RECURRING_ERROR_GUARD_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def classify_error(text: str) -> Optional[str]:
    """从错误文本提取归一化错误类别；无匹配返回 None。

    Contract:
      Preconditions: text 为 str（可为空）
      Postconditions: 返回 _ERROR_CLASSES 中首个命中的类别名，或 None
    """
    if not text:
        return None
    for pattern, cls in _ERROR_CLASSES:
        if pattern.search(text):
            return cls
    return None


def _counts(sid: str) -> Dict[str, int]:
    return dict(get_session_state(sid, _NAMESPACE).get("counts", {}))


def _codified(sid: str) -> bool:
    return bool(get_session_state(sid, _NAMESPACE).get("codified", False))


def on_post_tool_call(**kwargs) -> None:
    """错误指纹计数 + 沉淀动作检测。"""
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = str(kwargs.get("tool_name", ""))
    args = kwargs.get("args") or {}
    result = kwargs.get("result") or {}
    status = str(kwargs.get("status", ""))

    # 沉淀动作检测
    if tool_name in _CODIFICATION_TOOLS:
        if tool_name == "terminal":
            cmd = str(args.get("command", ""))
            if _CODIFICATION_TERMINAL.search(cmd):
                get_session_state(sid, _NAMESPACE)["codified"] = True
        else:
            get_session_state(sid, _NAMESPACE)["codified"] = True

    # 错误计数：status=error 或结果文本可归类
    err_text = ""
    if isinstance(result, dict):
        err_text = str(result.get("error") or result.get("output") or "")
    elif result is not None:
        err_text = str(result)
    cls = classify_error(err_text) if status == "error" or err_text else None
    if cls is None:
        return
    key = f"{tool_name}×{cls}"
    st = get_session_state(sid, _NAMESPACE)
    counts = st.get("counts", {})
    counts[key] = counts.get(key, 0) + 1
    st["counts"] = counts


def on_pre_llm_call(**kwargs) -> Optional[Dict[str, Any]]:
    """达到阈值且未见沉淀 → 注入道生法提醒。"""
    if _plugin_disabled():
        return None
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    if _codified(sid):
        return None
    over = {k: v for k, v in _counts(sid).items() if v >= _THRESHOLD}
    if not over:
        return None
    key, n = sorted(over.items(), key=lambda kv: -kv[1])[0]
    tool, cls = key.split("×", 1)
    return {"context": _REMINDER.format(tool=tool, cls=cls, n=n)}


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("dao-recurring-error-guard registered")
