"""yinyang-restate-guard plugin — 阴阳冲和检查点（心法三：复述后再反驳）

pre_llm_call 钩子：
用户消息含质疑/反驳/纠正信号（不对/错了/瞎扯/你是不是/质疑类）时，
注入一条工作守则：反驳前必须先用一两句复述用户立场至对方视角成立，
再指出分歧点。会话内最多提醒 3 次（之后视为已内化，不再打扰）。

设计依据：落实笔记-七心法七功法.md 心法三——"反驳任何立场前，先复述
到对方认可"。阴阳冲和：对立面不是敌人是对配偶，先让两股力量互相听清。

触发面刻意窄：只看当轮 user_message，不看历史（避免误判）；信号词为
有限集合；达到会话上限即静默。

缓存安全：context 注入 user message（pre_llm_call 标准通道），不改
system prompt、不改历史、不换 toolset。

ACTIVATION: 需在 config.yaml plugins.enabled 添加 "yinyang-restate-guard"。
Set YINYANG_RESTATE_GUARD_DISABLE=1 to turn off.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "yinyang_restate_guard"

# 质疑/反驳信号（finite set；宁窄勿宽——漏报只是少一次提醒，误报每次打扰）
_CHALLENGE_SIGNALS = re.compile(
    r"(不对|不是这样|错了|搞错|瞎扯|胡说|乱来|你是不是|怎么说得|"
    r"自相矛盾|前言不搭|矛盾啊|讲不通|说不通|推翻|质疑|反对)",
)

_MAX_REMINDERS = 3

_REMINDER = (
    "[YinYangRestateGuard] 阴阳冲和检查点：用户本轮在质疑/纠正你。\n"
    "  规则：反驳前必须先用一两句复述对方的立场——复述到从对方视角看成立，"
    "再指出分歧在哪。禁止跳过复述直接反驳。\n"
    "  若本轮你确实错了：先承认具体错在哪一句，再修正——承认也是复述的一种。\n"
    "  若确认对方误解：复述其误解来源，再澄清。已内化则忽略本条。"
)


def _plugin_disabled() -> bool:
    return os.environ.get("YINYANG_RESTATE_GUARD_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def is_challenge(message: str) -> bool:
    """用户消息是否含质疑/反驳信号。

    Contract:
      Preconditions: message 为 str（可为空）
      Postconditions: 返回 bool；空串/无信号词 → False
    """
    if not message:
        return False
    return bool(_CHALLENGE_SIGNALS.search(message))


def _reminder_count(sid: str) -> int:
    return int(get_session_state(sid, _NAMESPACE).get("count", 0))


def on_pre_llm_call(**kwargs) -> Optional[Dict[str, Any]]:
    """用户质疑 + 未达提醒上限 → 注入复述守则。"""
    if _plugin_disabled():
        return None
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    user_message = kwargs.get("user_message") or ""
    if not is_challenge(str(user_message)):
        return None
    if _reminder_count(sid) >= _MAX_REMINDERS:
        return None
    get_session_state(sid, _NAMESPACE)["count"] = _reminder_count(sid) + 1
    return {"context": _REMINDER}


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("yinyang-restate-guard registered")
