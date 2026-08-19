"""yinyang-restate-guard plugin — 阴阳冲和检查点（心法三：复述后再反驳）

pre_llm_call 钩子：
用 LLM judge 语义判定用户消息是否在质疑/反驳/纠正（人话语义不可穷举，
禁关键词/正则匹配），命中时注入工作守则：反驳前必须先用一两句复述用户
立场至对方视角成立，再指出分歧点。会话内最多提醒 3 次。

成本控制：消息 hash 去重（同一消息只判一次）+ judge 调用每会话硬上限
30 次 + fail-open（judge 失败/超时视为非质疑，不阻塞主流程）。

设计依据：落实笔记-七心法七功法.md 心法三——"反驳任何立场前，先复述
到对方认可"。阴阳冲和：对立面不是敌人是对配偶，先让两股力量互相听清。

缓存安全：context 注入 user message（pre_llm_call 标准通道），不改
system prompt、不改历史、不换 toolset。

ACTIVATION: 需在 config.yaml plugins.enabled 添加 "yinyang-restate-guard"。
Set YINYANG_RESTATE_GUARD_DISABLE=1 to turn off.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Optional

from plugins._llm_judge import llm_judge_bool
from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "yinyang_restate_guard"

_MAX_REMINDERS = 3
_MAX_JUDGE_CALLS = 30

_JUDGE_SYSTEM = (
    "你是对话信号哨兵。判断下面这条用户消息是否在质疑、反驳或纠正 AI 的"
    "某个说法/结论/行为（含直接指出错误、表达不认同、要求纠正）。"
    "单纯提问、补充信息、闲聊、下达新任务不算。"
    "只回答 JSON：{\"challenge\": true} 或 {\"challenge\": false}"
)

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


def is_challenge(message: str) -> Optional[bool]:
    """LLM judge 语义判定用户消息是否含质疑/反驳信号；None=fail-open。

    Contract:
      Preconditions: message 为 str（可为空）
      Postconditions: 空串 → None；否则返回 True/False/None，绝不 raise
    """
    if not message:
        return None
    return llm_judge_bool(
        task="yinyang_restate_guard",
        system=_JUDGE_SYSTEM,
        text=message,
        true_key="challenge",
        timeout=8.0,
    )


def _state(sid: str) -> Dict[str, Any]:
    return get_session_state(sid, _NAMESPACE)


def on_pre_llm_call(**kwargs) -> Optional[Dict[str, Any]]:
    """语义判定质疑 + 未达上限 → 注入复述守则。fail-open。"""
    try:
        if _plugin_disabled():
            return None
        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
        user_message = str(kwargs.get("user_message", "") or "")
        if not user_message.strip():
            return None
        st = _state(sid)
        if int(st.get("count", 0)) >= _MAX_REMINDERS:
            return None
        if int(st.get("judge_calls", 0)) >= _MAX_JUDGE_CALLS:
            return None
        h = hashlib.sha1(user_message.encode("utf-8")).hexdigest()[:16]
        seen = st.get("seen")
        if not isinstance(seen, set):
            seen = set()
            st["seen"] = seen
        if h in seen:
            return None
        seen.add(h)
        st["judge_calls"] = int(st.get("judge_calls", 0)) + 1
        if is_challenge(user_message) is not True:
            return None
        st["count"] = int(st.get("count", 0)) + 1
        return {"context": _REMINDER}
    except Exception as e:
        logger.warning("yinyang-restate-guard hook failed: %s", e,
                       exc_info=True)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("yinyang-restate-guard registered")
