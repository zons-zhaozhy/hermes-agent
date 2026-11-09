"""brief_before_act — 先亮牌后动手守卫。

契约 SSOT = ~/.hermes/memories/soul.md「先亮牌后动手」节：改代码/改文案/
对外产物，动手前必先输出理解/方案/决策/动作四要素，无大小豁免。

本守卫做事后检测：上一回合窗口（最近一条 user 消息之后）内出现了写操作
（patch / write_file 的 tool_call），但其前 assistant 文本没有亮牌标记
（四要素命中 >=3 或「亮牌」字样）时，向下一轮注入 SOP 要求补亮牌。

设计取舍（相对 declare_act_guard 的 LLM 语义判定）：
- 标记检测为结构化 str 方法，零 LLM 延迟（_llm_judge 实测每守卫 5~10s）；
- 亮牌标记是用户令钦定的固定四词，天然可枚举，不属于「自然语言意图不可
  穷举」场景，词表判定不适用 0918 拍板的禁令对象。

Contract:
  Preconditions: pre_llm_call kwargs 含 conversation_history（list[dict]），
    assistant 消息可含 tool_calls 字段（OpenAI 形态，name 或 function.name）
  Postconditions: 命中且未超封顶时返回 {"context": ...} 注入一次；
    不命中/超封顶返回 None；永不 raise
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("hermes.plugins.brief_before_act")

_NAMESPACE = "brief_before_act"
_MAX_INJECTS = 5          # 每会话注入封顶（防噪音循环）
_MARKERS = ("理解", "方案", "决策", "动作")
# 写工具语义 owner = plugins/guards/tool_classification（C2 收敛，禁各持字面量）
from plugins.guards.tool_classification import FILE_WRITE_TOOLS as _WRITE_TOOLS


def _msg_text(msg: dict) -> str:
    """提取消息文本（str content 或多模态 parts 拼接）。

    Contract:
      Preconditions: msg 为 dict
      Postconditions: 返回 str（可为空）；绝不 raise
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    return ""


def _tool_names(msg: dict) -> list[str]:
    """提取消息 tool_calls 里的工具名（name 或 function.name 两形态）。

    Contract:
      Preconditions: msg 为 dict
      Postconditions: 返回 list[str]（可为空）
    """
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return []
    names: list[str] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name and isinstance(c.get("function"), dict):
            name = c["function"].get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _has_brief_marker(text: str) -> bool:
    """亮牌判定：四要素词命中 >=3，或明示「亮牌」。

    Contract:
      Preconditions: text 为 str
      Postconditions: 纯读判断返回 bool
    """
    if "亮牌" in text:
        return True
    hits = sum(1 for m in _MARKERS if m in text)
    return hits >= 3


def _find_violation(history: list) -> bool:
    """扫「最近 user 消息之后」窗口：写操作出现前无亮牌标记 = 违规。

    Contract:
      Preconditions: history 为 list[dict]
      Postconditions: 纯读判断返回 bool；无 user 消息（无从界定新任务）→ False
    """
    last_user = -1
    for i, msg in enumerate(history):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user = i
    if last_user < 0:
        return False
    text_before_write = ""
    for msg in history[last_user + 1:]:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        names = _tool_names(msg)
        wrote = any(n in _WRITE_TOOLS for n in names)
        if wrote:
            return not _has_brief_marker(text_before_write)
        text_before_write += _msg_text(msg) + "\n"
    return False


def _build_injection() -> dict[str, str]:
    """构造注入 SOP 文本。

    Contract:
      Preconditions: 无
      Postconditions: 返回 {"context": str}
    """
    return {
        "context": (
            "【先亮牌后动手守卫】上一回合在输出理解/方案/决策/动作四要素之前"
            "就直接执行了写操作（patch/write_file），违反「先亮牌后动手」契约"
            "（SSOT=soul.md）。本轮起：动手改代码/改文案/对外产物前，先亮出"
            "四要素——理解（怎么理解的）、方案（打算怎么做）、决策（关键选择"
            "及理由）、动作（步骤清单）——再执行；亮牌后 3-5 分钟用户未回复"
            "即按建议方案自动推进，禁止干等。"
        )
    }


def _on_pre_llm_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    """pre_llm_call 主回调：检测上回合裸写，注入补亮牌 SOP。

    Contract:
      Preconditions: kwargs 含 conversation_history
      Postconditions: 返回 {"context": ...} 或 None；永不 raise
    """
    try:
        from plugins._shared_state import get_session_state

        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "") or "__default__"
        state = get_session_state(sid, _NAMESPACE)
        if state.get("injected_count", 0) >= _MAX_INJECTS:
            return None

        history = kwargs.get("conversation_history") or []
        if not _find_violation(history):
            return None

        state["injected_count"] = state.get("injected_count", 0) + 1
        return _build_injection()
    except Exception:  # hook 异常绝不影响 agent loop
        logger.exception("brief_before_act hook failed")
        return None


def register(ctx: Any) -> None:
    """插件入口：注册 pre_llm_call 单钩。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("brief_before_act 已注册——先亮牌后动手守卫(pre_llm_call)")
