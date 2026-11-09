"""declare_act_guard — 宣告-动手配对守卫。

上一条 assistant 回复宣告了行动意图（「我建议实施/我准备改/接下来就做/
等我确认后动手」类），且该回复本身没有伴随任何 tool 调用、也不含合法
拍板标记（「必须先确认：」/「须拍板：」）时，注入强制 SOP：要么当轮就
动手，要么明确列为须拍板项并附默认动作——禁裸宣告后把决策成本丢回用户。

判定走 LLM 语义 judge（plugins/_llm_judge.llm_judge_bool），禁词表/正则
（用户 0918 拍板「词表做法很傻逼」：自然语言宣告意图不可穷举）。

判定时机（2026-11-10）：在上一条回复**刚产出的时刻**（post_llm_call）就先判一次，
结果落进 plugins._llm_judge 的内容寻址判定缓存；下一轮 pre_llm_call 读同一段文本
直接命中缓存，关键路径上零等待。缓存未命中（进程刚起、插件热载）时 pre_llm_call
回退为就地判定——行为与以前完全一致，只是不再每次都付一次同步 3~4s。

Contract:
  Preconditions: pre_llm_call kwargs 含 conversation_history（list[dict]）；
    末条 assistant 消息可含 tool_calls 字段（OpenAI 形态）
  Postconditions: 命中且未超封顶时返回 {"context": ...} 注入一次；
    不命中/judge 失败(None=放行)/超封顶返回 None；永不 raise
"""

from __future__ import annotations

import logging
import sys
from importlib import import_module as _import_module
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.declare_act_guard")

_NAMESPACE = "declare_act_guard"
_MAX_INJECTS = 5          # 每会话注入封顶（防死循环噪音）
_MIN_TEXT_LEN = 20        # 极短文本不触发 judge


def _last_assistant(history: list | None) -> dict:
    """取最近一条 assistant 消息 dict，无则空 dict。

    Contract:
      Preconditions: history 为 list 或 None
      Postconditions: 返回 dict（可能为空），绝不 raise
    """
    if not history:
        return {}
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg
    return {}


def _msg_text(msg: dict) -> str:
    """提取消息文本（str content 或多模态 parts 拼接）。

    Contract:
      Preconditions: msg 为 dict
      Postconditions: 返回 str（可为空）
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    return ""


def _has_tool_calls(msg: dict) -> bool:
    """消息是否带 tool_calls / 当轮已动手的结构化证据。

    Contract:
      Preconditions: msg 为 dict
      Postconditions: 返回 bool
    """
    calls = msg.get("tool_calls")
    if isinstance(calls, list) and calls:
        return True
    return bool(msg.get("tool_use"))


def _gate_ok(state: dict, text: str) -> bool:
    """注入闸：未超封顶且非同一违规文本重放。

    Contract:
      Preconditions: state 为本插件 session state dict；text 非空 str
      Postconditions: 纯读判断返回 bool，无副作用
    """
    if state.get("injected_count", 0) >= _MAX_INJECTS:
        return False
    return state.get("last_text", "") != text[:200]


def _judge_reply_side(text: str):
    """共享回复侧合并判定（reply-certainty-checker 的 judge_reply_side）：同一回复文本进程内只判一次。

    运行时插件模块名 = hermes_plugins.<slug>（连字符转下划线，
    见 hermes_cli/plugins.py _directory_module_name）；依次尝试两个命名空间。
    """
    for modname in ("hermes_plugins.reply_certainty_checker",
                    "plugins.reply_certainty_checker"):
        try:
            mod = sys.modules.get(modname) or _import_module(modname)
            return mod.judge_reply_side(text)
        except Exception as e:
            logger.debug("reply-side merge via %s unavailable: %s", modname, e)
    return None


def _judge_violation(text: str) -> Optional[bool]:
    """LLM 语义判定「宣告了却未动手且未列拍板」。失败返回 None。

    与 certainty/boundary/术语漂移同文本合并判定（judge_reply_side 的 violation 键）；
    合并通道不可用时回退本插件独立 judge（llm_judge_bool 原路径）。

    Contract:
      Preconditions: text 为非空 str
      Postconditions: 返回 True/False/None；绝不 raise
    """
    merged = _judge_reply_side(text)
    if merged is not None and merged.get("violation") is not None:
        return merged["violation"]
    from plugins._llm_judge import llm_judge_bool

    system = (
        "你是行为纪律审查器。判断下面这条 AI 助手回复是否违反「宣告-动手配对」纪律。"
        "违反的定义：回复中宣告了具体的行动意图（例如声称建议实施/准备去做/接下来就做/"
        "等某事完成后就动手/不说就按某方案办），并且回复里既没有已经动手的证据"
        "（工具调用、代码修改、命令执行、真实结果输出），也没有把该行动明确列为"
        "「必须先确认/须拍板」事项。只答 JSON：{\"violation\": true/false}。"
        "陈述事实、回答问题、汇报已完成工作、纯提问澄清需求、明确说「须拍板」"
        "并附默认方案的，都不违反。"
    )
    return llm_judge_bool(
        task="declare_act_guard",
        system=system,
        text=text[:4000],
        true_key="violation",
    )


def _build_injection() -> dict[str, str]:
    """构造注入 SOP 文本。

    Contract:
      Preconditions: 无
      Postconditions: 返回 {"context": str}
    """
    return {
        "context": (
            "【宣告-动手配对守卫】你上一条回复宣告了行动意图却没有动手，"
            "也没有把它列为须拍板项。二选一，本条回复内完成：\n"
            "1) 直接动手：立即调用工具执行宣告的事（可逆操作属自我决策域，"
            "办完报结果）；\n"
            "2) 明确须拍板：以「必须先确认：」开头列出该事项+你的建议默认"
            "方案+不回复时的默认动作。\n"
            "禁止裸宣告后停轮把决策成本丢回用户。"
        )
    }


def _on_pre_llm_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    """pre_llm_call 主回调：审上一条 assistant，违规注入 SOP。

    Contract:
      Preconditions: kwargs 含 conversation_history
      Postconditions: 返回 {"context":...} 或 None；永不 raise
    """
    try:
        from plugins._shared_state import get_session_state

        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "") or "__default__"
        state = get_session_state(sid, _NAMESPACE)

        history = kwargs.get("conversation_history") or []
        msg = _last_assistant(history)
        text = _msg_text(msg)
        if len(text.strip()) < _MIN_TEXT_LEN or _has_tool_calls(msg):
            return None  # 极短/已动手的回合不审
        if not _gate_ok(state, text):
            return None

        if _judge_violation(text) is not True:  # False/None 都放行(fail-open)
            return None

        state["injected_count"] = state.get("injected_count", 0) + 1
        state["last_text"] = text[:200]
        return _build_injection()
    except Exception:  # hook 异常绝不影响 agent loop
        logger.exception("declare_act_guard hook failed")
        return None


def _on_post_llm_call(**kwargs: Any) -> None:
    """post_llm_call 预热：在回复刚产出时先判一次，把结论放进判定缓存。

    为什么在这里：这一轮已经结束了，判定的 3~4s 不再落在「用户发下一条消息之后」
    的关键路径上。pre_llm_call 仍按原逻辑判同一段文本，但会直接命中缓存。

    Contract:
      Preconditions: kwargs 含 session_id / assistant_response / conversation_history
      Postconditions: 只预热判定缓存，不注入、不改回复、不写 session state；永不 raise
    """
    try:
        # 文本必须与 pre_llm_call 取到的那一段逐字节一致，否则缓存键不同、预热白做
        history = kwargs.get("conversation_history") or []
        msg = _last_assistant(history)
        text = _msg_text(msg)
        if len(text.strip()) < _MIN_TEXT_LEN:
            return
        if _has_tool_calls(msg):
            return  # 已动手的回合不审（与 pre_llm_call 同一前置）
        _judge_violation(text)
    except Exception:  # 预热失败只影响下一次的等待时长，绝不影响本轮
        logger.debug("declare_act_guard: post_llm_call warm failed", exc_info=True)
    return


def register(ctx: Any) -> None:
    """插件入口：注册 pre_llm_call（判定+注入）与 post_llm_call（判定预热）。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    logger.info(
        "declare_act_guard 已注册——宣告-动手配对语义守卫(pre_llm_call + post_llm_call 判定预热)"
    )
