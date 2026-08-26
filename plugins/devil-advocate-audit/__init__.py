"""devil-advocate-audit plugin — 反方审查检查点（防"我想错"：单视角决策拦截）

pre_llm_call 钩子：用辅助 LLM 语义判断当轮消息是否构成"重大方案定稿/决策承诺"
（不可穷举，禁关键词匹配），且本会话尚未见 delegate_task（反方审查）时，
注入红牌：重大判断必须先经反方视角（delegate 唱反调子代理或用户明示豁免）才可定稿。

设计依据：落实笔记-七心法七功法.md 心法一（反者道之动）——
"插件防的是忘了，防不了想错；防想错=决策与审查分脑"。

判定方式：LLM judge（语义，不可穷举场景唯一可行），非关键词。
成本控制：
- 消息 hash 去重（同一消息只判一次）
- judge 调用每会话硬上限 _MAX_JUDGE_CALLS（防高负荷会话烧 token）
- delegate_task 调用后本会话静默（reviewed=True）
- 红牌每会话最多 _MAX_REMINDERS 次
- fail-open：judge 失败/超时记日志透传，不阻塞主流程

缓存安全：注入 context 不改 system prompt/历史/toolset。

ACTIVATION: config.yaml plugins.enabled 添加 "devil-advocate-audit"。
Set DEVIL_ADVOCATE_AUDIT_DISABLE=1 to turn off.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Any, Dict, Optional

from plugins._llm_judge import llm_judge_bool
from plugins._shared_state import get_session_state
# 用户侧合并判定：challenge/decision 一次调用（唯一出入口收拢）。
# 延迟导入：避免插件间循环依赖；yinyang 未启用时 judge_user_side 不可用，
# 此时本插件回退独立判定（llm_judge_bool 原路径）。
from importlib import import_module as _import_module


def _judge_user_side(message: str):
    # 运行时插件模块名 = hermes_plugins.<slug>（连字符转下划线，
    # 见 hermes_cli/plugins.py _directory_module_name）；依次尝试两个命名空间。
    for modname in ("hermes_plugins.yinyang_restate_guard",
                    "plugins.yinyang_restate_guard"):
        try:
            mod = sys.modules.get(modname) or _import_module(modname)
            return mod.judge_user_side(message)
        except Exception as e:
            logger.warning("user-side merge via %s unavailable: %s", modname, e)
    return None

logger = logging.getLogger(__name__)

_NAMESPACE = "devil_advocate_audit"

_MAX_REMINDERS = 2
_MAX_JUDGE_CALLS = 30
_JUDGE_TIMEOUT = 8.0  # 慢调用截断：fail-open 漏一次提醒 << 用户等 13s

_JUDGE_SYSTEM = (
    "你是决策审查哨兵。判断下面这条会话消息是否构成'重大方案定稿或决策承诺'——"
    "即：即将拍板采用某技术方案/架构选型/上生产部署/模型替换/大规模重构等"
    "影响面大且难回退的决策。只是讨论、提问、调研、汇报进度不算。"
    "只回答 JSON：{\"decision\": true} 或 {\"decision\": false}"
)

_REMINDER = (
    "[DevilAdvocateAudit] 反方审查检查点：检测到重大决策/方案定稿信号，"
    "但本会话尚未见反方视角审查（delegate_task 委派唱反调子代理）。\n"
    "  铁律：决策与审查必须分脑——单视角定稿=未审计。\n"
    "  现在做：委派一个只找漏洞的反方子代理审这个方案，或请用户明示豁免。\n"
    "若本次已在审查或用户已豁免，忽略本条。"
)


def _plugin_disabled() -> bool:
    return os.environ.get("DEVIL_ADVOCATE_AUDIT_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _count(sid: str, key: str = "count") -> int:
    return int(get_session_state(sid, _NAMESPACE).get(key, 0))


def _seen_hash(sid: str) -> set:
    """取会话已判消息 hash 集合（惰性建）。

    Contract:
      Postconditions: 返回可变 set 且已写回 session state（后续 add 持久生效）
    """
    st = get_session_state(sid, _NAMESPACE)
    s = st.get("seen")
    if not isinstance(s, set):
        s = set()
        st["seen"] = s
    return s


def _is_major_decision(text: str) -> Optional[bool]:
    """LLM judge 语义判定；失败返回 None（fail-open，视为非决策）。

    Contract:
      Preconditions: text is non-empty str
      Postconditions: 返回 True/False/None；绝不 raise
    """
    assert text, "text must be non-empty"
    return llm_judge_bool(
        task="devil_advocate_audit",
        system=_JUDGE_SYSTEM,
        text=text,
        timeout=_JUDGE_TIMEOUT,
    )


def on_post_tool_call(**kwargs) -> None:
    """delegate_task 调用 = 反方审查已做 → 本会话静默。

    Contract:
      Postconditions: 仅当 tool_name 属于 delegate 集合时写 reviewed 标记
    """
    tool_name = str(kwargs.get("tool_name", ""))
    if tool_name in {"delegate_task", "delegate"}:
        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
        if sid:
            get_session_state(sid, _NAMESPACE)["reviewed"] = True


def on_pre_llm_call(**kwargs) -> Optional[Dict[str, Any]]:
    """语义判定重大决策 + 未见反方审查 → 注入红牌。fail-open。"""
    try:
        if _plugin_disabled():
            return None
        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
        if not sid:
            return None
        st = get_session_state(sid, _NAMESPACE)
        if st.get("reviewed"):
            return None
        if _count(sid) >= _MAX_REMINDERS:
            return None
        if _count(sid, "judge_calls") >= _MAX_JUDGE_CALLS:
            return None
        text = str(kwargs.get("user_message", "") or "")
        if not text.strip():
            return None
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        if h in _seen_hash(sid):
            return None
        _seen_hash(sid).add(h)
        st["judge_calls"] = _count(sid, "judge_calls") + 1
        # 合并判定：优先走 yinyang 的多键一次调用；两插件各自 hash 去重仍生效，
        # 但合并结果同时供 yinyang 消费（见 yinyang on_pre_llm_call 侧缓存）。
        merged = _judge_user_side(text)
        if merged is None:
            # 合并通道不可用（yinyang 未启用等）→ 回退原独立判定
            if _is_major_decision(text) is not True:
                return None
        elif merged.get("decision") is not True:
            return None
        st["count"] = _count(sid) + 1
        return {"context": _REMINDER}
    except Exception as e:
        logger.warning("devil-advocate-audit hook failed: %s", e,
                       exc_info=True)
        return None


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("devil-advocate-audit registered")
