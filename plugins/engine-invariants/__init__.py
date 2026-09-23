"""engine-invariants plugin — 引擎正确性不变量注册表。

借鉴 deepseek-harness 的 invariants companion 模式：每个引擎子系统声明
自己拥有的运行时不变量，独立的检验器定期核对。dsh 用它守护引擎自身
（inbox FIFO 守恒、压缩锁配对、流语法），Hermes 的对应缺口是消息序列
不变量——修复器（sanitize_api_messages 等）很强但静默，违反信号的
暴露面为零。

本插件填补检测层：post_llm_call 时对最终 conversation_history 跑
不变量注册表。检测项（与修复器的修复语义一一对应，但不重复修复）：

1. tool_pairing — 每个 tool 消息的 tool_call_id 必须命中之前某个
   assistant.tool_calls 条目；每个 assistant.tool_calls 条目之后
   必须跟到对应 tool 结果（除非轮次被用户中断——post_llm_call 的
   conversation_history 是完成轮，不应有悬挂调用）。
2. role_alternation — user/assistant 交替合法性：连续同角色裸消息
   需带合并语义标记。修复器的 merge 修复对应此检测。
3. orphan_tool_results — 无对应 tool_call 的 tool 消息。

检测到违反只记结构化日志 + 会话累计计数（get_session_state），
不修复不阻断——修复归修复器，信号归检测器。违反计数可被
observability/judge 体系消费。

fail-open：任何异常吞掉只 warning，绝不破坏正常轮次。

缓存安全：不改 system prompt、不改历史消息、不换 toolset。
post_llm_call 在回合结束后触发，对 prompt caching 零影响。

ACTIVATION: 需在 config.yaml plugins.enabled 中添加 "engine-invariants"。
Set ENGINE_INVARIANTS_DISABLE=1 to turn off.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "engine_invariants"

# 会话级计数上限：超过后停止记录（防长会话日志膨胀）
_MAX_LOGGED = 50


def _plugin_disabled() -> bool:
    return os.environ.get("ENGINE_INVARIANTS_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _iter_tool_call_ids(msg: Dict[str, Any]) -> List[str]:
    """提取 assistant 消息的 tool_call id 列表（OpenAI 格式）。

    Contract:
      Preconditions: msg 为 dict（tool_calls 缺失/非 list 均合法）
      Postconditions: 返回有效 id 的 str 列表；绝不 raise
    """
    calls = msg.get("tool_calls") or []
    if not isinstance(calls, list):
        return []
    ids: List[str] = []
    for c in calls:
        if isinstance(c, dict):
            cid = c.get("id")
            if isinstance(cid, str) and cid:
                ids.append(cid)
    return ids


def check_tool_pairing(messages: List[Dict[str, Any]]) -> List[str]:
    """不变量 1+3：工具配对完整性。

    Contract:
      Preconditions: messages 为 List[dict]（空列表合法）
      Postconditions: 返回违反描述列表（空=无违反）；绝不 raise

    - 每条 tool 消息的 tool_call_id 必须命中之前 assistant 声明的调用
    - 每个 assistant 声明的 tool_call 在历史内必须有对应 tool 结果
      （历史是完成轮快照，悬挂 call = 配对断裂）
    """
    violations: List[str] = []
    declared: Dict[str, int] = {}  # tool_call_id -> index of assistant msg
    answered: set = set()
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            for cid in _iter_tool_call_ids(m):
                if cid in declared:
                    violations.append(
                        f"dup_tool_call_id[{cid}]@{i}")
                else:
                    declared[cid] = i
        elif role == "tool":
            cid = m.get("tool_call_id")
            if not isinstance(cid, str) or not cid:
                violations.append(f"tool_msg_missing_call_id@{i}")
            elif cid not in declared:
                violations.append(f"orphan_tool_result[{cid}]@{i}")
            else:
                answered.add(cid)
    for cid, idx in declared.items():
        if cid not in answered:
            violations.append(f"unanswered_tool_call[{cid}]@{idx}")
    return violations


def check_role_alternation(messages: List[Dict[str, Any]]) -> List[str]:
    """不变量 2：角色交替合法性（连续同角色裸消息 = 违反）。

    Contract:
      Preconditions: messages 为 List[dict]（空列表合法）
      Postconditions: 返回违反描述列表；绝不 raise
    """
    violations: List[str] = []
    last_role: Optional[str] = None
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue  # system/tool 不参与交替
        if role == last_role and role == "user":
            # 连续 user 消息：修复器应已 merge——残留即违反
            violations.append(f"consecutive_user@{i}")
        last_role = role
    return violations


# 不变量注册表：(名称, 检查函数)。新增引擎不变量时在此追加。
INVARIANTS: Dict[str, Any] = {
    "tool_pairing": check_tool_pairing,
    "role_alternation": check_role_alternation,
}


def run_all_invariants(messages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """跑全部注册的不变量，返回 {name: violations}（含空列表）。

    Contract:
      Preconditions: messages 为 list（元素可为任意，非 dict 跳过）
      Postconditions: 每个注册名必有条目；单项检查异常不影响其他项
    """
    results: Dict[str, List[str]] = {}
    for name, fn in INVARIANTS.items():
        try:
            results[name] = fn(messages)
        except Exception as e:  # 单项失败不影响其他
            results[name] = [f"checker_error:{type(e).__name__}:{e}"]
    return results


def register(ctx) -> None:
    """注册 post_llm_call 检测钩子。

    Contract:
      Postconditions: 钩子已注册；钩子内部任何异常不外泄（fail-open）
    """

    def check_invariants(
        conversation_history=None, session_id=None, **kwargs
    ) -> None:
        if _plugin_disabled():
            return
        try:
            messages = conversation_history or []
            if not messages:
                return
            sid = session_id or "_global"
            st = get_session_state(sid, _NAMESPACE)
            results = run_all_invariants(messages)
            for name, violations in results.items():
                if not violations:
                    continue
                key = f"violations_{name}"
                st[key] = int(st.get(key, 0)) + len(violations)
                logged = int(st.get("logged", 0))
                if logged < _MAX_LOGGED:
                    st["logged"] = logged + 1
                    logger.warning(
                        "engine-invariants: %s 违反 x%d: %s",
                        name, len(violations),
                        "; ".join(violations[:5]),
                    )
        except Exception as e:
            logger.warning("engine-invariants skipped: %s", e, exc_info=True)

    ctx.register_hook("post_llm_call", check_invariants)
    logger.info("engine-invariants registered")
