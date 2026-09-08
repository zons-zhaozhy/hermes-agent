"""lazy-blocker — 卡点即停拦截器。

上一条 assistant 回复以「阻塞/挂起/留给别人/CI 可跑」类声明收尾，且未附当轮
替代通道尝试证据时，注入强制 SOP：三选一替代通道（绕开阻塞源的本地最小验证 /
真正依赖方的可执行验证命令 / 看板登记+明确复跑条件），禁裸挂起。

Contract:
  Preconditions: pre_llm_call kwargs 含 conversation_history（list[dict]）
  Postconditions: 命中且未达标时返回 {"context": ...} 注入一次/轮；
    不命中或已达标返回 None；永不 raise（异常吞入日志）
"""

from __future__ import annotations

import logging

logger = logging.getLogger("hermes.plugins.lazy_blocker")

# 阻塞声明模式（意图分类词表，窄口径——只匹配明确的转交/挂起表态）
_DEFER_PATTERNS = (
    "挂起",
    "留给 CI",
    "留给CI",
    "CI 通道可跑",
    "CI通道可跑",
    "等其他会话",
    "等并行会话",
    "等它提交",
    "下次再补",
    "后续补跑",
    "该会话提交后",
)

# 替代通道达标证据（声明里必须至少含其一才放行）
_ALTERNATIVE_EVIDENCE = (
    "当轮已尝试",
    "已用",
    "实测绕开",
    "替代通道",
    "复跑命令",
    "预计",
)

_NAMESPACE = "lazy_blocker"
_MAX_TRACKED_SESSIONS = 512
_COOLDOWN_TURNS = 2  # 同一会话连续注入间隔，防每轮刷屏


def _last_assistant_text(history: list | None) -> str:
    """取最近一条 assistant 消息文本，无则空串。"""
    if not history:
        return ""
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # 多模态 parts
            return "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
    return ""


def _is_lazy_deferral(text: str) -> bool:
    """命中阻塞声明且缺替代证据 → True。

    Contract:
      Preconditions: text 为 str（可为空）
      Postconditions: 纯函数零副作用
    """
    if not text:
        return False
    hit = any(p in text for p in _DEFER_PATTERNS)
    if not hit:
        return False
    return not any(e in text for e in _ALTERNATIVE_EVIDENCE)


def _on_pre_llm_call(**kwargs) -> dict | None:
    try:
        from plugins._shared_state import get_session_state

        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "") or "__default__"
        state = get_session_state(sid, _NAMESPACE)

        history = kwargs.get("conversation_history") or []
        text = _last_assistant_text(history)
        if not _is_lazy_deferral(text):
            return None

        injected = state.get("injected_count", 0)
        last_turn = state.get("last_inject_turn", -10**9)
        turn = kwargs.get("turn_id", 0)
        if isinstance(turn, (int, float)) and (turn - last_turn) < _COOLDOWN_TURNS:
            return None
        if injected >= 5:  # 封顶：连续 5 次仍懒 → 不再注入（避免死循环噪音）
            return None

        state["injected_count"] = injected + 1
        if isinstance(turn, (int, float)):
            state["last_inject_turn"] = turn
        while len(state) > _MAX_TRACKED_SESSIONS:  # 防御性，state 本身有界
            break

        return {
            "context": (
                "【懒病拦截 lazy-blocker】你上一条回复以「挂起/留给 CI/等并行会话」"
                "类声明收尾，且没有给出当轮已尝试的替代通道证据。禁裸挂起。"
                "立即三选一并执行（不得再以声明收尾）：\n"
                "1) 绕开阻塞源做本地最小验证（如隔离编译单文件/独立进程跑目标测试）；\n"
                "2) 给出真正依赖方可执行的验证命令并写明验收判据；\n"
                "3) 看板登记复跑条件（谁解除阻塞、解除后跑什么命令），并继续推进"
                "本任务中不依赖该阻塞的其他事项。\n"
                "选 3 时必须接着做 1 或 2 至少一项，不许只登记就停。"
            )
        }
    except Exception:  # hook 异常绝不影响 agent loop
        logger.exception("lazy-blocker hook failed")
        return None


def register(ctx):
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
