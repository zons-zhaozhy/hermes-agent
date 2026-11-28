"""Shared engineering-intent classification for plugin injection gates."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

ENGINEERING_HINT_RE = re.compile(
    r"代码|文件|报错|错误|bug|测试|接口|函数|数据库|表结构|字段|配置|部署|构建|编译|"
    r"服务|页面|样式|日志|性能|分支|提交|仓库|git|api|http|python|typescript|"
    r"javascript|java|vue|react|pytest|ci|pr",
    re.IGNORECASE,
)
ENGINEERING_ACTION_RE = re.compile(
    r"继续|接着|重构|实现|开发|迁移|修复|编写|创建|删除|清理|部署|提交"
)


def has_engineering_intent(message: str, history_text: str = "") -> bool:
    """Return whether this turn starts or continues engineering work."""
    combined = f"{message}\n{history_text}" if history_text else message
    return bool(
        ENGINEERING_HINT_RE.search(combined)
        or ENGINEERING_ACTION_RE.search(message)
    )


def record_turn_intent(session_id: str, message: str) -> None:
    """Remember whether the current user turn is engineering work.

    Contract:
      Postconditions: writes ``turn_is_engineering`` into the ``message_intent`` session
        state; on failure the previous value stays and the miss is logged as a warning.
    """
    try:
        from plugins._shared_state import get_session_state

        get_session_state(session_id or "_global", "message_intent")["turn_is_engineering"] = has_engineering_intent(message)
    except Exception:  # 状态写不进去时下游闸门按「工程话题」照常执法
        logger.warning("record_turn_intent: could not store turn intent", exc_info=True)


def turn_is_engineering(session_id: str) -> bool:
    """Whether the current user turn is engineering work; unrecorded means True.

    Contract:
      Postconditions: returns the recorded flag for this session, or True when absent/unreadable.
    """
    try:
        from plugins._shared_state import get_session_state

        return bool(get_session_state(session_id or "_global", "message_intent").get("turn_is_engineering", True))
    except Exception:  # 读不到就按工程话题处理, 宁可多提醒一次也不漏拦
        logger.warning("turn_is_engineering: could not read turn intent", exc_info=True)
        return True
