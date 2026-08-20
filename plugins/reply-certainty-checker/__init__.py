"""reply-certainty-checker — 回复确定性检查（judge 版）

transform_llm_output：用 LLM judge 判定回复中是否存在"未标注实测来源的
技术性模糊断言"（可能/应该/大概率…），命中不修改用户可见回复，只记入
会话状态；下一轮 pre_llm_call 把提醒注入给 agent（复述守则通道），
强制重新确认事实。

设计修正史：
- v1 关键词黑名单+正则 → 被用户否定（禁关键词/正则匹配人话语义）
- v1 警告直接追加到用户可见回复 → 污染用户输出且自触发，改为注入给 agent
"""

import logging
import os
from typing import Any, Dict, Optional

from plugins._llm_judge import llm_judge_bool
from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "reply_certainty_checker"
_MAX_JUDGE_CALLS = 30
_JUDGE_TIMEOUT = 8.0

_JUDGE_SYSTEM = (
    "你是事实纪律审查员。判断下面这段助手回复是否包含'未标注验证来源的"
    "技术性模糊断言'——即用'可能/应该/大概率/也许'等修饰技术事实、且没有"
    "紧跟实测证据（如[实测]/日志/测试结果）。社交用语（'你可能需要…'）"
    "和已标注[未查证]/[推断]的诚实表述不算。只答 JSON："
    '{"uncertain": true} 或 {"uncertain": false}'
)


def _plugin_disabled() -> bool:
    return os.environ.get("REPLY_CERTAINTY_CHECKER_DISABLE") == "1"


def _has_unverified_hedge(text: str) -> Optional[bool]:
    """judge 判'未验证的技术性模糊断言'；None=fail-open。

    Contract:
      Preconditions: text 非空 str
      Postconditions: 返回 True/False/None，绝不 raise
    """
    if len(text) < 20:
        return False
    return llm_judge_bool(
        task="reply_certainty_checker",
        system=_JUDGE_SYSTEM,
        text=text[:4000],
        true_key="uncertain",
        timeout=_JUDGE_TIMEOUT,
    )


def _state(sid: str) -> Dict[str, Any]:
    return get_session_state(sid, _NAMESPACE)


def register(ctx):
    """注册 transform_llm_output（检测+记状态）与 pre_llm_call（注入提醒）。

    Contract:
      Postconditions: 两个钩子均已注册；钩子内部任何异常不外泄（fail-open）
    """

    def check_certainty(response_text, session_id=None, **kwargs) -> Optional[str]:
        """judge 判未验证断言 → 记状态供下轮注入；用户可见回复零改动。

        Contract:
          Postconditions: 恒返回 None（绝不修改用户回复）；异常不外泄
        """
        if _plugin_disabled():
            return None
        try:
            sid = session_id or ""
            st = _state(sid)
            if int(st.get("judge_calls", 0)) >= _MAX_JUDGE_CALLS:
                return None
            st["judge_calls"] = int(st.get("judge_calls", 0)) + 1
            flag = _has_unverified_hedge(response_text or "")
            if flag:
                st["pending_reminder"] = True
                logger.info(
                    "reply-certainty-checker: 未验证模糊断言已标记（下轮注入提醒）"
                )
            return None
        except Exception as exc:  # fail-open
            logger.warning("reply-certainty-checker failed: %s", exc)
            return None

    def inject_reminder(**kwargs) -> Optional[Dict[str, Any]]:
        """上一轮被标记 → 注入'先确认事实再断言'守则给 agent。

        Contract:
          Postconditions: 返回 None 或 {"context": str}；注入一次即消费标记
        """
        if _plugin_disabled():
            return None
        try:
            sid = kwargs.get("session_id") or ""
            st = _state(sid)
            if not st.get("pending_reminder"):
                return None
            del st["pending_reminder"]
            return {"context": (
                "【确定性检查】你的上一条回复含未标注验证来源的技术性模糊断言"
                "（可能/应该/大概率…）。本轮输出前重新确认这些断言：有实测证据的"
                "标注[实测]；没有的改为[未查证]或现在就去验证。"
            )}
        except Exception as exc:  # fail-open
            logger.warning("reply-certainty-checker inject failed: %s", exc)
            return None

    ctx.register_hook("transform_llm_output", check_certainty)
    ctx.register_hook("pre_llm_call", inject_reminder)
