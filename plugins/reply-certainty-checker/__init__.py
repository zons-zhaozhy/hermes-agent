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

from plugins._llm_judge import llm_judge_multi
from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "reply_certainty_checker"
_MAX_JUDGE_CALLS = 30

# —— 回复侧合并判定（certainty + boundary + 消项闭环 + 宣告配对 + 术语漂移 共用一次 LLM 调用）——
# 各插件同为回复文本输入（transform_llm_output 与下一轮 pre_llm_call），合并成一次 multi judge，
# 判定语义与各自独立时完全一致，仅省掉串行调用（与用户侧
# yinyang+devil 合并同范式）。

REPLY_SIDE_SYSTEM = (
    "你是回复纪律审查员。对下面这段 AI 最终回复同时判定七个维度：\\n"
    "1. uncertain：是否包含'未标注验证来源的技术性模糊断言'——即用"
    "'可能/应该/大概率/也许'等修饰技术事实、且没有紧跟实测证据"
    "（如[实测]/日志/测试结果）。社交用语（'你可能需要…'）和已标注"
    "[未查证]/[推断]的诚实表述不算。\\n"
    "2. needs_audit：是否同时满足 a) 包含交付完成声明（声称任务/修复/"
    "测试/部署已完成、全部通过、已交付）；b) 未披露任何未验证边界"
    "（未列出未测路径/环境/已知风险/局限/未覆盖）。两条都满足才算 true；"
    "只是进度汇报、已含边界声明、闲聊、提问不算。\\n"
    "3. done_claim：是否包含交付完成声明（声称任务/修复/测试/部署"
    "已完成、全部通过、已交付）。只答 true/false。\\n"
    "4. has_boundary：是否披露了未验证边界（列出了未测路径/环境/"
    "已知风险/局限/未覆盖等待验证项）。\\n"
    "5. violation：回复是否违反「宣告-动手配对」纪律——宣告了具体的行动意图"
    "（建议实施/准备去做/接下来就做/等某事完成后动手/不说就按某方案办），"
    "且既没有已动手的证据（工具调用、代码修改、命令执行、真实结果输出），"
    "也未把该行动明确列为「必须先确认/须拍板」事项。陈述事实、回答问题、"
    "汇报已完成工作、纯提问澄清需求、明确说须拍板并附默认方案的都不违反。\\n"
    "6. naming_drift：是否同一概念混用不同称谓/词形（同义词互换、同一对象"
    "中英混称、名词形态不一致）。\\n"
    "7. jargon_coined：是否把完整词组硬压成含糊缩写/未定义先行词/行话未定义"
    "（首次出现的术语无定义无完整形式、用自造词或行话替代标准说法且读者"
    "无法从字面理解）。引用代码标识符、文件路径、引号内原文引用、对不同事物"
    "的合理区分、已在本段定义清楚后继续使用的术语都不算。\\n"
    "只回答一个 JSON 对象，含全部键："
    '{\\\"uncertain\\\": true/false, \\\"needs_audit\\\": true/false, '
    '\\\"done_claim\\\": true/false, \\\"has_boundary\\\": true/false, '
    '\\\"violation\\\": true/false, \\\"naming_drift\\\": true/false, \\\"jargon_coined\\\": true/false}'
)

REPLY_SIDE_KEYS = ["uncertain", "needs_audit", "done_claim", "has_boundary",
                  "violation", "naming_drift", "jargon_coined"]

# 进程级结果缓存：同一段回复（sha1 前 16 位）只判一次，certainty/boundary 共享。
_REPLY_SIDE_CACHE: dict[str, dict[str, Optional[bool]]] = {}


def judge_reply_side(text: str) -> dict[str, Optional[bool]]:
    """一次调用同时判 uncertain/needs_audit/done_claim/has_boundary/violation/naming_drift/jargon_coined；异常 → 全 None。

    同一回复文本进程内只发起一次真实调用（缓存命中直接返回），供
    reply-certainty-checker / finish_guard.completion_boundary_audit 两个
    transform_llm_output 钩子共享，消除重复调用。

    Contract:
      Preconditions: text 为非空 str
      Postconditions: 返回 dict 恰含全部键（True/False/None）；绝不 raise
    """
    import hashlib

    assert text, "text must be non-empty"
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    cached = _REPLY_SIDE_CACHE.get(h)
    if cached is not None:
        return cached
    result = llm_judge_multi(
        task="reply_side_guards",
        system=REPLY_SIDE_SYSTEM,
        text=text[:4000],
        keys=REPLY_SIDE_KEYS,
        max_tokens=200,  # 七个键的 JSON 回复需要比默认 64 更宽的输出预算
    )
    _REPLY_SIDE_CACHE[h] = result
    return result

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
    result = judge_reply_side(text)
    return result.get("uncertain")


def _state(sid: str) -> dict[str, Any]:
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

    def inject_reminder(**kwargs) -> Optional[dict[str, Any]]:
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
