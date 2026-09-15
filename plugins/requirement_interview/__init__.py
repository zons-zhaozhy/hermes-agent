"""
反向采访澄清插件（requirement_interview）
==========================================

思想源：「8条元提示词」之反向采访+盲点扫描——AI 开工前主动把需求采访清楚，
并明确标注哪些信息是确认过的、哪些是假设、哪些要持续观察。
出处：docs/research/2026-08-30-thinking-systems-hermes-insights.md A-2 条。

机制：pre_llm_call hook，会话首轮注入一次「采访规则」。是否属于「需求交付型
任务」由 LLM 依据规则文本自含的适用条件语义判断——不在 Python 侧用关键词/
正则猜任务类型：前缀表结构上不可穷举自然语言变体（英文、口语、动词不在句首），
语义分类不属于确定性代码层（2026-09-15 用户点名「关键字匹配的做法太蠢了」）。

与 failure_preflight 互补：那个管风险预演，这个管需求歧义。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_INTERVIEW_RULE = """\
[开工前采访——需求歧义清单，跳过即视为默认理解正确]
适用条件（自行判断）：仅当用户消息是「需求交付型任务」——要求做/写/建/改/修/产出
某个东西时适用；讨论、问答、闲聊、纯运维指令（部署/提交/查看）则忽略本规则。
适用时动手前先做盲点扫描，输出三分类（没有项可省略该类）：
1. 必须先确认：仅限【重大歧义】——会改变数据源/表结构/架构方向/不可逆结果的歧义
   → 才允许用一次批量提问问清（附你的建议默认方案），问完再动。提问收尾固定带
   「必须先确认：」前缀（finish_guard 据此豁免）。其余歧义一律不许停。
   拍板铁律：凡最终请用户拍板（提问/选项/对比表），同轮必须给出你自己的建议方案+
   依据（实测证据或明确推理），并列出若用户不回复时的默认动作——禁裸选项丢回用户。
2. 可暂时假设：不影响主干、事后可改 → 显式标注「假设：xxx」后直接继续（默认路径）
3. 需持续观察：执行中才能验证的 → 列出观察点，做完回头核对
判定纪律：拿不准算不算重大歧义时，按「可暂时假设」处理（标注假设直接做）；
禁止把普通实现选择（命名/顺序/样式/工具选型）升级为提问。停下的代价高于选错可逆项。
若需求已足够明确（一句话可完成的简单任务），声明「无重大歧义」后直接开工。
"""


def _on_pre_llm_call(**kwargs: Any) -> dict:
    """pre_llm_call 回调：会话首轮注入一次采访规则。

    Contract:
        Preconditions: kwargs 由 ``agent/turn_context.py::_collect_pre_llm_call_context``
          传入，含 ``is_first_turn`` 与 ``user_message``。
        Postconditions: 首轮返回 ``{"context": _INTERVIEW_RULE}``；非首轮返回 ``{}``
          （规则已在会话历史中，重复注入只浪费 token）。
    """
    if not kwargs.get("is_first_turn"):
        return {}
    logger.debug(
        "requirement_interview: 注入采访规则 (user_message: %s)",
        kwargs.get("user_message", ""),
    )
    return {"context": _INTERVIEW_RULE}


def register(ctx: Any) -> None:
    """插件入口。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("requirement_interview 插件已注册——反向采访澄清就绪")
