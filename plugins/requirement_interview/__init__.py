"""
反向采访澄清插件（requirement_interview）
==========================================

思想源：「8条元提示词」之反向采访+盲点扫描——AI 开工前主动把需求采访清楚，
并明确标注哪些信息是确认过的、哪些是假设、哪些要持续观察。
出处：docs/research/2026-08-30-thinking-systems-hermes-insights.md A-2 条。

机制：pre_llm_call hook。检测到「需求交付型任务」（做/写/建/改/出 X 类）时，
注入「采访清单」元要求：开工前输出三分类盲点扫描，重大歧义一次问清（批量提问），
可暂假设项显式标注后继续——堵住「直接开工」和「全停下问」两个极端。

与 failure_preflight 互补：那个管风险预演，这个管需求歧义。
"""

import logging
import re

logger = logging.getLogger(__name__)

_INTERVIEW_RULE = """\
[开工前采访——需求歧义清单，跳过即视为默认理解正确]
本轮是需求交付型任务。动手前先做盲点扫描，输出三分类（没有项可省略该类）：
1. 必须先确认：仅限【重大歧义】——会改变数据源/表结构/架构方向/不可逆结果的歧义
   → 才允许用一次批量提问问清（附你的建议默认方案），问完再动。提问收尾固定带
   「必须先确认：」前缀（finish_guard 据此豁免）。其余歧义一律不许停。
2. 可暂时假设：不影响主干、事后可改 → 显式标注「假设：xxx」后直接继续（默认路径）
3. 需持续观察：执行中才能验证的 → 列出观察点，做完回头核对
判定纪律：拿不准算不算重大歧义时，按「可暂时假设」处理（标注假设直接做）；
禁止把普通实现选择（命名/顺序/样式/工具选型）升级为提问。停下的代价高于选错可逆项。
若需求已足够明确（一句话可完成的简单任务），声明「无重大歧义」后直接开工。
"""

# 需求交付型任务信号（与 failure_preflight 同源动词表，但语义不同：
# 这里要的是「产出物交付」，纯运维动词如部署/提交不算）
_DELIVER_RE = re.compile(
    r"^(做|写|建|创建|改|修|重构|实现|开发|迁移|搭|搭一个|出一个|出一份|出一张|生成|设计|画|复刻)"
)

_DISCUSS_RE = re.compile(
    r"^(什么是|是什么|为啥|为什么|怎么理解|怎么看|如何理解|能不能|能不能实现|是否|啥意思|听说过|介绍一下|评价|对比)"
)


def _is_deliver_task(user_message: str) -> bool:
    """判断是否为需求交付型任务（而非讨论/闲聊/纯运维指令）。

    Contract:
        Preconditions: user_message 为字符串（可为空）
        Postconditions: 返回 True 当且仅当消息以交付动词开头且非讨论句式
    """
    if not user_message:
        return False
    msg = user_message.strip()
    if _DISCUSS_RE.match(msg):
        return False
    return bool(_DELIVER_RE.match(msg))


def _on_pre_llm_call(**kwargs) -> dict:
    """pre_llm_call 回调：交付型任务注入采访清单要求。"""
    user_message = kwargs.get("user_message", "")
    if not _is_deliver_task(user_message):
        return {}
    logger.debug(
        "requirement_interview: 注入采访要求 (user_message: %s)",
        user_message,
    )
    return {"context": _INTERVIEW_RULE}


def register(ctx):
    """插件入口。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("requirement_interview 插件已注册——反向采访澄清就绪")
