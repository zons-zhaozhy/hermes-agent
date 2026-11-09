"""
反向采访澄清插件（requirement_interview）
==========================================

思想源：「8条元提示词」之反向采访+盲点扫描——AI 开工前主动把需求采访清楚，
并明确标注哪些信息是确认过的、哪些是假设、哪些要持续观察。
出处：docs/research/2026-08-30-thinking-systems-hermes-insights.md A-2 条。

机制：register_system_prompt_section —— 规则作为 session 级稳定段冻结进每个新
会话的 system prompt（core 持久化、resume 时从持久化 prompt 反解恢复），压缩
永不灭失，注意力权威同 system 级；是否属于「需求交付型任务」由 LLM 依据规则
文本自含的适用条件语义判断，不在 Python 侧用关键词/正则猜任务类型。

历史（2026-09-15 两轮迭代）：
1. 正则前缀表判定交付任务 → 用户点名「关键字匹配的做法太蠢了」：前缀表结构上
   不可穷举自然语言变体，语义分类不属于确定性代码层。
2. pre_llm_call 首轮注入 → 反方审查（deleg_1f8da3d0）抓出高危缺陷：规则活在
   首轮 user 消息 sidecar，in-place 压缩第 2 次起 protect_first_n 衰减为 0，
   规则被卷进 summary 后永久灭失（agent/context_compressor.py:3924-3928）；
   且 user 消息位置权威最弱。根因=注入锚选在 history 形状而非「规则是否稳定
   在上下文中」。
3. 本版：迁到 session 级 system prompt 段（core 规范通道 hermes_cli/
   plugins_dispatch.py::render_system_prompt_sections + agent/system_prompt.py::
   _frozen_plugin_prompt_sections）——一次根治灭失/稀释/中途生效/多模态丢弃
   四类缺陷。
4. 2026-10-12：澄清 GATE 由「一次性成批问出」升级为分层拷问——一轮只问前置条件已定
   的问题（分层待决前沿），每题带推荐答案，事实类问题先派子代理查、查不到才问用户；
   保留「必须先确认：」前缀（finish_guard 豁免依赖）与每轮 3-6 问上限。规则文本仍在
   冻结段内静态渲染，动态问题不得注入 system prompt（会破缓存前缀）。
5. 2026-10-28：开工令豁免两轮——首版用八字样枚举（「连做/不要等/…」类）触发，
   用户当场否决「各种说法啊」：第 1 条正则前缀表的同一错误在提示词层复发，开放集
   措辞不可枚举。终版判据=语义意图类别（立即按序连续执行/不等确认/不停顿），由主
   模型在读消息的既有推理内自判——零额外调用零延迟；不引入前置判定小模型（实测
   会加 0.7-2s 延迟，且结果注入要么破缓存前缀要么走 sidecar 有灭失史）。
6. 2026-11-09：常驻规则压缩为短契约。取消固定「每轮 3-6 问」和三分类输出模板，
   保留交付任务适用面、重大歧义停轮、可查事实先查、假设继续、拍板要素与催促连续
   执行豁免。

子代理/批处理不注入：它们没有 clarify 通道，规则反而诱导空转提问。

与 failure_preflight 互补：那个管风险预演，这个管需求歧义。
"""

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_INTERVIEW_RULE = """\
[开工前澄清]
仅交付任务适用；讨论、问答、闲聊、运维指令忽略。简单任务直接开工。
先查代码、配置、文档和数据；能查到的事实不问用户。
只有不可逆、高成本、对外承诺、数据结构或架构方向的重大歧义才停轮。停轮时以「必须先确认：」开头，一次列完问题；每题给推荐方案、成本和风险；不得同轮自问自答。
其他可逆选择写「假设：」并说明依据，随后继续执行，并保留可追溯记录。
请用户拍板时，每个决策给情况说明、两个及以上可选方案、推荐方案和不回复时的默认动作；只有单一路径时说明原因。
用户催促连续执行或授权按顺序完成时，不停轮提问；除不可逆、高成本、对外承诺外，记录假设后连续交付。交付前核对假设，未验证处写明未查证。"""

# 子代理与批处理会话没有 clarify 通道，规则只会诱导空转提问——按 session_info 的
# platform 字段跳过。
_EXCLUDED_PLATFORMS = {"subagent", "batch"}


def _interview_section(session_info: Mapping[str, Any]) -> str:
    """按会话元数据渲染采访规则段；子代理/批处理返回空（空段被 core 跳过）。

    Contract:
        Preconditions: session_info 为 core 冻结的只读映射（agent/system_prompt.py::
          _plugin_session_info），含 ``platform`` 键。
        Postconditions: 交付型会话返回 ``_INTERVIEW_RULE`` 原文；排除平台返回 ""。
    """
    if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
        return ""
    return _INTERVIEW_RULE


def register(ctx: Any) -> None:
    """插件入口。"""
    ctx.register_system_prompt_section("requirement_interview", _interview_section)
    logger.info("requirement_interview 插件已注册——反向采访澄清就绪（session 级稳定段）")
