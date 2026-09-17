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

子代理/批处理不注入：它们没有 clarify 通道，规则反而诱导空转提问。

与 failure_preflight 互补：那个管风险预演，这个管需求歧义。
"""

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_INTERVIEW_RULE = """\
[开工前采访——需求歧义清单，跳过即视为默认理解正确]
适用条件（自行判断）：仅当用户消息是「需求交付型任务」——要求做/写/建/改/修/产出
某个东西时适用；讨论、问答、闲聊、纯运维指令（部署/提交/查看）则忽略本规则。
适用时动手前先做盲点扫描，输出三分类（没有项可省略该类）：
1. 必须先确认：仅限【重大歧义】——会改变数据源/表结构/架构方向/不可逆结果的歧义
   → 才允许用一次批量提问问清（附你的建议默认方案），问完再动。提问收尾固定带
   「必须先确认：」前缀（finish_guard 据此豁免）。其余歧义一律不许停。
   澄清 GATE：问题必须一次性成批问出（3-6 个、各带推荐默认值，从分析真正悬而未决
   处取材，禁凑数），问完本轮终止等用户回答——禁止同轮自问自答，禁止问完直接开工。
   含糊回答（如「优雅处理」）须反射回具体选项再问一次；用户拒绝回答（「直接写」）
   时尊重之，但每个未答项落成「假设：」行并给受影响任务标注 GOTCHA，禁静默猜。
   拍板铁律：凡最终请用户拍板（提问/选项/对比表），同轮必须给出你自己的建议方案+
   依据（实测证据或明确推理），并列出若用户不回复时的默认动作——禁裸选项丢回用户。
2. 可暂时假设：不影响主干、事后可改 → 显式标注「假设：xxx」后直接继续（默认路径）
3. 需持续观察：执行中才能验证的 → 列出观察点，做完回头核对
判定纪律：拿不准算不算重大歧义时，按「可暂时假设」处理（标注假设直接做）；
禁止把普通实现选择（命名/顺序/样式/工具选型）升级为提问。停下的代价高于选错可逆项。
若需求已足够明确（一句话可完成的简单任务），声明「无重大歧义」后直接开工。"""

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
