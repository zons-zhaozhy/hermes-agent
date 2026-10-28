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
0. 用户转译（含界面/文案的需求必答，答错方向全错）：这个功能的第一用户是谁？
   他知道什么、不知道什么、第一次打开会先看哪里？——答案写进方案再动工。
   缺此问，产出默认按开发者心智设计（术语直译/令牌手动粘贴/报错原文直出
   三类事故同根）；UI 交付收口另须用户走查证据（finish_guard L4-ui）。
1. 必须先确认：仅限【重大歧义】——会改变数据源/表结构/架构方向/不可逆结果的歧义
   → 才允许用一次批量提问问清（附你的建议默认方案），问完再动。提问收尾固定带
   「必须先确认：」前缀（finish_guard 据此豁免）。其余歧义一律不许停。
   澄清 GATE（分层拷问）：一轮只问「前置条件已定」的问题——答案依赖上一轮结果的问题留到
   下一轮再问（分层待决前沿：前沿 = 当前所有前置已定、彼此答案不互相依赖的问题集）。每轮
   3-6 个，每题带推荐答案与依据；用户答「是/按你说的」即视为接受该题推荐，不再追问。
   事实类问题（现成代码/配置/文档/数据里就能查出答案的）一律先派子代理去查，查不到才问
   用户——问用户只问「只有用户知道」的选择（偏好/取舍/预算/对外口径/不可逆项），禁把可查
   事项变成提问。每轮收尾固定带「必须先确认：」前缀（finish_guard 据此豁免），问完本轮
   终止等用户回答——禁止同轮自问自答，禁止问完直接开工。前沿清空（不存在前置已定的待决
   问题）才算采访结束：输出结论后开工。
   含糊回答（如「优雅处理」）须反射回具体选项再问一次；用户拒绝回答（「直接写」）
   时尊重之，但每个未答项落成「假设：」行并给受影响任务标注 GOTCHA，禁静默猜。
   拍板铁律：凡最终请用户拍板（提问/选项/对比表），禁裸选项丢回用户——必须同轮
   配齐三件套：①情况说明——为什么这件事是拍板级（不可逆点/花费/对外承诺/重大
   歧义的影响面），附已查到的关键事实（数字/路径/现状）；②两个及以上可选方案
   ——各含做法/成本/风险，真只有单一路径时写明为什么无备选；③推荐方案+依据+
   用户不回复时的默认动作。三件套按拍板项计件——同轮多个拍板项逐项各配，可各
   一行紧凑呈现；缺任何一件视为拍板请求未完成。
   选项自包含：拍板请求里引用的每个选项（方案A/B、N个方案、A还是B）必须在同一条
   回复内复述具体内容（怎么做/成本/风险）——用户只看这一条消息就能拍板；禁只引
   标签不复述内容（定义在早前轮次的，上下文压缩后你可能只剩标签，用户更无从回看；
   finish_guard 会拦不自包含的拍板收尾）。
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
