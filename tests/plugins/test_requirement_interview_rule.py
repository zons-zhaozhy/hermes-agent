"""契约测试：开工前采访规则的不可破坏面。

规则文本是 session 级 system prompt 冻结段的一部分——改错不会报错，只会静默改变行为。
因此对三类硬约束加机器断言：finish_guard 依赖的字面前缀、每轮问数上限、分层拷问的三条
原则，外加子代理/批处理的跳过语义。
"""

from plugins.requirement_interview import _INTERVIEW_RULE, _interview_section


def test_rule_keeps_finish_guard_exemption_prefix() -> None:
    """finish_guard 按字面「必须先确认：」识别合法澄清提问；丢了它，提问会被拦成死锁。"""
    assert "必须先确认：" in _INTERVIEW_RULE  # 期望: 豁免前缀在位


def test_rule_keeps_per_round_question_bound() -> None:
    assert "3-6" in _INTERVIEW_RULE  # 期望: 每轮 3-6 问的上限仍在


def test_rule_states_the_three_layered_grilling_principles() -> None:
    assert "前置条件已定" in _INTERVIEW_RULE  # 期望: 一轮只问前置已定的问题
    assert "推荐答案" in _INTERVIEW_RULE  # 期望: 每题带推荐答案
    assert "子代理" in _INTERVIEW_RULE  # 期望: 事实类问题先派子代理查


def test_rule_ends_interview_only_when_the_frontier_is_empty() -> None:
    assert "前沿清空" in _INTERVIEW_RULE  # 期望: 结束条件锚在待决前沿为空,不是问满几轮


def test_rule_states_the_decision_three_piece_set() -> None:
    """拍板三件套（用户 1028 要求）无机械执法者，全靠文本锚定防静默删改。"""
    assert "三件套" in _INTERVIEW_RULE  # 期望: 三件套命名在位
    assert "两个及以上可选方案" in _INTERVIEW_RULE  # 期望: 多方案硬要求在位
    assert "为什么无备选" in _INTERVIEW_RULE  # 期望: 单路径豁免须书面理由在位
    assert "情况说明" in _INTERVIEW_RULE  # 期望: ①情况说明要素在位
    assert "默认动作" in _INTERVIEW_RULE  # 期望: ③不回复默认动作在位
    assert "按拍板项计件" in _INTERVIEW_RULE  # 期望: 计件范围=按拍板项非按整轮


def test_rule_states_work_order_exemption() -> None:
    """开工令豁免（用户 2026-10-28 要求「做事不要等不要停不要歇」）锚定。

    根因：澄清门「问完本轮终止」条款与用户节奏指令打架；首版修复用字样枚举
    （八措辞列表）被用户当场否决——开放集不可穷举。本版判据=语义意图类别，
    由主模型按规则自含适用条件判（与插件 2026-09-15 正则前缀表被否决的修法
    同构：意图分类不属于确定性代码层/字样列表层）。
    """
    assert "开工令豁免" in _INTERVIEW_RULE  # 期望: 豁免条款命名在位
    assert "语义判意图，禁字样枚举" in _INTERVIEW_RULE  # 期望: 判据形态=语义意图
    assert "不等确认、不停顿" in _INTERVIEW_RULE  # 期望: 意图类别定义锚定
    assert "措辞变体不影响识别" in _INTERVIEW_RULE  # 期望: 开放集措辞鲁棒性在位
    assert "禁停轮提问" in _INTERVIEW_RULE  # 期望: 停轮禁令在位
    assert "落「假设：」行后直接连做" in _INTERVIEW_RULE  # 期望: 歧义处置路径=假设行
    assert "三类事项仍可停轮拍板" in _INTERVIEW_RULE  # 期望: 拍板通道保留边界清晰



def test_subagent_and_batch_sessions_receive_no_rule() -> None:
    assert _interview_section({"platform": "subagent"}) == ""  # 期望: 子代理无 clarify 通道
    assert _interview_section({"platform": "batch"}) == ""  # 期望: 批处理同样跳过
    assert _interview_section({"platform": "cli"}) == _INTERVIEW_RULE  # 期望: 交互会话注入原文
