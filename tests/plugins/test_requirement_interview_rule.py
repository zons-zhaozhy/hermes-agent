"""Requirement-interview prompt-section contract."""

from plugins.requirement_interview import _INTERVIEW_RULE, _interview_section


def test_rule_stays_bounded_and_names_delivery_scope() -> None:
    """The rule targets deliverables and stays small enough for every session."""
    assert 300 <= len(_INTERVIEW_RULE) <= 700
    assert "仅交付任务适用" in _INTERVIEW_RULE
    assert "讨论、问答、闲聊、运维指令忽略" in _INTERVIEW_RULE
    assert "简单任务直接开工" in _INTERVIEW_RULE


def test_rule_stops_only_for_major_irreversible_decisions() -> None:
    """Only high-impact ambiguity may stop execution."""
    for term in ("不可逆", "高成本", "对外承诺", "数据结构", "架构方向"):
        assert term in _INTERVIEW_RULE
    assert "必须先确认：" in _INTERVIEW_RULE
    assert "一次列完" in _INTERVIEW_RULE
    assert "推荐方案" in _INTERVIEW_RULE


def test_rule_researches_facts_and_continues_on_reversible_choices() -> None:
    """Check available evidence first; do not turn implementation choices into questions."""
    assert "代码、配置、文档和数据" in _INTERVIEW_RULE
    assert "能查到的事实不问用户" in _INTERVIEW_RULE
    assert "假设：" in _INTERVIEW_RULE
    assert "随后继续执行" in _INTERVIEW_RULE


def test_rule_keeps_decision_package_requirements() -> None:
    """A decision request must be self-contained and include a default."""
    for term in ("情况说明", "两个及以上可选方案", "成本", "风险", "默认动作"):
        assert term in _INTERVIEW_RULE
    assert "只有单一路径" in _INTERVIEW_RULE


def test_rule_respects_user_request_to_keep_executing() -> None:
    """Urgency or blanket authorization must not cause repeated pauses."""
    assert "催促连续执行" in _INTERVIEW_RULE
    assert "不停轮提问" in _INTERVIEW_RULE
    assert "记录假设后连续交付" in _INTERVIEW_RULE


def test_subagent_and_batch_sessions_receive_no_rule() -> None:
    assert _interview_section({"platform": "subagent"}) == ""
    assert _interview_section({"platform": "batch"}) == ""
    assert _interview_section({"platform": "cli"}) == _INTERVIEW_RULE
