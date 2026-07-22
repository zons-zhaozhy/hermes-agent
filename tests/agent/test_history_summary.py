"""_build_history_summary 专项测试 + 历史上下文端到端验证。"""

from agent.read_think_gate import _build_history_summary


class TestBuildHistorySummary:
    """_build_history_summary 提取逻辑测试。"""

    def test_empty_history(self):
        assert _build_history_summary(None) == ""
        assert _build_history_summary([]) == ""

    def test_no_user_messages(self):
        history = [
            {"role": "assistant", "content": "好的"},
            {"role": "system", "content": "提示词"},
        ]
        assert _build_history_summary(history) == ""

    def test_extracts_user_messages_in_order(self):
        history = [
            {"role": "user", "content": "重构系统"},
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "修bug"},
        ]
        result = _build_history_summary(history)
        assert "重构系统" in result
        assert "修bug" in result
        # 保持时间顺序
        assert result.find("重构系统") < result.find("修bug")

    def test_truncates_at_600_chars(self):
        history = [{"role": "user", "content": "x" * 1000}]
        result = _build_history_summary(history)
        assert len(result) <= 600

    def test_max_15_messages(self):
        history = []
        for i in range(30):
            history.append({"role": "user", "content": f"消息{i}"})
        result = _build_history_summary(history)
        # 只取最近 15 条
        assert "消息29" in result
        assert "消息15" in result
        assert "消息0" not in result  # 第 30 条前的被截断
