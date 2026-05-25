"""Completion gate 纯逻辑单元测试 — 不依赖LLM，全是确定性判断。"""

from agent.completion_gate import (
    check_completion,
    check_iteration_floor,
    check_tool_prerequisite,
    compute_iteration_floor,
    _is_vague_response,
)


# =============================================================================
# 门控1：完成度检查
# =============================================================================

class TestCheckCompletion:
    """完成度检查的各种场景。"""

    def test_pass_when_no_issues(self):
        """正常完成 — 改了文件也验证了。"""
        should_continue, nudge = check_completion(
            turn_tool_names=["read_file", "write_file", "terminal"],
            turn_failed_mutations={},
            original_user_message="修复这个bug",
            final_response="已修复，测试全绿。",
            completion_retry_count=0,
        )
        assert should_continue is False
        assert nudge is None

    def test_block_on_failed_mutations(self):
        """有文件写入失败时应推回。"""
        should_continue, nudge = check_completion(
            turn_tool_names=["write_file"],
            turn_failed_mutations={"/tmp/test.py": {"error": "permission denied"}},
            original_user_message="写文件",
            final_response="写好了",
            completion_retry_count=0,
        )
        assert should_continue is True
        assert "文件写入失败" in nudge

    def test_block_on_mutate_without_verify(self):
        """修改了代码但没有验证（没有 terminal/execute_code）。"""
        should_continue, nudge = check_completion(
            turn_tool_names=["read_file", "write_file"],
            turn_failed_mutations={},
            original_user_message="改代码",
            final_response="改好了",
            completion_retry_count=0,
        )
        assert should_continue is True
        assert "没有验证结果" in nudge

    def test_block_on_write_without_read(self):
        """直接写文件但没先读 — 跳步。"""
        should_continue, nudge = check_completion(
            turn_tool_names=["write_file"],
            turn_failed_mutations={},
            original_user_message="改代码",
            final_response="改好了",
            completion_retry_count=0,
        )
        assert should_continue is True
        assert "跳步" in nudge

    def test_block_on_vague_response_no_tools(self):
        """回答是纯泛话且没调任何工具。"""
        should_continue, nudge = check_completion(
            turn_tool_names=[],
            turn_failed_mutations={},
            original_user_message="修复这个bug",
            final_response="好的，我来帮你处理。",
            completion_retry_count=0,
        )
        assert should_continue is True
        assert "没有执行任何操作" in nudge

    def test_pass_long_vague_response(self):
        """长回答即使匹配泛话模式也算通过（可能是真的在解释）。"""
        long_text = "好的，我来帮你处理。这个问题涉及到三个方面：第一，你需要先确认环境配置是否正确，包括Python版本和依赖库；第二，检查项目的配置文件中是否有语法错误或者路径问题；第三，运行完整的测试套件来验证修改是否正确。具体步骤如下：首先打开终端..."
        should_continue, nudge = check_completion(
            turn_tool_names=[],
            turn_failed_mutations={},
            original_user_message="怎么用",
            final_response=long_text,
            completion_retry_count=0,
        )
        # 长回答不触发泛话检测
        assert should_continue is False

    def test_pass_on_retry_exhausted(self):
        """超过最大重试次数必须放行，防止无限循环。"""
        should_continue, nudge = check_completion(
            turn_tool_names=["write_file"],
            turn_failed_mutations={"/tmp/a.py": {"error": "fail"}},
            original_user_message="写文件",
            final_response="好的",
            completion_retry_count=3,  # >= _MAX_COMPLETION_RETRIES
        )
        assert should_continue is False

    def test_multiple_failures_all_reported(self):
        """多个失败条件同时存在时全部报告。"""
        should_continue, nudge = check_completion(
            turn_tool_names=["write_file"],
            turn_failed_mutations={"/tmp/a.py": {"error": "fail"}},
            original_user_message="改代码",
            final_response="好的，我来处理。",
            completion_retry_count=0,
        )
        assert should_continue is True
        # 应该包含：文件写入失败 + 跳步 + 没有验证
        assert "文件写入失败" in nudge
        assert "跳步" in nudge
        assert "没有验证结果" in nudge


# =============================================================================
# 泛话检测
# =============================================================================

class TestIsVagueResponse:
    def test_empty_is_vague(self):
        assert _is_vague_response("") is True

    def test_whitespace_is_vague(self):
        assert _is_vague_response("   ") is True

    def test_short_commitment_is_vague(self):
        assert _is_vague_response("好的，我来处理。") is True

    def test_short_will_help_is_vague(self):
        assert _is_vague_response("我会帮你搞定这个。") is True

    def test_long_response_not_vague(self):
        assert _is_vague_response("x" * 200) is False

    def test_short_substantive_not_vague(self):
        """短但有具体内容的回答不算泛话。"""
        # 不匹配任何泛话模式
        assert _is_vague_response("已创建 /tmp/test.py") is False

    def test_none_handled(self):
        assert _is_vague_response(None) is True


# =============================================================================
# 门控2：工具前置约束
# =============================================================================

class TestCheckToolPrerequisite:
    def test_read_tool_no_prerequisite(self):
        """读操作不需要前置条件。"""
        assert check_tool_prerequisite("read_file", set()) is None
        assert check_tool_prerequisite("search_files", set()) is None
        assert check_tool_prerequisite("terminal", set()) is None

    def test_write_after_read_passes(self):
        """先读了再写，通过。"""
        assert check_tool_prerequisite("write_file", {"read_file"}) is None

    def test_write_after_search_passes(self):
        """先搜索了再写，通过。"""
        assert check_tool_prerequisite("patch", {"search_files"}) is None

    def test_write_without_read_blocks(self):
        """没读过就写，阻止。"""
        result = check_tool_prerequisite("write_file", set())
        assert result is not None
        assert "先读取" in result

    def test_patch_without_read_blocks(self):
        """没读过就patch，阻止。"""
        result = check_tool_prerequisite("patch", set())
        assert result is not None
        assert "先读取" in result

    def test_write_after_browser_snapshot_passes(self):
        """browser_snapshot 也算读取。"""
        assert check_tool_prerequisite("write_file", {"browser_snapshot"}) is None


# =============================================================================
# 门控3：迭代下限
# =============================================================================

class TestIterationFloor:
    def test_non_mutate_task_no_floor(self):
        """非修改类任务不限制。"""
        floor = compute_iteration_floor(["read_file", "web_search"], 1)
        assert floor == 0

    def test_mutate_task_has_floor(self):
        """修改类任务有下限。"""
        floor = compute_iteration_floor(["write_file"], 1)
        assert floor == 3  # _ITERATION_FLOOR_MUTATE

    def test_floor_met(self):
        """已达到下限，放行。"""
        cont, nudge = check_iteration_floor(3, 3)
        assert cont is False

    def test_floor_exceeded(self):
        """超过下限，放行。"""
        cont, nudge = check_iteration_floor(5, 3)
        assert cont is False

    def test_floor_not_met(self):
        """未达到下限，推回。"""
        cont, nudge = check_iteration_floor(1, 3)
        assert cont is True
        assert "至少需要 3 轮" in nudge

    def test_zero_floor_always_passes(self):
        """零下限始终通过。"""
        cont, nudge = check_iteration_floor(0, 0)
        assert cont is False
