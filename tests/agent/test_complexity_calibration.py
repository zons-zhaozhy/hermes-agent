"""ReadThinkGate 复杂度检测校准测试。

验证两级机制：
  1. LLM 分类（mock 验证调用链路）
  2. 关键词 fallback（LLM 不可用时降级）

运行：scripts/run_tests.sh tests/agent/test_complexity_calibration.py -q
"""

import pytest
from unittest.mock import patch, MagicMock

from agent.read_think_gate import (
    detect_complexity,
    _classify_via_llm,
    _fallback_detect,
    _complexity_cache,
)


# ── 测试数据 ────────────────────────────────────────────────────────

COMPLEX_CASES = [
    "重构整个匹配引擎架构",
    "从零搭建认证系统",
    "全面审计代码质量问题",
    "系统设计文档审查",
    "数据库设计重构",
    "技术选型评估",
    "多服务联调",
    "微服务架构改造",
]

SIMPLE_CASES = [
    "修个typo",
    "改一行注释",
    "换个变量名",
    "fix a typo",
    "加个空格",
    "quick fix",
    "改个拼写错误",
    "nit: fix formatting",
]

NORMAL_CASES = [
    "修复一个bug",
    "帮我看看这个函数",
    "写个工具类",
    "修复 calculate_total 函数",
    "更新这个API的返回格式",
    "加个参数校验",
    "为什么这个报错",
    "帮我debug一下",
    "写个测试",
    "看看这段代码",
]

# 包含宽泛词但实际不是 complex
BOUNDARY_NOT_COMPLEX = [
    "改个架构里的变量名",
    "系统日志加一行",
    "设计一个typo修复方案",
    "全局搜索找一下",
    "升级一个pip包",
    "操作系统相关代码",
    "基础设施层有个typo",
    "迁移一个文件到新目录",
    "底层有个bug",
]


# ── 1. 关键词 fallback 测试 ─────────────────────────────────────────


class TestFallbackDetect:
    """关键词 fallback——LLM 不可用时的降级路径。"""

    @pytest.mark.parametrize("message", COMPLEX_CASES)
    def test_complex_detection(self, message):
        result = _fallback_detect(message.lower())
        assert result == "complex", f"'{message}' should be complex, got {result}"

    @pytest.mark.parametrize("message", SIMPLE_CASES)
    def test_simple_detection(self, message):
        result = _fallback_detect(message.lower())
        assert result == "simple", f"'{message}' should be simple, got {result}"

    @pytest.mark.parametrize("message", NORMAL_CASES)
    def test_normal_detection(self, message):
        result = _fallback_detect(message.lower())
        assert result == "normal", f"'{message}' should be normal, got {result}"

    def test_empty_string(self):
        assert _fallback_detect("") == "normal"

    @pytest.mark.parametrize("message", BOUNDARY_NOT_COMPLEX)
    def test_boundary_not_complex(self, message):
        """含宽泛词但实际不是 complex 任务——关键词 fallback 不应误判。"""
        result = _fallback_detect(message.lower())
        assert result != "complex", f"'{message}' should NOT be complex, got {result}"


# ── 2. LLM 分类测试（mock） ────────────────────────────────────────


class TestLLMClassifier:
    """LLM 分类调用链路——mock API 验证解析和缓存。"""

    def setup_method(self):
        _complexity_cache.clear()

    def _mock_response(self, text: str):
        """构造一个 fake OpenAI response。"""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = text
        return mock_resp

    @patch("agent.auxiliary_client.get_text_auxiliary_client", create=True)
    def test_llm_returns_complex(self, _):
        """LLM 返回 'complex' 时正确解析（带历史摘要）。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("complex")

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                   return_value=(mock_client, "deepseek-v4-flash")):
            result = _classify_via_llm("重构整个系统", "用户之前要求全面审计")
        assert result == "complex"

    @patch("agent.auxiliary_client.get_text_auxiliary_client")
    def test_llm_returns_simple(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("simple")
        mock_get_client.return_value = (mock_client, "deepseek-v4-flash")

        result = _classify_via_llm("修一个typo")
        assert result == "simple"

    @patch("agent.auxiliary_client.get_text_auxiliary_client")
    def test_llm_returns_normal(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("normal")
        mock_get_client.return_value = (mock_client, "deepseek-v4-flash")

        result = _classify_via_llm("修复一个bug")
        assert result == "normal"

    @patch("agent.auxiliary_client.get_text_auxiliary_client")
    def test_llm_unparseable_falls_back(self, mock_get_client):
        """LLM 返回无法解析的内容 → None → 调用方 fallback 到关键词。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("我不知道")
        mock_get_client.return_value = (mock_client, "deepseek-v4-flash")

        result = _classify_via_llm("随便什么")
        assert result is None

    @patch("agent.auxiliary_client.get_text_auxiliary_client")
    def test_llm_exception_returns_none(self, mock_get_client):
        """API 调用异常 → None → fallback。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("network error")
        mock_get_client.return_value = (mock_client, "deepseek-v4-flash")

        result = _classify_via_llm("随便什么")
        assert result is None

    @patch("agent.auxiliary_client.get_text_auxiliary_client")
    def test_no_client_returns_none(self, mock_get_client):
        """没有 auxiliary client → None → fallback。"""
        mock_get_client.return_value = (None, None)
        result = _classify_via_llm("随便什么")
        assert result is None

    @patch("agent.auxiliary_client.get_text_auxiliary_client")
    def test_cache_hit_skips_api(self, mock_get_client):
        """相同消息第二次走缓存，不重复调 API。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("complex")
        mock_get_client.return_value = (mock_client, "deepseek-v4-flash")

        _classify_via_llm("重构系统")
        _classify_via_llm("重构系统")
        assert mock_client.chat.completions.create.call_count == 1


# ── 3. detect_complexity 集成（LLM → fallback 降级） ────────────────


class TestDetectComplexityDegradation:
    """detect_complexity 在 LLM 失败时降级到关键词。"""

    def test_llm_available_uses_llm_result(self):
        """LLM 可用时使用 LLM 结果。"""
        with patch("agent.read_think_gate._classify_via_llm", return_value="complex"):
            assert detect_complexity("修复一个bug") == "complex"

    def test_llm_unavailable_falls_back_to_keywords(self):
        """LLM 不可用时降级到关键词匹配。"""
        with patch("agent.read_think_gate._classify_via_llm", return_value=None):
            assert detect_complexity("重构系统") == "complex"
            assert detect_complexity("修typo") == "simple"
            assert detect_complexity("修复bug") == "normal"

    def test_empty_message(self):
        assert detect_complexity(None) == "normal"
        assert detect_complexity("") == "normal"
