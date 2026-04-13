from agent.smart_model_routing import choose_cheap_model_route, _calculate_complexity_score


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
}


# ============================================================
# Basic guard tests
# ============================================================

def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None


def test_empty_message():
    assert choose_cheap_model_route("", _BASE_CONFIG) is None


# ============================================================
# English: simple queries -> cheap model (score <= 0)
# ============================================================

def test_routes_simple_question():
    # Simple keywords: what (-5)
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["routing_reason"] == "simple_turn"


def test_routes_status_check():
    # Simple keywords: status (-5), check (-5) = -10
    result = choose_cheap_model_route("check current status", _BASE_CONFIG)
    assert result is not None


def test_routes_config_query():
    # Simple keywords: config (-5), show (-5) = -10
    result = choose_cheap_model_route("show config", _BASE_CONFIG)
    assert result is not None


def test_routes_version_query():
    # Simple keywords: version (-5)
    result = choose_cheap_model_route("what version is installed?", _BASE_CONFIG)
    assert result is not None


def test_routes_list_query():
    # Simple keywords: list (-5)
    result = choose_cheap_model_route("list all files", _BASE_CONFIG)
    assert result is not None


# ============================================================
# English: complex tasks -> primary model (score > 0)
# ============================================================

def test_skips_debug_task():
    # Complex keyword: debug (+10)
    assert choose_cheap_model_route("debug this traceback", _BASE_CONFIG) is None


def test_skips_implementation_task():
    # Complex keyword: implement (+10) + function code pattern (+15)
    assert choose_cheap_model_route("implement a function to parse JSON", _BASE_CONFIG) is None


def test_skips_refactor_task():
    # Complex keyword: refactor (+10)
    assert choose_cheap_model_route("refactor this code", _BASE_CONFIG) is None


def test_skips_analysis_task():
    # Complex keyword: analyze (+10), architecture (+10)
    assert choose_cheap_model_route("please analyze this architecture", _BASE_CONFIG) is None


def test_skips_review_task():
    assert choose_cheap_model_route("review this code", _BASE_CONFIG) is None


def test_skips_comparison_task():
    assert choose_cheap_model_route("compare these two approaches", _BASE_CONFIG) is None


def test_skips_benchmark_task():
    assert choose_cheap_model_route("benchmark this function", _BASE_CONFIG) is None


def test_skips_optimize_task():
    assert choose_cheap_model_route("optimize this query", _BASE_CONFIG) is None


def test_skips_test_task():
    assert choose_cheap_model_route("write tests for this", _BASE_CONFIG) is None


def test_skips_plan_task():
    assert choose_cheap_model_route("plan this migration", _BASE_CONFIG) is None


def test_skips_delegate_task():
    assert choose_cheap_model_route("delegate this to subagent", _BASE_CONFIG) is None


# ============================================================
# English: structural signals -> primary model
# ============================================================

def test_skips_code_with_backticks():
    # Code pattern: ``` (+15)
    assert choose_cheap_model_route("```python\nraise ValueError('bad')\n```", _BASE_CONFIG) is None


def test_skips_code_with_function_definition():
    # Code pattern: def (+15)
    assert choose_cheap_model_route("def my_function():\n    return 1", _BASE_CONFIG) is None


def test_skips_code_with_import():
    # Code pattern: import (+15)
    assert choose_cheap_model_route("import os\nprint('hello')", _BASE_CONFIG) is None


def test_skips_url_in_prompt():
    # URL pattern: https:// (+10)
    assert choose_cheap_model_route("check https://example.com", _BASE_CONFIG) is None


def test_skips_very_long_text():
    # >500 chars (+10), no simple keywords to offset
    assert choose_cheap_model_route("a" * 600, _BASE_CONFIG) is None


def test_moderate_length_with_simple_keywords_still_cheap():
    # <500 chars with simple keywords -> still cheap
    prompt = "please summarize this carefully " * 10  # ~340 chars
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is not None


# ============================================================
# English: mixed signals
# ============================================================

def test_mixed_signals_prioritizes_complex():
    # debug (+10) > what (-5) = +5 -> primary
    assert choose_cheap_model_route("debug what is wrong", _BASE_CONFIG) is None


def test_multiple_simple_keywords_still_simple():
    # -5 * 3 = -15 -> cheap
    assert choose_cheap_model_route("what where when", _BASE_CONFIG) is not None


def test_multiple_complex_keywords_definitely_complex():
    # +10 * 3 = +30 -> primary
    assert choose_cheap_model_route("debug and implement the refactor", _BASE_CONFIG) is None


def test_code_trumps_simple_keywords():
    # code (+15) > what (-5) = +10 -> primary
    assert choose_cheap_model_route("what is this ```python code```", _BASE_CONFIG) is None


# ============================================================
# Chinese: simple queries -> cheap model (score <= 0)
# ============================================================

def test_routes_chinese_simple_what():
    # Simple ZH: 是什么 (-5)
    result = choose_cheap_model_route("这是什么", _BASE_CONFIG)
    assert result is not None


def test_routes_chinese_status_check():
    # Simple ZH: 查看 (-5) + 状态 (-5) = -10
    result = choose_cheap_model_route("查看当前状态", _BASE_CONFIG)
    assert result is not None


def test_routes_chinese_config_query():
    # Simple ZH: 检查 (-5) + 配置 (-5) = -10
    result = choose_cheap_model_route("检查配置", _BASE_CONFIG)
    assert result is not None


def test_routes_chinese_version_query():
    # Simple ZH: 现在 (-5) + 版本 (-5) = -10
    result = choose_cheap_model_route("现在有什么版本", _BASE_CONFIG)
    assert result is not None


def test_routes_chinese_look():
    # Simple ZH: 看看 (-5)
    result = choose_cheap_model_route("帮我看看", _BASE_CONFIG)
    assert result is not None


# ============================================================
# Chinese: complex tasks -> primary model (score > 0)
# ============================================================

def test_skips_chinese_refactor():
    # Complex ZH: 重构 (+10)
    assert choose_cheap_model_route("请重构这段代码", _BASE_CONFIG) is None


def test_skips_chinese_debug():
    # Complex ZH: 调试 (+10) + 错误 (+10) = +20
    assert choose_cheap_model_route("调试一下这个错误", _BASE_CONFIG) is None


def test_skips_chinese_analysis():
    # Complex ZH: 分析 (+10) + 架构 (+10) = +20
    assert choose_cheap_model_route("帮我分析这个架构", _BASE_CONFIG) is None


def test_skips_chinese_implement():
    # Complex ZH: 实现 (+10) = +10
    assert choose_cheap_model_route("实现一个JSON解析函数", _BASE_CONFIG) is None


def test_skips_chinese_review():
    # Complex ZH: 审查 (+10) = +10
    assert choose_cheap_model_route("审查一下这段代码", _BASE_CONFIG) is None


def test_skips_chinese_optimize():
    # Complex ZH: 优化 (+10) + 性能 (+10) = +20
    assert choose_cheap_model_route("帮我做个性能优化", _BASE_CONFIG) is None


def test_skips_chinese_compare():
    # Complex ZH: 比较 (+10) = +10
    assert choose_cheap_model_route("比较这两个方案", _BASE_CONFIG) is None


def test_skips_chinese_test():
    # Complex ZH: 测试 (+10) = +10
    assert choose_cheap_model_route("写个测试用例", _BASE_CONFIG) is None


def test_skips_chinese_plan():
    # Complex ZH: 规划 (+10) + 迁移 (+10) = +20
    assert choose_cheap_model_route("规划一下迁移方案", _BASE_CONFIG) is None


def test_skips_chinese_deploy():
    # Complex ZH: 部署 (+10) = +10
    assert choose_cheap_model_route("帮我部署一下", _BASE_CONFIG) is None


def test_skips_chinese_investigate():
    # Complex ZH: 排查 (+10) = +10
    assert choose_cheap_model_route("排查这个问题", _BASE_CONFIG) is None


# ============================================================
# Chinese: mixed signals
# ============================================================

def test_chinese_complex_overrides_simple():
    # Complex ZH: 调试 (+10) + Simple ZH: 检查 (-5) = +5 -> primary
    assert choose_cheap_model_route("调试一下帮我检查", _BASE_CONFIG) is None


def test_chinese_no_keywords_score_zero_is_cheap():
    # No keywords at all -> score=0 -> cheap (conservative default)
    result = choose_cheap_model_route("你好", _BASE_CONFIG)
    assert result is not None


# ============================================================
# Resolve turn route fallback
# ============================================================

def test_resolve_turn_route_falls_back_to_primary_when_route_runtime_cannot_be_resolved(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "***",
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None
