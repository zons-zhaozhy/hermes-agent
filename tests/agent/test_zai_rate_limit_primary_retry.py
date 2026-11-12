"""Z.AI Coding GLM transient 429s exhaust primary retries before model fallback."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.error_classifier import classify_api_error
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent


def _make_agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        return AIAgent(
            api_key="primary-key-abcdef12",
            provider="zai",
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
            model="glm-5.3",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{
                "provider": "zai",
                "model": "glm-5.3-flash",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            }],
        )


def _response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


class ZaiTransientRateLimit(Exception):
    status_code = 429

    def __init__(self) -> None:
        super().__init__("Error code: 429 - 请求过于频繁，请稍后重试。")
        self.body = {"error": {"code": "1302", "message": "请求过于频繁，请稍后重试。"}}
        self.response = SimpleNamespace(headers={})


def test_transient_glm_429_retries_primary_before_glm_fallback():
    """The plan-backed primary gets its configured retry budget first; only exhaustion moves
    down the GLM fallback chain. A first-attempt switch is the paid-fallback bug."""
    agent = _make_agent()
    agent._api_max_retries = 3
    calls: list[tuple[str, str]] = []

    def api_call(_api_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) <= 3:
            raise ZaiTransientRateLimit()
        return _response("recovered on fallback")

    fallback_client = MagicMock()
    fallback_client.api_key = "primary-key-abcdef12"
    fallback_client.base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
    fallback_client._custom_headers = None
    fallback_client.default_headers = None

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client, "glm-5.3-flash")) as resolve,
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda model, _provider: model),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert result["final_response"] == "recovered on fallback"
    assert calls == [
        ("zai", "glm-5.3"),
        ("zai", "glm-5.3"),
        ("zai", "glm-5.3"),
        ("zai", "glm-5.3-flash"),
    ]
    resolve.assert_called_once()


def test_main_route_1302_carries_short_reset_into_credential_pool():
    """主路径必须把分类器给 Z.AI 1302 的短冷却 reset_at 传进凭据池，而不是丢掉它。

    1302 是瞬时限流（请求频率），分类器会给出 60s 的 reset_at；若主路径只透传原始
    ``extract_api_error_context``（其中没有 reset_at），凭据池就会按默认 3600s 把两把
    key 一起 bench 一小时，主路径无法在短暂退避后重试。"""
    exc = ZaiTransientRateLimit()
    classified = classify_api_error(
        exc, provider="custom", model="glm-5.3",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    )
    assert 0 < classified.error_context["reset_at"] - time.time() <= 60

    calls: list[dict] = []

    class _Agent:
        log_prefix = ""
        provider = "custom"
        api_mode = ""
        base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
        model = "glm-5.3"
        api_key = "sk-zai-primary"

        def _recover_with_credential_pool(self, **kwargs):
            calls.append(kwargs)
            return False, False

        def _extract_api_error_context(self, error):
            from agent.agent_runtime_helpers import extract_api_error_context
            return extract_api_error_context(error)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    agent = _Agent()
    recover_after_classification(
        agent, exc, classified, TurnRetryState(), status_code=429,
        error_context=agent._extract_api_error_context(exc),
        messages=[], api_messages=[],
    )

    assert calls, "主路径没有走到凭据池恢复"
    ctx = calls[0]["error_context"]
    assert ctx.get("reset_at") == classified.error_context["reset_at"]
    assert ctx.get("message")
