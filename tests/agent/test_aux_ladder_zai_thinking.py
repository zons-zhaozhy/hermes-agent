"""ZAI 1210 thinking-conflict ladder rung: strip temperature/thinking and retry once.

Observed live (2026-09-15): glm-5.3-flash on the global endpoint api.z.ai rejects requests
with code 1210 whose message never names the offending parameter ("This model always engages
in thinking and cannot be disabled"), so the generic unsupported-parameter detector misses it
and auxiliary calls surfaced a raw 400. Expectations derived from that incident, independent
of implementation:

- 1210 + "thinking" in message + zai host -> one retry without temperature and without the
  thinking extra_body, plus a WARNING naming the actionable fix (explicit CN base_url).
- Non-zai hosts or non-thinking 1210 messages -> no thinking-strip retry at all.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest

import agent.auxiliary_client as aux


class _FakeCompletions:
    """Fake provider endpoint. ``fail_times`` = how many calls raise ``err`` BEFORE the
    ladder was entered (the production caller only invokes the ladder after the first
    failure), so the fake starts already past those: it replays them only for calls the
    ladder itself triggers."""

    def __init__(self, calls: List[Dict[str, Any]], fail_with: Exception, fail_times: int = 0) -> None:
        self._calls = calls
        self._err = fail_with
        self._fail_remaining = fail_times

    def create(self, **kw: Any) -> Any:
        self._calls.append(kw)
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._err

        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    def __init__(self, calls: List[Dict[str, Any]], err: Exception, base_url: str) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(calls, err)
        self.base_url = base_url


def _zai_thinking_error() -> RuntimeError:
    return RuntimeError(
        "Error code: 400 - {'error': {'code': '1210', 'message': "
        "'This model always engages in thinking and cannot be disabled; "
        "please use low, high, or max'}}"
    )


def _drive(gen: Any) -> Tuple[Any, Optional[Exception], Dict[str, Any]]:
    """Drive a ladder generator via the production driver so step/exception semantics
    match the real call path exactly."""
    resp, err, kwargs = aux._drive_ladder(gen, lambda step: step.args[0].chat.completions.create(**step.args[1]))
    return resp, err, kwargs


def _run_ladder(calls: List[Dict[str, Any]], err: Exception, base_url: str) -> Tuple[Any, Optional[Exception], Dict[str, Any]]:
    from agent.auxiliary_client import _LadderRoute

    route = _LadderRoute(
        client=_FakeClient(calls, err, base_url), task="goal_judge", tag="", async_mode=False,
        base_info=base_url, resolved_provider="zai", resolved_model="glm-5.3-flash",
        resolved_base_url=base_url, resolved_api_key="k", resolved_api_mode=None,
        final_model="glm-5.3-flash", main_runtime=None, route_info=None,
    )
    kwargs: Dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0,
                              "extra_body": {"thinking": {"type": "low"}}, "max_tokens": 16}
    gen = aux._ladder_parameter_rungs(err, route, kwargs, 16)
    return _drive(gen)


def test_zai_thinking_1210_strips_temperature_and_thinking_then_recovers() -> None:
    calls: List[Dict[str, Any]] = []
    resp, err, kwargs = _run_ladder(calls, _zai_thinking_error(), "https://api.z.ai/api/paas/v4/")
    assert err is None and resp is not None
    # The pre-ladder failure never reaches the fake; the ONLY recorded call is the retry,
    # which must carry neither temperature nor the thinking extra_body.
    assert len(calls) == 1
    assert "temperature" not in calls[0]
    assert "thinking" not in calls[0]["extra_body"]


def test_zai_thinking_1210_retry_failure_falls_through_with_narrowed_error() -> None:
    calls: List[Dict[str, Any]] = []

    class _AlwaysFail(_FakeCompletions):
        def create(self, **kw: Any) -> Any:
            calls.append(kw)
            raise _zai_thinking_error()

    client = _FakeClient.__new__(_FakeClient)
    client.chat = type("Chat", (), {})()
    client.chat.completions = _AlwaysFail(calls, _zai_thinking_error())
    client.base_url = "https://open.bigmodel.cn/api/coding/paas/v4"

    from agent.auxiliary_client import _LadderRoute

    route = _LadderRoute(
        client=client, task="t", tag="", async_mode=False, base_info=client.base_url,
        resolved_provider="zai", resolved_model="m", resolved_base_url=client.base_url,
        resolved_api_key="k", resolved_api_mode=None, final_model="m", main_runtime=None,
        route_info=None,
    )
    gen = aux._ladder_parameter_rungs(_zai_thinking_error(), route,
                                      {"messages": [], "temperature": 0,
                                       "extra_body": {"thinking": {"type": "low"}}}, None)
    # Retry fails with the same 1210, which _param_rung_accepts does not accept -> the
    # rung re-raises. That re-raise IS the documented fall-through (error surfaces to the
    # caller instead of being swallowed), so assert it propagates.
    with pytest.raises(Exception, match="1210"):
        _drive(gen)


@pytest.mark.parametrize("base_url,message,expect_retry", [
    # Non-zai host: no parameter rung matches -> zero retries, error returned as-is.
    ("https://api.openai.com/v1", "Error code: 400 - thinking unsupported", False),
    # ZAI host with a NON-thinking 1210-ish message: the thinking-strip rung must NOT
    # fire; the pre-existing max_tokens rung legitimately retries once (that is its
    # documented behavior, unrelated to this change), then re-raises the original error.
    ("https://api.z.ai/api/paas/v4/", "Error code: 400 - max_tokens unsupported", True),
])
def test_non_matching_errors_do_not_trigger_thinking_strip(base_url: str, message: str, expect_retry: bool) -> None:
    calls: List[Dict[str, Any]] = []
    err = RuntimeError(message)
    resp, narrowed, kwargs = _run_ladder(calls, err, base_url)
    assert len(calls) == (1 if expect_retry else 0)
    if expect_retry:
        # The retry came from the max_tokens rung: it strips max_tokens but must never
        # strip temperature or thinking (that is the thinking rung's job, which did not fire).
        assert "max_tokens" not in calls[0]
        assert calls[0].get("temperature") == 0
        assert calls[0]["extra_body"].get("thinking") == {"type": "low"}
    else:
        assert narrowed is not None and "thinking" in kwargs.get("extra_body", {})
