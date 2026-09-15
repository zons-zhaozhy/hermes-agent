# -*- coding: utf-8 -*-
"""goal 独立反方复核（cross-review）单测。

根因背景：主 judge 单点判定 done，同一个 judge 模型被反复访问会学出话术套路，
agent 也会学会用措辞（"全部通过/已完成/已验证"）骗过它。机制：judge 判 done 后，
用 DIFFERENT provider/model（auxiliary.goal_cross_review）做独立反方复核，专门找
"钻空子"证据。fail-open：复核不可用（import/API/parse 失败）时放行 done，绝不 wedge 循环。

期望值独立推导（不读实现凑数）：
1. challenge=true 的 JSON → 反方否决 done
2. challenge=false → 确认 done
3. 非 JSON / 缺 challenge 字段 → parse_failed=True（调用方 fail-open）
4. 空 goal / API 抛异常 / parse 失败 → _run_cross_review 返回 None（fail-open）
5. 连续 DEFAULT_MAX_CROSS_REVIEW_CHALLENGES 次否决 → 自动 pause（人工裁决）
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Optional

from hermes_cli.goals import (
    DEFAULT_MAX_CROSS_REVIEW_CHALLENGES,
    GoalState,
    _parse_cross_review_response,
    _render_cross_review_prompt,
    _run_cross_review,
)


def _fake_call_llm(content: str = "", exc: Optional[Exception] = None) -> Callable[..., Any]:
    """构造 fake call_llm：返回带 .choices[0].message.content 的响应，或抛 exc。"""
    def _inner(**kwargs: Any) -> Any:
        if exc is not None:
            raise exc
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return _inner


class TestParseCrossReviewResponse:
    def test_challenge_true_json(self) -> None:
        ok, reason, failed = _parse_cross_review_response('{"challenge": true, "reason": "no test output"}')
        assert ok is True
        assert "no test output" in reason
        assert failed is False

    def test_challenge_false_json(self) -> None:
        ok, reason, failed = _parse_cross_review_response('{"challenge": false, "reason": "exit 0 shown"}')
        assert ok is False
        assert failed is False

    def test_string_veto_accepted(self) -> None:
        ok, _, failed = _parse_cross_review_response('{"challenge": "veto", "reason": "x"}')
        assert ok is True
        assert failed is False

    def test_non_json_fails_open(self) -> None:
        _, _, failed = _parse_cross_review_response("I think it's fine, no JSON here")
        assert failed is True

    def test_missing_challenge_fails_open(self) -> None:
        _, _, failed = _parse_cross_review_response('{"reason": "no challenge field"}')
        assert failed is True

    def test_empty_fails_open(self) -> None:
        _, _, failed = _parse_cross_review_response("")
        assert failed is True


class TestRunCrossReview:
    def test_empty_goal_returns_none(self, monkeypatch) -> None:
        # 空 goal 无物可审 → fail-open（None），绝不 wedge
        assert _run_cross_review("", "done claim", "judge ok", None) is None

    def test_api_error_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm(exc=RuntimeError("boom")))
        assert _run_cross_review("goal", "claim", "judge ok", None) is None

    def test_unparseable_reply_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm(content="not json"))
        assert _run_cross_review("goal", "claim", "judge ok", None) is None

    def test_veto_returns_challenge_true(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            _fake_call_llm(content='{"challenge": true, "reason": "no observed evidence"}'),
        )
        result = _run_cross_review("ship the API", "all done!", "judge said done", "observed_evidence=0")
        assert result is not None
        assert result[0] is True
        assert "observed" in result[1]

    def test_confirm_returns_challenge_false(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            _fake_call_llm(content='{"challenge": false, "reason": "exit 0 pasted"}'),
        )
        result = _run_cross_review("ship the API", "done, exit 0", "judge ok", "observed_evidence=1")
        assert result == (False, "exit 0 pasted")


class TestRenderCrossReviewPrompt:
    def test_carries_goal_and_judge_reason(self) -> None:
        prompt = _render_cross_review_prompt("ship the API", "done", "judge: evidence seen", None)
        assert "ship the API" in prompt
        assert "judge: evidence seen" in prompt


class TestGoalStateRoundTrip:
    def test_cross_review_challenges_serialize(self) -> None:
        st = GoalState(goal="g", status="active")
        st.cross_review_challenges = 2
        loaded = GoalState.from_json(st.to_json())
        assert loaded.cross_review_challenges == 2

    def test_old_rows_default_to_zero(self) -> None:
        import json
        old = json.dumps({"goal": "g", "status": "active", "turns_used": 1})
        loaded = GoalState.from_json(old)
        assert loaded.cross_review_challenges == 0


class TestEvaluateAfterTurnDoneBranch:
    """done 分支 + cross-review 的决策语义（mock judge_goal / _run_cross_review / save_goal）。

    独立推导：judge 判 done 后——
    - cross-review 否决 → status 仍是 active，should_continue=True，verdict=continue
    - cross-review 确认 → status=done，should_continue=False
    - cross-review None（fail-open）→ status=done
    - 连续 N 次否决 → status=paused（人工裁决）
    """

    @staticmethod
    def _make_manager(monkeypatch: Any) -> Any:
        monkeypatch.setattr("hermes_cli.goals.load_goal", lambda sid: None)
        monkeypatch.setattr("hermes_cli.goals.save_goal", lambda sid, st: None)
        from hermes_cli.goals import GoalManager
        return GoalManager("test-session")

    def test_veto_sends_agent_back(self, monkeypatch) -> None:
        mgr = self._make_manager(monkeypatch)
        mgr.set("ship the API")
        monkeypatch.setattr(
            "hermes_cli.goals.judge_goal",
            lambda *a, **k: ("done", "judge: done", False, None, False),
        )
        monkeypatch.setattr(
            "hermes_cli.goals._run_cross_review",
            lambda goal, resp, reason, tc: (True, "no observed evidence"),
        )
        decision = mgr.evaluate_after_turn("all done!", tool_calls_summary="observed_evidence=0")
        assert decision["status"] == "active"
        assert decision["should_continue"] is True
        assert decision["verdict"] == "continue"
        assert mgr._state.cross_review_challenges == 1

    def test_confirm_marks_done(self, monkeypatch) -> None:
        mgr = self._make_manager(monkeypatch)
        mgr.set("ship the API")
        monkeypatch.setattr(
            "hermes_cli.goals.judge_goal",
            lambda *a, **k: ("done", "judge: done", False, None, False),
        )
        monkeypatch.setattr(
            "hermes_cli.goals._run_cross_review",
            lambda goal, resp, reason, tc: (False, "exit 0 pasted"),
        )
        decision = mgr.evaluate_after_turn("done, exit 0", tool_calls_summary="observed_evidence=1")
        assert decision["status"] == "done"
        assert decision["should_continue"] is False
        assert mgr._state.cross_review_challenges == 0

    def test_unavailable_reviewer_fails_open_to_done(self, monkeypatch) -> None:
        mgr = self._make_manager(monkeypatch)
        mgr.set("ship the API")
        monkeypatch.setattr(
            "hermes_cli.goals.judge_goal",
            lambda *a, **k: ("done", "judge: done", False, None, False),
        )
        monkeypatch.setattr("hermes_cli.goals._run_cross_review", lambda *a, **k: None)
        decision = mgr.evaluate_after_turn("done", tool_calls_summary="observed_evidence=1")
        assert decision["status"] == "done"

    def test_consecutive_vetoes_auto_pause(self, monkeypatch) -> None:
        mgr = self._make_manager(monkeypatch)
        mgr.set("ship the API")
        monkeypatch.setattr(
            "hermes_cli.goals.judge_goal",
            lambda *a, **k: ("done", "judge: done", False, None, False),
        )
        monkeypatch.setattr(
            "hermes_cli.goals._run_cross_review",
            lambda goal, resp, reason, tc: (True, "still no evidence"),
        )
        decision = None
        for _ in range(DEFAULT_MAX_CROSS_REVIEW_CHALLENGES):
            decision = mgr.evaluate_after_turn("still claiming done", tool_calls_summary="observed_evidence=0")
        assert decision is not None
        assert decision["status"] == "paused"
        assert "cross-review" in decision["message"].lower()
