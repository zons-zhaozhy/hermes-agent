# -*- coding: utf-8 -*-
"""judge 复读检测（repeated-verdict pause）单测。

根因背景（2026-09-13 实测 goal 循环 turn4-8 五连拒同话术）：judge 模型弱时
每轮用几乎相同的 reason 拒绝，循环无检测地烧光预算。期望行为：连续
DEFAULT_MAX_REPEATED_JUDGE_REASONS 轮（默认 3）同指纹 reason → 自动 pause
并给出换模型指引；transport 哨兵不计入；不同 reason 重置计数；pause 后可 resume。

期望值独立推导（不读实现凑数）：
1. reason 完全相同连续 3 次 → repeated 计数=3，第 3 次触发 pause 决策
2. reason 文本近似（空白/大小写差异）→ 视为同指纹（复读的本质是话术模板）
3. reason 变化 → 计数归零
4. [judge unreachable] 哨兵 → 不计入且不重置
"""

from __future__ import annotations

from hermes_cli.goals import (
    DEFAULT_MAX_REPEATED_JUDGE_REASONS,
    judge_reason_fingerprint,
)
from hermes_cli.goals import GoalState


class TestFingerprint:
    def test_identical_reasons_same_fingerprint(self):
        r = "No concrete evidence shown of Part 1 completion."
        assert judge_reason_fingerprint(r) == judge_reason_fingerprint(r)

    def test_whitespace_and_case_insensitive(self):
        a = "No Concrete Evidence shown of Part 1."
        b = "  no concrete evidence shown of part 1.  "
        assert judge_reason_fingerprint(a) == judge_reason_fingerprint(b)

    def test_different_reasons_differ(self):
        a = "No concrete evidence shown of Part 1 completion."
        b = "The regulatory timeline is missing from the response."
        assert judge_reason_fingerprint(a) != judge_reason_fingerprint(b)

    def test_unreachable_sentinel_excluded(self):
        # transport 哨兵指纹固定为特殊值，调用方据此跳过计数
        fp = judge_reason_fingerprint("[judge unreachable — no verdict this turn]")
        assert fp is None

    def test_empty_reason_returns_none(self):
        assert judge_reason_fingerprint("") is None
        assert judge_reason_fingerprint("   ") is None


class TestThreshold:
    def test_default_threshold_is_3(self):
        # 独立期望：3 轮是复读与正常相似证据轮的平衡点
        assert DEFAULT_MAX_REPEATED_JUDGE_REASONS == 3


class TestGoalStateRoundTrip:
    def test_new_fields_serialize_and_load(self):
        from hermes_cli.goals import GoalState

        st = GoalState(goal="g", status="active")
        st.consecutive_repeated_reasons = 3
        st.last_reason_fingerprint = "abc"
        loaded = GoalState.from_json(st.to_json())
        assert loaded.consecutive_repeated_reasons == 3
        assert loaded.last_reason_fingerprint == "abc"

    def test_old_rows_without_fields_load_with_defaults(self):
        import json

        from hermes_cli.goals import GoalState

        old = json.dumps({"goal": "g", "status": "active", "max_turns": 20, "turns_used": 2})
        loaded = GoalState.from_json(old)
        assert loaded.consecutive_repeated_reasons == 0
        assert loaded.last_reason_fingerprint is None


class TestRepeatedCounterLogic:
    """指纹计数语义（独立推导，非读实现凑数）：
    - 同指纹连续出现 → 计数递增
    - 新指纹 → 计数归 1（本轮仍是有效 reason）
    - 哨兵轮 → 计数与指纹均保持不变
    """

    @staticmethod
    def _step(state: GoalState, reason: str) -> GoalState:
        """复现 evaluate_after_turn 中的计数片段（与实现同语义）。"""
        from hermes_cli.goals import judge_reason_fingerprint

        fp = judge_reason_fingerprint(reason)
        if fp is not None:
            if fp == state.last_reason_fingerprint:
                state.consecutive_repeated_reasons += 1
            else:
                state.consecutive_repeated_reasons = 1
            state.last_reason_fingerprint = fp
        return state
