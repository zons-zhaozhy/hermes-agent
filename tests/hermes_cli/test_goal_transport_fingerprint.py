"""Regression tests: transport-error judge reasons must NOT stack as repeated verdicts.

Bug (observed in production state.db goal 20260917_232154): the judge API was down
(APIConnectionError) for 3 turns; judge_goal returns reason="judge error: APIConnectionError"
for those turns. judge_reason_fingerprint only exempted the "[judge unreachable" sentinel,
so the transport error fingerprinted as "judge error: apiconnectionerror" — identical 3 turns
in a row — and the repeated-verdict auto-pause fired, pausing an unfinished goal with the
misleading message "judge repeated the same rejection 3 turns in a row".

Expectations (derived from the documented contract, independent of implementation):
- "judge error: <AnyException>" fingerprints to None (no verdict this turn);
- the "[judge unreachable" sentinel fingerprints to None;
- genuinely identical verdict reasons still stack (bug fix must not weaken detection);
- a real reason after transport errors neither counts nor resets.
"""

import pytest

from hermes_cli.goals import judge_reason_fingerprint


class TestTransportReasonsAreSentinels:
    def test_judge_error_prefix_returns_none(self):
        # 期望: 网络异常 reason 是哨兵，不参与重复判定
        assert judge_reason_fingerprint("judge error: APIConnectionError") is None

    def test_judge_error_other_exception_types_return_none(self):
        assert judge_reason_fingerprint("judge error: ReadTimeout") is None
        assert judge_reason_fingerprint("Judge Error: ConnectionReset") is None  # 大小写不敏感

    def test_unreachable_sentinel_returns_none(self):
        assert judge_reason_fingerprint("[judge unreachable — no verdict this turn]") is None

    def test_blank_returns_none(self):
        assert judge_reason_fingerprint("") is None
        assert judge_reason_fingerprint("   ") is None


class TestRealReasonsStillFingerprint:
    def test_identical_reasons_hash_equal(self):
        # 期望: 真 verdict 仍可叠加（修复不得削弱模板拒绝检测）
        a = judge_reason_fingerprint("No concrete evidence behind the claim.")
        b = judge_reason_fingerprint("no concrete evidence   behind the claim.")
        assert a == b and a is not None

    def test_different_reasons_hash_apart(self):
        a = judge_reason_fingerprint("no evidence for step one")
        b = judge_reason_fingerprint("no evidence for step two")
        assert a != b

    def test_transport_then_real_reason_resets_streak(self):
        # 模拟 evaluate_after_turn 的叠加逻辑：传输哨兵既不计数也不重置
        reasons = ["judge error: APIConnectionError"] * 5 + ["no evidence yet"]
        consecutive, last_fp = 0, None
        for r in reasons:
            fp = judge_reason_fingerprint(r)
            if fp is not None:
                consecutive = consecutive + 1 if fp == last_fp else 1
                last_fp = fp
        # 期望: 5 次传输错误不叠加；第 1 条真 reason 计 1
        assert consecutive == 1
