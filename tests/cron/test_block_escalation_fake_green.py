"""block_escalation 升级终止假绿防线：final_response 含 ESCALATION_MARKER 时，
scheduler 必须按失败记账（last_status=error），禁当成功回复投递。

修前行为：插件只在 transform_llm_output 追加警示文本，turn completed=True 逃过
final_response failed/completed 检查，job 被标 ok（假绿——ontox-health-check
29d16bbb6297 实测：02:27 block_escalation 终止，02:30 jobs.json last_status=ok）。
修后行为：marker 命中 → RuntimeError → run_one_job except 路径回写 error。
"""

import pytest

from cron.scheduler import _final_response_from_result
from plugins.block_escalation import ESCALATION_MARKER
from agent.turn_explainers import TurnExplainersMixin


class _AIAgent:
    _format_turn_completion_explanation = staticmethod(TurnExplainersMixin._format_turn_completion_explanation)


def test_escalation_marker_raises_so_job_marked_error():
    # 期望: 插件升级终止路径 = completed True + 尾部追加 marker 文本
    result = {
        "final_response": "巡检中断" + ("\n\n" + ESCALATION_MARKER + " 同一写操作意图已连续 2 次被拦…"),
        "failed": False, "completed": True,
        "turn_exit_reason": "", "messages": [], "api_calls": 3,
    }
    # 期望: marker 命中必须 raise（→ run_one_job except → last_status=error）
    with pytest.raises(RuntimeError, match="block-escalation"):
        _final_response_from_result(result, "job1", "Morning brief", _AIAgent)


def test_normal_reply_without_marker_passes_through():
    # 期望: 正常回复不含 marker，原样返回禁误伤
    result = {
        "final_response": "巡检通过：磁盘 4.8Gi，无异常。",
        "failed": False, "completed": True,
        "turn_exit_reason": "", "messages": [], "api_calls": 3,
    }
    assert _final_response_from_result(result, "job1", "Morning brief", _AIAgent) == "巡检通过：磁盘 4.8Gi，无异常。"


def test_failed_flag_takes_priority_over_marker_check():
    # 期望: failed=True 时第一分支先 raise（保留原始 error 文本）
    result = {
        "final_response": ESCALATION_MARKER + " residual", "failed": True,
        "completed": False, "error": "boom",
        "turn_exit_reason": "", "messages": [], "api_calls": 3,
    }
    with pytest.raises(RuntimeError, match="boom"):
        _final_response_from_result(result, "job1", "Morning brief", _AIAgent)
