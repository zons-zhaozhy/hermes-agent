# dupcheck-ok: 新增文件;全库无 goal judge failed-exits 既有测试(tests/hermes_cli/test_goal*.py 已 grep 核对)
"""The goal judge must see failed background exits and resist open-ended-goal self-reports.

Two structural gaps, both observed live in one session (2026-10-14): an open-ended quality
goal ("make X world-class") was closed DONE by a single all-green iteration report, while a
background process that exited -15 (SIGTERM) never reached the judge — its notification arrived
AFTER the verdict, so a failure signal and a DONE verdict coexisted without the judge knowing.
"""
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from hermes_cli import goals


def _resp(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _capture_prompt(**judge_kwargs: Any) -> str:
    seen: dict[str, Any] = {}

    def fake_call_llm(*a: Any, **kw: Any) -> SimpleNamespace:
        seen["prompt"] = kw.get("messages") or a
        return _resp('{"verdict": "continue", "reason": "x"}')

    with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
        goals.judge_goal("ship it", "shipped it", **judge_kwargs)
    return str(seen["prompt"])


def test_judge_prompt_lists_failed_exits():
    """judge_goal renders non-zero-exit processes into the prompt with exit code and command."""
    text = _capture_prompt(
        failed_processes=[
            {
                "pid": 20239,
                "session_id": "proc_abc",
                "command": "uvicorn app.main:app --port 18741",
                "exit_code": -15,
                "output_preview": "Shutting down",
            }
        ]
    )
    assert "NON-ZERO" in text  # 期望: 失败退出块标题存在
    assert "-15" in text  # 期望: 退出码原样呈现
    assert "uvicorn app.main:app" in text  # 期望: 命令可见, judge 才能判相关性


def test_no_failed_exits_keeps_prompt_clean():
    """failed_processes=None renders nothing — byte-identical to the pre-feature prompt."""
    text = _capture_prompt(failed_processes=None)
    assert "NON-ZERO" not in text
    text_default = _capture_prompt()
    assert "NON-ZERO" not in text_default


def test_judge_prompt_binds_open_ended_goals_to_a_criterion():
    """DONE criteria must tell the judge an open-ended goal is not done on a round's self-report."""
    text = _capture_prompt()
    assert "open-ended" in text  # 期望: 判据含开放式目标条款
    assert "completion criterion" in text  # 期望: 条款要求 goal 自带完成判据


def test_gather_failed_exits_filters_and_caps():
    """gather_failed_exits keeps only non-zero exits, newest first, capped at 5."""
    import time as _time
    now = _time.time()
    fake_sessions = [
        {"session_id": "run1", "pid": 1, "status": "running", "command": "a"},
        {"session_id": "ok0", "pid": 2, "status": "exited", "exit_code": 0, "command": "b", "exited_at": now - 60},
        {"session_id": "old", "pid": 3, "status": "exited", "command": "c", "exited_at": now - 60},
        {"session_id": "ancient", "pid": 6, "status": "exited", "exit_code": 2, "command": "f", "exited_at": now - 7200},
        {"session_id": "sig", "pid": 4, "status": "exited", "exit_code": -15, "command": "d", "exited_at": now - 300},
        {"session_id": "err", "pid": 5, "status": "exited", "exit_code": 1, "command": "e", "exited_at": now - 120},
    ]
    with patch("tools.process_registry.process_registry") as reg:
        reg.list_sessions.return_value = fake_sessions
        out = goals.gather_failed_exits()
    assert [s["session_id"] for s in out] == ["err", "sig"]  # 期望: 仅非零退出且在30分钟窗口内, exited_at 降序(err=2分钟前 先于 sig=5分钟前); ancient 超窗被排除
