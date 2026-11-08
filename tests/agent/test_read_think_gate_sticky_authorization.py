"""ReadThinkGate session-sticky authorization and scratch-log exemptions.

2026-11-08 live session (agent.log 15:20-15:45): a disk-benchmark task the
user had authorized got re-gated eight times in a row, because (a) every
progress question（「跑得怎么样了」）reset the per-turn intent snapshot to
read_only, and (b) every relaunch command（``ssh ssdt-ts '… | tee
/tmp/quick336.log'``）was classified as a file write by the high-confidence
tee pattern.

Expected verdicts derive from the gate lifecycle contract, not from the
implementation: one user authorization covers the session's task until the
user explicitly forbids modification, and log capture into the OS scratch
directories is not a repository write.
"""

from __future__ import annotations

from pathlib import Path

from plugins.read_think_gate_host import ReadThinkGate, ReadThinkGateConfig, _terminal_writes_file


def _strict_gate(tmp_path: Path, tag: str) -> ReadThinkGate:
    return ReadThinkGate(
        ReadThinkGateConfig.from_mapping(
            {
                "enabled": True,
                "complexity_adaptive": False,
                "use_llm_judge": False,
                "use_llm_intent_classifier": False,
                "min_read_only_calls": 2,
                "min_reasoning_chars": 200,
                "max_reasoning_rounds": 6,
            }
        ),
        session_id=f"sticky-{tag}-{tmp_path.name}",
    )


def _record_complete_investigation(gate: ReadThinkGate, target: Path) -> None:
    gate._read_only_count = 3
    gate._investigation_evidence = [
        f"read_file: {target}",
        "search_files(callers): callers",
        "search_files(tests): tests",
        "terminal(readonly): git log -S target",
    ]
    gate._files_read = {str(target)}
    gate._four_axis_found = {"影响面", "原意图", "根因", "风险"}
    gate._satisfied = False
    gate._reasoning_rounds = 0


def _analysis_content() -> str:
    return (
        "影响面、原意图、根因、风险均已核实；验证命令为 scripts/run_tests.sh。"
        "调用方覆盖 app.main，现有测试覆盖成功和失败路径，最坏风险可回滚。" * 3
    )


def test_progress_question_keeps_session_write_authorization(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    gate = _strict_gate(tmp_path, "keep")
    gate.reset_for_turn("修复这个文件的错误，并补测试")

    gate.reset_for_turn("跑批进展怎么样了")
    _record_complete_investigation(gate, target)

    result = gate.check_batch(
        _analysis_content(),
        ["patch"],
        [{"path": str(target), "old": "value = 1", "new": "value = 2"}],
    )

    assert result is None
    assert gate.is_satisfied


def test_explicit_no_modify_revokes_session_authorization(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    gate = _strict_gate(tmp_path, "revoke")
    gate.reset_for_turn("修复这个文件的错误，并补测试")

    gate.reset_for_turn("先不要动了，只看看现在什么状态")
    _record_complete_investigation(gate, target)

    result = gate.check_batch(
        _analysis_content(),
        ["patch"],
        [{"path": str(target), "old": "value = 1", "new": "value = 2"}],
    )

    assert result is not None
    assert "用户意图" in result
    assert "只读" in result


def test_scratch_log_capture_is_not_a_gated_write() -> None:
    # 2026-11-08 实测被拦形态：远端跑批 tee 到 /tmp 日志
    assert _terminal_writes_file(
        "ssh -o ConnectTimeout=15 ssdt-ts 'cd ~/ssdt && ssdt run plans/quick.yaml "
        "--environment real 2>&1 | tee /tmp/quick336.log; tmux ls'"
    ) is False
    assert _terminal_writes_file("bash scripts/run_tests.sh tests/ > /tmp/rtg.log 2>&1") is False
    assert _terminal_writes_file("echo data | tee /var/tmp/batch.log") is False


def test_non_scratch_writes_stay_gated() -> None:
    assert _terminal_writes_file("echo hi | tee /Users/stan/notes.log") is True
    assert _terminal_writes_file("echo hi > out.txt") is True
    assert _terminal_writes_file("echo hi | tee ~/result.log") is True
    assert _terminal_writes_file("sed -i s/a/b/ /tmp/config.yml") is True
    assert _terminal_writes_file("cp /tmp/a.log /Users/stan/code/src.py") is True
