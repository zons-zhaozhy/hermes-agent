"""Regression test: strict-mode block message must list ALL gaps at once.

2026-10-28 fix: `_build_block_message` used a chain of early-return `if`
branches — the panel only revealed the FIRST failing layer (read-count →
diversity → four-axis → char-count). The blocked side patched one layer,
retried, hit the next layer, and burned a full API round per layer (real
session: 3 consecutive blocks on one task). The fix enumerates every gap
via `_strict_gap_items` and joins them into one panel.

Expectations are derived from the unlock contract in `_try_unlock`, not
from the message wording of any single branch:
  - all four gates unmet + patch batch  → all four gap items present
  - read-count + chars met, no search tool → only the diversity gap
  - everything met → no block at all (gate stays open, returns None)
  - terminal-only batch (dynamically gated write form) → no four-axis item
    (mirrors _try_unlock's requesting_core_edit carve-out)
"""

from plugins.read_think_gate_host import ReadThinkGate, ReadThinkGateConfig


def _strict_gate() -> ReadThinkGate:
    cfg = ReadThinkGateConfig.from_mapping(
        {
            "unlock_after_investigation": False,
            "use_llm_judge": False,
            "complexity_adaptive": False,
            "min_read_only_calls": 2,
            "min_reasoning_chars": 200,
            "max_reasoning_rounds": 6,
        }
    )
    gate = ReadThinkGate(cfg)
    gate.reset_for_turn(user_message="")
    return gate


def test_all_gaps_listed_at_once():
    """Zero investigation + short content + patch batch → ①②③④ all present."""
    gate = _strict_gate()
    assert gate._read_only_count == 0  # 期望: 新 turn 计数清零
    msg = gate._build_block_message("patch", 50, ["patch"])
    assert "① 只读调查 0/2" in msg  # 期望: 调查量缺口带当前值/目标值
    assert "② 搜索类多样性未达标" in msg  # 期望: 无搜索类调用缺口
    assert "③ 分析字数 50/200" in msg  # 期望: 字数缺口带差值
    assert "④ 四轴缺「影响面、原意图、根因、风险」" in msg  # 期望: 四轴四轴全缺


def test_diversity_gap_listed_when_reads_met():
    """Read-count met (2 read_file), chars met, no search tool → only ②."""
    gate = _strict_gate()
    gate._read_only_count = 2
    gate._investigation_evidence = ["read_file: a.py", "read_file: b.py"]
    msg = gate._build_block_message("patch", 250, ["patch"])
    assert "①" not in msg  # 期望: 调查量已达标不再列
    assert "② 搜索类多样性未达标" in msg  # 期望: read_file 不满足搜索类
    assert "③" not in msg  # 期望: 字数已达标不再列
    assert "④ 四轴缺「影响面、原意图、根因、风险」" in msg  # 期望: 四轴仍缺


def test_no_block_when_all_gates_met():
    """All gates met (reads+search+chars+four axes) → check_batch returns None."""
    gate = _strict_gate()
    gate._read_only_count = 2
    gate._investigation_evidence = ["search_files: callers"]  # 期望: 满足搜索类
    gate._four_axis_found = {"影响面", "原意图", "根因", "风险"}
    result = gate.check_batch("x" * 300, ["patch"])
    assert result is None  # 期望: 四关全过 → 放行不拦


def test_terminal_batch_has_no_four_axis_item():
    """terminal-only batch (write-shaped) → gaps listed but ④ absent.

    _try_unlock only enforces four axes for FILE_WRITE_TOOLS batches;
    the gap panel must not advertise a requirement the unlock path never
    checks (a listed-but-unneeded item sends the blocked side hunting
    evidence that does not unlock anything).
    """
    gate = _strict_gate()
    gate._gated_tools.add("terminal")  # 模拟 _terminal_writes_file 动态门控
    msg = gate._build_block_message("terminal", 50, ["terminal"])
    assert "① 只读调查 0/2" in msg  # 期望: terminal 批次同样列调查量缺口
    assert "④" not in msg  # 期望: terminal 批次不列四轴项
