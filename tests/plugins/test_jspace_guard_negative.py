"""jspace_guard 负例契约测试——三态标注三道闸必须真拦（违规必红）。

Contract:
    Preconditions: plugins/discipline/jspace_guard.py 存在（2026-11-10 由
        独立插件 plugins/jspace-guard 迁入 discipline 套件）
    Postconditions: 声称验证无覆盖/声称完成未回读/断言无三态标注均返回非空 warning
"""
from plugins.discipline import jspace_guard as guard


def test_negative_coverage_gate_fires_on_bare_claim():
    out = guard._check_coverage_gate("测试通过了，功能正常")
    assert out != ""  # 期望: 非空=声称验证无覆盖描述被拦


def test_negative_coverage_gate_passes_with_scope():
    out = guard._check_coverage_gate("测试通过了，覆盖 tests/plugins 全部 12 个文件")
    assert out == ""  # 期望: 空=带覆盖描述放行


def test_negative_goal_rebase_gate_fires_on_completion_claim():
    out = guard._check_goal_rebase_gate("全部完成，任务结束", turn_tools=set())
    assert out != ""  # 期望: 非空=声称完成且未回读目标被拦


def test_negative_tri_state_gate_fires_on_unsourced_assertion():
    out = guard._check_tri_state_gate("已经修复了这个 bug，问题解决")
    assert out != ""  # 期望: 非空=事实断言无三态标注被拦


def test_negative_tri_state_gate_passes_with_tag():
    out = guard._check_tri_state_gate("已经修复了这个 bug，问题解决 [实测]")
    assert out == ""  # 期望: 空=带三态标注放行
