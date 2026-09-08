"""block_escalation 插件行为测试：意图指纹升级终止。

A/B 对照：不同目标各被拦 1 次 = 不升级；同一目标换通道第 2 次被拦 = 升级
（用户可见提示注入）；窗口外重置。
"""

from plugins.block_escalation import (
    _note_blocked, _on_post_tool_call, _on_transform_llm_output, _streaks, _escalated)


def setup_function():
    _streaks.clear()
    _escalated.clear()


def _blocked(tool, **args):
    _on_post_tool_call(tool_name=tool, status="blocked", error_type="plugin_block", args=args)


def test_same_target_two_channels_escalates():
    # 第 1 次：terminal 通道写 foo.py 被拦
    _blocked("terminal", command="sed -i 's/a/b/' src/foo.py")
    assert not _escalated
    # 第 2 次：execute_code 通道写同一文件被拦 → 升级
    _blocked("execute_code", code="open('src/foo.py','w').write('x')")
    assert any(any("foo.py" in part for part in fp) for fp in _escalated)
    out = _on_transform_llm_output(response_text="done")
    assert "[拦截升级]" in out


def test_different_targets_do_not_escalate():
    _blocked("terminal", command="sed -i 's/a/b/' src/foo.py")
    _blocked("terminal", command="sed -i 's/a/b/' src/bar.py")
    assert not _escalated
    out = _on_transform_llm_output(response_text="ok")
    assert out == "ok"


def test_non_blocked_status_ignored():
    _on_post_tool_call(tool_name="terminal", status="ok", args={"command": "x.py"})
    assert not _streaks


def test_escalation_flag_consumed_once():
    _blocked("terminal", command="sed -i 's/a/b/' a.py")
    _blocked("patch", path="a.py", old_string="x", new_string="y")
    first = _on_transform_llm_output(response_text="r1")
    second = _on_transform_llm_output(response_text="r2")
    assert "[拦截升级]" in first
    assert second == "r2"


def test_window_expiry_resets():
    _blocked("terminal", command="sed -i 's/a/b/' z.py")
    rec = next(iter(_streaks.values()))
    rec["last_ts"] -= 99999  # 模拟窗口过期
    _blocked("terminal", command="sed -i 's/a/b/' z.py")
    assert not _escalated
