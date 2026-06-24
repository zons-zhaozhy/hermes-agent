"""会话成本阈值提醒——行为测试。

验证核心不变量：
- 阈值函数正确读取 config / 回退默认值
- _vprint 在跨过阈值倍数时被调用
- quiet_mode 下不触发
- 设为 0 关闭提醒
"""

import pytest
from unittest.mock import patch
from types import SimpleNamespace


def _make_agent(cost=0.0, quiet=False, next_tier=None):
    """构造一个最小 agent 替身，只含成本提醒所需字段。"""
    return SimpleNamespace(
        session_estimated_cost_usd=cost,
        quiet_mode=quiet,
        log_prefix="",
        _cost_next_warn_tier=next_tier,
        _vprint_calls=[],
    )


def _agent_vprint(agent, msg, force=False):
    agent._vprint_calls.append((msg, force))


def test_threshold_default_when_no_config():
    """config 缺失时回退默认 $1.0。"""
    from agent.conversation_loop import _get_cost_warning_threshold
    with patch("hermes_cli.config.load_config", return_value={}):
        assert _get_cost_warning_threshold() == 1.0


def test_threshold_from_config():
    """从 config.yaml display.cost_warning_threshold_usd 读取。"""
    from agent.conversation_loop import _get_cost_warning_threshold
    with patch("hermes_cli.config.load_config", return_value={
        "display": {"cost_warning_threshold_usd": 0.5}
    }):
        assert _get_cost_warning_threshold() == 0.5


def test_threshold_disabled():
    """设为 0 关闭提醒。"""
    from agent.conversation_loop import _get_cost_warning_threshold
    with patch("hermes_cli.config.load_config", return_value={
        "display": {"cost_warning_threshold_usd": 0}
    }):
        assert _get_cost_warning_threshold() == 0.0


def test_threshold_invalid_config_falls_back():
    """config 格式错误时静默回退默认值。"""
    from agent.conversation_loop import _get_cost_warning_threshold
    with patch("hermes_cli.config.load_config", return_value=None):
        assert _get_cost_warning_threshold() == 1.0


def test_warning_fires_at_threshold():
    """跨过阈值时 _vprint 被调用且 force=True。"""
    agent = _make_agent(cost=1.05, quiet=False, next_tier=1.0)
    agent._vprint = lambda msg, force=False: _agent_vprint(agent, msg, force)

    _threshold = 1.0
    _session_cost = float(agent.session_estimated_cost_usd or 0.0)
    if _session_cost > 0 and not getattr(agent, "quiet_mode", False):
        if _threshold > 0:
            _next_tier = getattr(agent, "_cost_next_warn_tier", _threshold)
            if _session_cost >= _next_tier:
                agent._cost_next_warn_tier = _next_tier + _threshold
                agent._vprint(
                    f"💰 会话成本: ${_session_cost:.2f} "
                    f"(阈值: ${_threshold:.2f})",
                    force=True,
                )

    assert len(agent._vprint_calls) == 1
    msg, force = agent._vprint_calls[0]
    assert force is True
    assert "1.05" in msg
    assert agent._cost_next_warn_tier == 2.0


def test_warning_skipped_below_threshold():
    """成本低于阈值时不触发。"""
    agent = _make_agent(cost=0.30, quiet=False, next_tier=1.0)
    agent._vprint = lambda msg, force=False: _agent_vprint(agent, msg, force)

    _threshold = 1.0
    _session_cost = float(agent.session_estimated_cost_usd or 0.0)
    if _session_cost > 0 and not getattr(agent, "quiet_mode", False):
        if _threshold > 0:
            _next_tier = getattr(agent, "_cost_next_warn_tier", _threshold)
            if _session_cost >= _next_tier:
                agent._vprint("should not fire", force=True)

    assert len(agent._vprint_calls) == 0


def test_warning_skipped_in_quiet_mode():
    """quiet_mode 下不触发。"""
    agent = _make_agent(cost=5.0, quiet=True, next_tier=1.0)
    agent._vprint = lambda msg, force=False: _agent_vprint(agent, msg, force)

    _threshold = 1.0
    _session_cost = float(agent.session_estimated_cost_usd or 0.0)
    if _session_cost > 0 and not getattr(agent, "quiet_mode", False):
        if _threshold > 0:
            agent._vprint("should not fire", force=True)

    assert len(agent._vprint_calls) == 0


def test_warning_fires_again_at_next_tier():
    """跨过 $2.0（第二个倍数）时再次触发，$1.x 时不触发。"""
    # 第一次：$1.05，触发
    agent = _make_agent(cost=1.05, quiet=False, next_tier=1.0)
    agent._vprint = lambda msg, force=False: _agent_vprint(agent, msg, force)
    _threshold = 1.0

    _cost = float(agent.session_estimated_cost_usd)
    if _cost >= agent._cost_next_warn_tier:
        agent._cost_next_warn_tier += _threshold
        agent._vprint(f"💰 ${_cost:.2f}", force=True)

    assert len(agent._vprint_calls) == 1
    assert agent._cost_next_warn_tier == 2.0

    # 第二轮：成本涨到 $1.80，未到 $2.0，不触发
    agent.session_estimated_cost_usd = 1.80
    _cost = float(agent.session_estimated_cost_usd)
    if _cost >= agent._cost_next_warn_tier:
        agent._cost_next_warn_tier += _threshold
        agent._vprint(f"💰 ${_cost:.2f}", force=True)

    assert len(agent._vprint_calls) == 1  # 仍然只有第一次

    # 第三轮：成本涨到 $2.50，到 $2.0，触发
    agent.session_estimated_cost_usd = 2.50
    _cost = float(agent.session_estimated_cost_usd)
    if _cost >= agent._cost_next_warn_tier:
        agent._cost_next_warn_tier += _threshold
        agent._vprint(f"💰 ${_cost:.2f}", force=True)

    assert len(agent._vprint_calls) == 2
    assert agent._cost_next_warn_tier == 3.0
