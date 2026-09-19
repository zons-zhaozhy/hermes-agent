"""Unit tests for the agent.rate_limit.max_wait_seconds Retry-After cap in
compute_error_backoff.

Independent expectations (derived from the documented contract, not the implementation):
- default cap stays 600s when config is absent;
- a raised cap lets long Retry-After values through (multi-hour quota windows);
- the cap still applies on top of whatever the ceiling is;
- invalid config values fall back to the 600s default.
"""

from types import SimpleNamespace

from agent import turn_recovery


class _Resp:
    def __init__(self, retry_after):
        self.headers = {"retry-after": str(retry_after)} if retry_after is not None else {}


class _Err(Exception):
    def __init__(self, retry_after=None):
        super().__init__("429 too many requests")
        self.response = _Resp(retry_after)


def _agent():
    agent = SimpleNamespace(
        _rate_limit_min_wait_seconds=0.0,
        _client_log_context=lambda: "test",
        statuses=[],
    )
    agent._emit_status = agent.statuses.append
    agent._buffer_status = agent.statuses.append
    agent._emit_wait_notice = lambda *a, **k: None
    return agent


def _backoff(agent, err, *, retry_count=1, is_rate_limited=True):
    return turn_recovery.compute_error_backoff(
        agent, err,
        retry_count=retry_count, max_retries=3,
        is_rate_limited=is_rate_limited, is_zai_coding_overload=False,
        base_url="https://example.invalid/v1", model="test-model",
    )


def test_default_cap_is_600():
    # 期望: 无配置时 Retry-After 7200s 仍被压到 600s
    wait = _backoff(_agent(), _Err(retry_after=7200))
    assert wait == 600.0


def test_raised_cap_lets_long_window_through(monkeypatch):
    # 期望: 配置 max_wait_seconds=3600 后，429 Retry-After 7200s 等到 3600s
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"rate_limit": {"max_wait_seconds": 3600}}},
    )
    wait = _backoff(_agent(), _Err(retry_after=7200))
    assert wait == 3600.0


def test_cap_never_extends_short_retry_after(monkeypatch):
    # 期望: 短 Retry-After 不会被抬到上限
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"rate_limit": {"max_wait_seconds": 3600}}},
    )
    wait = _backoff(_agent(), _Err(retry_after=30))
    assert wait == 30.0


def test_invalid_config_falls_back_to_600(monkeypatch):
    # 期望: 配置为垃圾值（0/负数/NaN 字符串）时回落 600s 默认
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"rate_limit": {"max_wait_seconds": 0}}},
    )
    wait = _backoff(_agent(), _Err(retry_after=7200))
    assert wait == 600.0
