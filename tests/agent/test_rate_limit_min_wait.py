"""Unit tests for the agent.rate_limit.min_wait_seconds floor in compute_error_backoff.

Independent expectations (derived from the documented contract, not the implementation):
- floor 0 (default) leaves every wait unchanged;
- a positive floor raises short waits to the floor for rate-limited retries;
- the floor never lowers a longer Retry-After or adaptive value;
- non-rate-limited retries are untouched by the floor.
"""
from types import SimpleNamespace

from agent import turn_recovery
from agent.retry_utils import parse_retry_after_seconds


class _Resp:
    def __init__(self, retry_after):
        self.headers = {"retry-after": str(retry_after)} if retry_after is not None else {}


class _Err(Exception):
    def __init__(self, retry_after=None):
        super().__init__("429 too many requests")
        self.response = _Resp(retry_after)


def _agent(floor):
    agent = SimpleNamespace(
        _rate_limit_min_wait_seconds=floor,
        _client_log_context=lambda: "test",
        statuses=[],
    )
    agent._emit_status = agent.statuses.append
    agent._buffer_status = agent.statuses.append
    return agent


def _backoff(agent, err, *, retry_count=1, is_rate_limited=True):
    return turn_recovery.compute_error_backoff(
        agent, err,
        retry_count=retry_count, max_retries=3,
        is_rate_limited=is_rate_limited, is_zai_coding_overload=False,
        base_url="https://example.invalid/v1", model="test-model",
    )


def test_floor_zero_is_a_noop(monkeypatch):
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 7.0)
    wait = _backoff(_agent(0.0), _Err())
    assert wait == 7.0


def test_floor_raises_short_rate_limit_waits(monkeypatch):
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 7.0)
    wait = _backoff(_agent(60.0), _Err())
    assert wait == 60.0


def test_floor_never_lowers_retry_after(monkeypatch):
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 7.0)
    wait = _backoff(_agent(60.0), _Err(retry_after=120))
    assert wait == 120.0


def test_floor_ignores_non_rate_limited_retries(monkeypatch):
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 7.0)
    wait = _backoff(_agent(60.0), _Err(), is_rate_limited=False)
    assert wait == 7.0


def test_retry_after_parser_still_sane():
    assert parse_retry_after_seconds({"Retry-After": "120"}) == 120.0
    assert parse_retry_after_seconds("nonsense") is None
