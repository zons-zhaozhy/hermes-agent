"""Consecutive-timeout quarantine backoff for auxiliary endpoints.

2026-10-19 audit: a wedged local judge endpoint (port listening, inference
dead for ~17h) was retried with the full timeout budget every 60s — 359
doomed attempts in one outage window. The quarantine TTL must grow with the
streak so a persistent outage stops re-paying the full budget every minute.
"""

from agent import auxiliary_client as ac

_JUDGE_URL = "http://127.0.0.1:11434/v1"
_OTHER_URL = "http://127.0.0.1:8081/v1"


def test_timeout_quarantine_ttl_doubles_with_streak():
    ac._reset_aux_unhealthy_cache()
    provider = "custom:ollama-judge"
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 60    # 期望: 首次 60s
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 120   # 期望: 第2次翻倍
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 240   # 期望: 第3次翻倍
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 480   # 期望: 第4次翻倍
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 960   # 期望: 第5次翻倍
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 1800  # 期望: 第6次封顶
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 1800  # 期望: 之后保持封顶


def test_timeout_streak_resets_after_quiet_window():
    ac._reset_aux_unhealthy_cache()
    provider = "custom:ollama-judge"
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 60    # 期望: 首次 60s
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 120   # 期望: 连续第2次 120s
    key = ac._unhealthy_cache_key(provider, _JUDGE_URL)
    last_at, streak = ac._aux_timeout_streak[key]
    ac._aux_timeout_streak[key] = (last_at - ac._TIMEOUT_STREAK_RESET_SECONDS - 1, streak)
    assert ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL) == 60    # 期望: 静默超窗后重置 60s


def test_timeout_quarantine_survives_huge_streak_without_overflow():
    ac._reset_aux_unhealthy_cache()
    provider = "custom:ollama-judge"
    key = ac._unhealthy_cache_key(ac._normalize_chain_label(provider), _JUDGE_URL)
    # A persistent-timeout band (5s budget, hundreds of judge calls per hour) reaches a four-digit
    # streak inside one window; the unbounded exponent overflowed float there (OverflowError).
    ac._aux_timeout_streak[key] = (ac.time.time(), 2000)
    ttl = ac._transient_timeout_quarantine_ttl(provider, _JUDGE_URL)
    assert ttl == 1800  # 期望: 大 streak 封顶 1800 且不抛 OverflowError


def test_timeout_streak_isolated_per_endpoint():
    ac._reset_aux_unhealthy_cache()
    ac._transient_timeout_quarantine_ttl("custom:ollama-judge", _JUDGE_URL)
    ac._transient_timeout_quarantine_ttl("custom:ollama-judge", _JUDGE_URL)
    assert ac._transient_timeout_quarantine_ttl("custom:ornith-local", _OTHER_URL) == 60  # 期望: 端点独立计数
