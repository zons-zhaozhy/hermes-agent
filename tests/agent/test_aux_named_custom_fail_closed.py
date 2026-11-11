"""Named-custom aux resolution must fail closed when no key resolves for a
remote endpoint — never send a doomed placeholder request that 401s.

Red for: auxiliary `_resolve_named_custom_branch` built a client with the
``no-key-required`` placeholder for remote hosts (open.bigmodel.cn), sent the
request and burned a 401 + circuit-open cycle before anyone could diagnose the
missing credential (seen live: profile hermes-sentinel, 2026-11-11 10:51-11:00).
"""

import logging

import pytest

import agent.auxiliary_client as aux


def _req(provider: str = "zai") -> aux._ResolveRequest:
    return aux._ResolveRequest(
        provider=provider,
        original_provider=provider,
        model="glm-5.3",
        async_mode=False,
        raw_codex=False,
        explicit_base_url=None,
        explicit_api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
        task=None,
    )


def _entry(base_url: str) -> dict:
    # No api_key / key_env / key_cmd — mirrors profiles/<p>/config.yaml providers.zai
    return {"name": "zai", "base_url": base_url, "models": ["glm-5.3"]}


@pytest.fixture
def _no_pool(monkeypatch):
    monkeypatch.setattr(
        "agent.credential_pool.custom_provider_pool_key_candidates", lambda *a, **k: []
    )


def test_remote_endpoint_without_key_fails_closed(monkeypatch, _no_pool, caplog):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider",
        lambda name: _entry("https://open.bigmodel.cn/api/coding/paas/v4"),
    )
    with caplog.at_level(logging.ERROR, logger="agent.auxiliary_client"):
        client, model = aux._resolve_named_custom_branch(_req())
    assert client is None and model is None
    assert any("no resolvable api_key" in r.message for r in caplog.records)


def test_loopback_endpoint_keeps_placeholder(monkeypatch, _no_pool):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider",
        lambda name: _entry("http://127.0.0.1:11434/v1"),
    )
    res = aux._resolve_named_custom_branch(_req())
    # Local servers legitimately ignore auth; the branch still builds a client.
    assert res is not None and res[0] is not None


def test_resolved_key_still_builds_client(monkeypatch, _no_pool):
    entry = _entry("https://open.bigmodel.cn/api/coding/paas/v4")
    entry["api_key"] = "sk-test-key-material"
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider", lambda name: entry
    )
    res = aux._resolve_named_custom_branch(_req())
    assert res is not None and res[0] is not None
