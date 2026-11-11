"""Main-runtime named custom providers must fail closed without a remote key.

The auxiliary resolver already refuses placeholder-authenticated requests for
remote custom endpoints; the main runtime must apply the same contract instead
of sending ``Authorization: Bearer no-key-required`` and reporting the server's
401 as an expired user credential.
"""

import pytest

from hermes_cli import runtime_provider as rp


def _entry(base_url: str, **extra: object) -> dict:
    return {"name": "zai", "base_url": base_url, "models": ["glm-5.3"], **extra}


@pytest.fixture
def named_custom(monkeypatch):
    def install(entry: dict) -> None:
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda _name: entry)
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(rp, "_host_gated_env_key_candidates", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(rp, "_get_model_config", dict)

    return install


def test_remote_endpoint_without_key_fails_closed(named_custom):
    named_custom(_entry("https://open.bigmodel.cn/api/coding/paas/v4"))

    with pytest.raises(rp.AuthError, match="hermes auth list zai") as excinfo:
        rp.resolve_runtime_provider(requested="zai")

    assert excinfo.value.code == "missing_api_key"


def test_loopback_endpoint_keeps_placeholder(named_custom):
    named_custom(_entry("http://127.0.0.1:11434/v1"))

    resolved = rp.resolve_runtime_provider(requested="zai")

    assert (resolved["provider"], resolved["base_url"], resolved["api_key"]) == (
        "custom", "http://127.0.0.1:11434/v1", "no-key-required")


def test_resolved_entry_key_still_builds_runtime(named_custom):
    named_custom(_entry("https://open.bigmodel.cn/api/coding/paas/v4", api_key="sk-test-key-material"))

    resolved = rp.resolve_runtime_provider(requested="zai")

    assert resolved["api_key"] == "sk-test-key-material"


def test_declared_key_env_resolving_empty_fails_closed_and_logs(named_custom, monkeypatch, caplog):
    named_custom(_entry("https://api.example.test/v1", key_env="UNSET_LLM_KEY"))
    monkeypatch.delenv("UNSET_LLM_KEY", raising=False)

    with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
        with pytest.raises(rp.AuthError, match="named custom provider 'zai'"):
            rp.resolve_runtime_provider(requested="zai")

    hits = [record for record in caplog.records if "UNSET_LLM_KEY" in record.getMessage()]
    assert len(hits) == 1 and "zai" in hits[0].getMessage()
