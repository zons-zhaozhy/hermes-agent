"""Named-custom-provider key resolution, split out of ``auxiliary_client`` to
offset its file-size ratchet; behaviour unchanged. Caller:
``auxiliary_client._resolve_named_custom_branch``.
"""

from __future__ import annotations

import contextlib
from typing import Any


def _named_custom_api_key(custom_entry: dict[str, Any], provider: str, custom_base: str, model: Any = None) -> Any:
    """Credential for a named custom provider: inline api_key → key_env → key_cmd → credential pool → placeholder.

    Aux resolves named custom providers here, not via _resolve_named_custom_runtime, so key_cmd must be
    honoured at the same precedence or every aux call 401s.

    Preconditions: ``custom_entry`` is one providers-dict row; ``custom_base`` non-empty.
    Postconditions: returns a usable key string/callable, or ``"no-key-required"``
    when nothing resolves (the caller fails closed for remote endpoints).
    """
    custom_key: Any = (custom_entry.get("api_key") or "").strip()
    custom_key_env = (custom_entry.get("key_env") or custom_entry.get("api_key_env") or "").strip()
    if not custom_key and custom_key_env:
        from agent.auxiliary_client import _scoped_key_env
        custom_key = _scoped_key_env(custom_key_env)
    custom_key_cmd = str(custom_entry.get("key_cmd", "") or "").strip()
    if custom_key_cmd:
        from agent.command_token_source import build_command_token_provider
        custom_key = build_command_token_provider(custom_key_cmd, custom_entry.get("name") or provider) or custom_key
    if not custom_key:
        with contextlib.suppress(Exception):
            from agent.credential_pool import custom_provider_pool_key_candidates, load_pool
            from agent.custom_entry_pool import entry_scoped_pool
            pool_name = custom_entry.get("provider_key") or custom_entry.get("name") or provider
            for pool_key in custom_provider_pool_key_candidates(custom_base, pool_name):
                try:
                    pool = load_pool(pool_key)
                except Exception:
                    continue
                if (pool := entry_scoped_pool(pool, pool_key, custom_entry)) is None or not pool.has_credentials():
                    continue
                if (pool_entry := pool.select(model=model)) is None:
                    continue
                pool_api_key = getattr(pool_entry, "runtime_api_key", None) or getattr(pool_entry, "access_token", "") or ""
                if str(pool_api_key).strip():
                    custom_key = str(pool_api_key).strip()
                    break
    return custom_key or "no-key-required"
