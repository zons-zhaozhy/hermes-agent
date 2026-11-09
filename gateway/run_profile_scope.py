"""Off-loop preparation for a routed profile's runtime scope."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable


def prepare_profile_runtime_scope(profile_home: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load the profile's secrets and terminal policy under its home override."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    from tui_gateway.launch_profile_policy import owner_secret_scope
    from tools.terminal_scope import prepare_profile_terminal_scope

    home = Path(profile_home)
    token = set_hermes_home_override(str(home))
    try:
        hydrate_profile_secret_sources(home)
        return owner_secret_scope(home), prepare_profile_terminal_scope(home)
    finally:
        reset_hermes_home_override(token)


@asynccontextmanager
async def async_profile_runtime_scope(
    profile_home: Path, sync_scope: Callable[..., Any],
) -> AsyncIterator[None]:
    """Enter *sync_scope* only after its blocking profile reads have left the loop."""
    secrets, terminal = await asyncio.to_thread(
        prepare_profile_runtime_scope, Path(profile_home))
    with sync_scope(
            Path(profile_home), secrets, prepared_terminal_scope=terminal):
        yield
