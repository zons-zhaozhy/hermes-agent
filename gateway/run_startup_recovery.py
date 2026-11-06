"""Left-core recovery used during gateway startup."""

from __future__ import annotations

from pathlib import Path


def recover_left_core_in(home: Path, *, hydrate_secrets: bool = True) -> None:
    """Install the catalog plugin of every feature that left core (Home Assistant) *home* uses, in
    *home*'s runtime scope (its config, secrets and allow_lazy_installs); once per process per home.
    Blocking: call it off the event loop."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.left_core_migration import recover_at_startup
    with _profile_runtime_scope(Path(home), hydrate_secrets=hydrate_secrets):
        recover_at_startup(gateway_start=True)


def recover_left_core_at_gateway_start() -> None:
    """Left-core migration for the homes whose platform config the runner loads first: the launch
    home and the default root (the multiplex primary; a named launcher is then served as a
    secondary). Every secondary runs :func:`recover_left_core_in` before its plugins are discovered,
    so a platform plugin it gets starts its adapter in the same gateway start."""
    from hermes_cli.left_core_migration import recover_at_startup
    from hermes_constants import get_default_hermes_root, get_hermes_home
    recover_at_startup(gateway_start=True)
    root = Path(get_default_hermes_root())
    if root.resolve() != Path(get_hermes_home()).resolve():
        recover_left_core_in(root)
