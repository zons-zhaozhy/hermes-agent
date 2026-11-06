"""Scope ONE gateway process's cron tick set (``cron.tick_profiles``).

A single host gateway normally ticks every profile it serves, which is correct for a
one-process install: ``profiles_to_serve`` names them, the multiplex ticker visits them all.
It is wrong once the install is split. On 2026-11-06 a single multiplexer ticked 30 profiles;
the scheduled fleet's turns were 1445 of the process's 1768 model calls (81.7%) and their
concurrency inflated every interactive call's latency (r=+0.69 between in-flight sessions and
latency: 6.4s median at zero concurrency, 37-149s at 3-8). The interactive CLI ran inside the
same event loop, so it also paid the fleet's blocked DB reads and its restart drains.

``cron.tick_profiles`` lets each gateway own a disjoint tick set, which is what makes a
two-gateway split (interactive vs scheduled) expressible at all — ``profiles_to_serve`` always
includes the default profile, so two multiplexers cannot otherwise coexist.

The gate still applies on top: ``gateway.run._cron_profile_gate`` holds a profile owned by
another live gateway, so an accidental overlap degrades to "one ticker" instead of a double fire.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def tick_profile_override(config: object) -> list[str]:
    """``cron.tick_profiles``: the explicit profile names THIS process ticks, or ``[]``.

    Absent/empty keeps the historical "everything I serve" behaviour, so an install that never
    sets it is unaffected. ``config`` is unused today (the key is read raw, like ``cron.provider``,
    because the gateway does not go through ``DEFAULT_CONFIG``) and is kept for call-site symmetry.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        raw = cfg_get(load_config(), "cron", "tick_profiles", default=None)
    except Exception:
        # Deliberate boundary: load_config can fail on a malformed/unreadable config. Falling back
        # to "tick everything I serve" is the safe direction — never leave cron with no ticker.
        logger.debug("cron.tick_profiles unreadable; ticking every served profile", exc_info=True)
        return []
    if isinstance(raw, str):
        raw = [part for part in (p.strip() for p in raw.split(",")) if part]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(name).strip() for name in raw if str(name).strip()]


def override_tick_homes(
    config: object, *, served: list[tuple[str, Path]],
) -> list[tuple[str, Path]] | None:
    """Resolved tick set when ``cron.tick_profiles`` names at least one live profile, else None.

    ``served`` is the set this process serves, supplied by the caller so the caller keeps its own
    seam (``gateway.run._multiplex_profile_homes``) patchable. Unknown names are dropped with a
    warning: a typo must not stop the REST of the fleet from firing. A list that resolves to
    nothing is a misconfiguration and falls back to the served set rather than going idle.
    """
    wanted = tick_profile_override(config)
    if not wanted:
        return None

    from hermes_cli.profiles import get_profile_dir, profile_exists

    resolved: list[tuple[str, Path]] = []
    missing: list[str] = []
    for name in wanted:
        if not profile_exists(name):
            missing.append(name)
            continue
        resolved.append((name, get_profile_dir(name)))
    if missing:
        logger.warning("cron.tick_profiles names unknown profile(s) %s; ignoring them", missing)
    if not resolved:
        logger.warning("cron.tick_profiles resolved no profile; falling back to the served set")
        return None

    # A split install fails silently when the other half is not running: name what this process
    # stopped ticking, so a half-applied cut shows up in the gateway log instead of as jobs that
    # quietly never fire again.
    ticked = {name for name, _home in resolved}
    unticked = sorted(name for name, _home in served if name not in ticked)
    if unticked:
        logger.warning(
            "cron.tick_profiles excludes %d profile(s) this process serves: %s — their jobs fire "
            "only if another gateway ticks them", len(unticked), unticked,
        )
    logger.info(
        "Cron tick set comes from cron.tick_profiles (%d profile(s)): %s",
        len(resolved), [name for name, _home in resolved],
    )
    return resolved
