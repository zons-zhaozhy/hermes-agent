"""
Subagent Model Tier Resolution — cost-aware model selection per task.

Inspired by vLLM Semantic Router's Confidence pattern (cost-aware escalation)
and Superpowers' model selection strategy (use the cheapest model that can
handle each role).  The key insight from Superpowers: "turn count beats token
price — the cheapest models routinely take 2-3x the turns on multi-step work."

This module provides tier-based model resolution for delegate_task:

- ``tier=cheap``     → fast/cheap model for mechanical tasks (1-2 files,
                       clear spec, transcription + testing)
- ``tier=standard``  → mid-tier model for integration tasks (multi-file
                       coordination, pattern matching)
- ``tier=capable``   → most capable model for architecture/design tasks

Config (config.yaml ``delegation:`` section):

.. code-block:: yaml

    delegation:
      tiers:
        cheap:
          model: "deepseek/deepseek-chat"
          provider: "openrouter"
        standard:
          model: "anthropic/claude-sonnet-4"
          provider: "openrouter"
        capable:
          model: "anthropic/claude-opus-4"
          provider: "openrouter"

When a tier is not configured, it falls back to the parent model (standard)
or the delegation.provider/model config. This means P2 is purely opt-in:
if no tiers are configured, behavior is identical to before.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Valid tier values
VALID_TIERS = frozenset({"cheap", "standard", "capable"})


def _load_tier_config() -> Dict[str, Dict[str, Any]]:
    """Load tier configuration from delegation config.

    Returns a dict like::

        {
            "cheap": {"model": "deepseek/...", "provider": "openrouter", ...},
            "standard": {"model": "...", "provider": "...", ...},
            "capable": {"model": "...", "provider": "...", ...},
        }

    Missing tiers return an empty dict (caller falls back to parent model).
    """
    try:
        from tools.delegate_tool import _load_config

        cfg = _load_config()
        tiers = cfg.get("tiers")
        if not isinstance(tiers, dict):
            return {}
        # Validate each tier has at least a model
        result: Dict[str, Dict[str, Any]] = {}
        for tier_name, tier_cfg in tiers.items():
            if tier_name not in VALID_TIERS:
                logger.warning(
                    "Unknown delegation tier %r — valid tiers: %s",
                    tier_name,
                    ", ".join(sorted(VALID_TIERS)),
                )
                continue
            if not isinstance(tier_cfg, dict):
                continue
            model = tier_cfg.get("model")
            if not model:
                continue
            result[tier_name] = {
                "model": str(model),
                "provider": tier_cfg.get("provider"),
                "base_url": tier_cfg.get("base_url"),
                "api_key": tier_cfg.get("api_key"),
                "api_mode": tier_cfg.get("api_mode"),
            }
        return result
    except Exception:
        logger.warning("Failed to load tier config", exc_info=True)
        return {}


def resolve_tier_credentials(
    tier: Optional[str],
    base_creds: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve credentials for a given tier.

    If tier is None, empty, or not configured, returns ``base_creds``
    unchanged (identical to pre-tier behavior).

    If tier is configured, overlays the tier's model/provider/base_url
    onto the base credentials.  Fields not specified in the tier config
    are inherited from ``base_creds``.

    Parameters
    ----------
    tier : str or None
        One of "cheap", "standard", "capable", or None/empty for default.
    base_creds : dict
        The credentials dict from ``_resolve_delegation_credentials``.
        This is the base — tier overrides are applied on top.

    Returns
    -------
    dict
        Credentials with tier overrides applied.  Same shape as
        ``base_creds`` (model, provider, base_url, api_key, api_mode, etc.)
    """
    if not tier:
        return base_creds

    tier_norm = str(tier).strip().lower()
    if tier_norm not in VALID_TIERS:
        logger.warning("Unknown tier %r, ignoring (valid: %s)", tier, ", ".join(sorted(VALID_TIERS)))
        return base_creds

    tiers = _load_tier_config()
    tier_cfg = tiers.get(tier_norm)
    if not tier_cfg:
        # Tier requested but not configured — fail loud, don't silently
        # use the wrong model.  Log a warning and return base_creds so the
        # task still runs (fail-open, but visible).
        logger.warning(
            "Tier %r requested but not configured in delegation.tiers — "
            "falling back to base model. Configure delegation.tiers.%s "
            "in config.yaml to use this tier.",
            tier_norm,
            tier_norm,
        )
        return base_creds

    # Overlay tier-specific overrides onto base creds
    result = dict(base_creds)  # shallow copy
    result["model"] = tier_cfg.get("model") or base_creds.get("model")
    if tier_cfg.get("provider"):
        result["provider"] = tier_cfg["provider"]
    if tier_cfg.get("base_url"):
        result["base_url"] = tier_cfg["base_url"]
    if tier_cfg.get("api_key"):
        result["api_key"] = tier_cfg["api_key"]
    if tier_cfg.get("api_mode"):
        result["api_mode"] = tier_cfg["api_mode"]

    logger.info(
        "Tier %r resolved: model=%s provider=%s",
        tier_norm,
        result.get("model"),
        result.get("provider"),
    )
    return result


def resolve_per_task_tier(
    task: Dict[str, Any],
    default_tier: Optional[str],
) -> Optional[str]:
    """Extract tier from a task dict, falling back to default.

    Per-task ``tier`` overrides the batch-level default.  This lets a batch
    mix cheap and capable tasks::

        tasks=[
            {"goal": "write unit test for foo()", "tier": "cheap"},
            {"goal": "design the auth architecture", "tier": "capable"},
        ]
    """
    task_tier = task.get("tier")
    if task_tier:
        tier_norm = str(task_tier).strip().lower()
        if tier_norm in VALID_TIERS:
            return tier_norm
        logger.warning("Task has unknown tier %r, ignoring", task_tier)
    return default_tier
