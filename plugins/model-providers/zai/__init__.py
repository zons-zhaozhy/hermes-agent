"""ZAI / GLM provider profile.

Z.AI's GLM-4.5-and-later chat models default to thinking-mode ON when the
request omits ``thinking``.  Hermes' ``reasoning_config = {"enabled": False}``
was previously a silent no-op on this route — the base profile emits nothing,
so users who turned thinking off (desktop toggle, ``/reasoning none``,
``reasoning_effort: none``/``false`` in config.yaml) kept burning thinking
tokens on every turn.

:meth:`ZaiProfile.build_api_kwargs_extras` translates the Hermes reasoning
config into the wire shape Z.AI's OpenAI-compat endpoint expects:

    {"extra_body": {"thinking": {"type": "enabled" | "disabled"}}}

When no reasoning preference is set (``reasoning_config is None``) the field
is omitted so the server default applies, matching prior behavior.  GLM
models before 4.5 (e.g. ``glm-4-9b``) don't accept ``thinking`` and are left
untouched.

GLM-5.2 additionally exposes a native ``reasoning_effort`` knob with exactly
two enabled levels — ``high`` and ``max`` — on the OpenAI-compatible endpoint
(per Z.AI / BigModel docs).  Hermes' richer effort scale is collapsed onto
those two so the user's effort preference actually reaches the model instead
of being silently dropped.

GLM-5.3 (released 2026-08-14) changes the contract again: thinking can no
longer be disabled at all — ``thinking.type: "disabled"`` fails the whole
request — and the effort knob gains a third ``low`` tier (``low`` / ``high``
/ ``max``, default ``max``).  Per Z.AI's migration guidance, a request that
would have disabled thinking is translated to ``thinking.type: "enabled"``
plus ``reasoning_effort: "low"`` (the cheapest tier) instead of being sent
as-is and rejected server-side.
"""

import re
from typing import Any

from agent import reasoning_effort as re_
from providers import register_provider
from providers.base import ProviderProfile

_GLM_VERSION_RE = re.compile(r"^glm-(\d+)(?:\.(\d+))?")
# Alias spellings seen on relays (Fireworks ``glm-5p2``, ``zai-org-glm-5-2``…).
_GLM_5_3_TOKENS = ("glm-5.3", "glm-5-3", "glm-5p3")
_GLM_5_2_TOKENS = ("glm-5.2", "glm-5-2", "glm-5p2") + _GLM_5_3_TOKENS


def _model_supports_thinking(model: str | None) -> bool:
    """GLM thinking-capable model families: glm-4.5 and later (4.5, 4.6, 5…)."""
    match = _GLM_VERSION_RE.match((model or "").strip().lower())
    return bool(match) and (int(match.group(1)), int(match.group(2) or 0)) >= (4, 5)


def _has_token(model: str | None, tokens: tuple[str, ...]) -> bool:
    m = (model or "").strip().lower()
    if not m:
        return False
    return any(token in m for token in ("glm-5.3", "glm-5-3", "glm-5p3"))


def _glm_5_2_reasoning_effort(
    reasoning_config: dict | None, *, model: str | None = None
) -> str | None:
    """Map Hermes reasoning effort onto GLM's native vocabulary.

    GLM-5.2 supports two enabled effort levels (``high``/``max``);
    GLM-5.3 supports the graded ``low``/``medium``/``high``/``max`` scale.
    ``xhigh``/``max``/``ultra`` request the top tier; anything below the
    model's floor clamps to that floor. When reasoning is explicitly
    disabled, or no effort preference is supplied, the server default is
    left untouched.
    """
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        # GLM-5.3 cannot disable thinking: map "off" to the cheapest legal
        # tier (low) instead of leaving the server-default effort in place.
        return "low" if _is_glm_5_3(model) else None


def _glm_5_2_reasoning_effort(reasoning_config: dict | None, *, model: str | None = None) -> str | None:
    """Hermes effort -> GLM vocabulary (5.2: high/max; 5.3: low..max). Below-floor
    efforts clamp to the floor; disabled/unset leaves the server default."""
    effort = re_.requested_effort(reasoning_config)
    if effort is None or effort == "none":
        return None
    if _has_token(model, _GLM_5_3_TOKENS):
        efforts, overrides, floor = re_.GLM53_EFFORTS, re_.GLM53_OVERRIDES, "low"
    else:
        efforts, overrides, floor = re_.GLM52_EFFORTS, re_.GLM52_OVERRIDES, "high"
    clamped = re_.clamp_effort(effort, efforts, overrides)
    return clamped if clamped in efforts else floor



class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — extra_body.thinking on/off + GLM-5.2/5.3 reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_thinking(model) and not _is_glm_5_2(model) and not _is_glm_5_3(model):
            return extra_body, top_level
        # Only emit when the user expressed a preference (server default = enabled).
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled") is not False
            if not enabled and _is_glm_5_3(model):
                # GLM-5.3 rejects thinking.type=disabled outright; send the
                # official migration shape and let the effort knob below
                # select the cheapest tier (low).
                extra_body["thinking"] = {"type": "enabled"}
            else:
                extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

        if _is_glm_5_2(model):
            effort = _glm_5_2_reasoning_effort(reasoning_config, model=model)
            if effort is not None:
                top_level["reasoning_effort"] = effort
        return extra_body, top_level


zai = ZaiProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.3",
        "glm-5.2",
        "glm-5",
        "glm-4-9b",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)
