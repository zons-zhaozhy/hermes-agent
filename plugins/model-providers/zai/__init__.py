"""ZAI / GLM provider profile.

GLM-4.5+ defaults to thinking ON, so ``reasoning_config`` is translated to
``extra_body.thinking``; GLM-5.2/5.3 also take a native ``reasoning_effort``.
"""

import re
from typing import Any

from agent import reasoning_effort as re_
from providers import register_provider
from providers.base import ProviderProfile

_GLM_VERSION_RE = re.compile(r"glm-(\d+)(?:[.\-p](\d+))?")  # re-ok: 版本号提取,无结构化解析器可用
# Semantic tier gates — compare version tuples instead of enumerating alias
# spellings, so glm-5.4 / glm-6 fall into the right contract branch
# automatically (relay prefixes like "z-ai/glm-5.3" also match: the pattern
# is unanchored).
_GLM_5_3_MIN = (5, 3)
_GLM_5_2_MIN = (5, 2)


def _model_glm_version(model: str | None) -> tuple[int, int] | None:
    """Best GLM (major, minor) mentioned anywhere in the model string."""
    best: tuple[int, int] | None = None
    for match in _GLM_VERSION_RE.finditer((model or "").strip().lower()):
        version = (int(match.group(1)), int(match.group(2) or 0))
        if best is None or version > best:
            best = version
    return best


def _model_supports_thinking(model: str | None) -> bool:
    """GLM thinking-capable model families: glm-4.5 and later (4.5, 4.6, 5…)."""
    version = _model_glm_version(model)
    return version is not None and version >= (4, 5)


def _is_glm_5_3(model: str | None) -> bool:
    version = _model_glm_version(model)
    return version is not None and version >= _GLM_5_3_MIN


def _is_glm_5_2(model: str | None) -> bool:
    version = _model_glm_version(model)
    return version is not None and version >= _GLM_5_2_MIN


def _glm_native_reasoning_effort(reasoning_config: dict | None, *, model: str | None = None) -> str | None:
    """Hermes effort -> GLM vocabulary (5.2: high/max; 5.3: low..max). Below-floor
    efforts clamp to the floor; disabled/unset leaves the server default."""
    is_5_3 = _is_glm_5_3(model)
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        # GLM-5.3 cannot disable thinking: map "off" to the cheapest legal
        # tier (low) instead of leaving the server-default effort in place.
        return "low" if is_5_3 else None
    effort = re_.requested_effort(reasoning_config)
    if effort is None or effort == "none":
        return None
    if is_5_3:
        efforts, overrides, floor = re_.GLM53_EFFORTS, re_.GLM53_OVERRIDES, "low"
    else:
        efforts, overrides, floor = re_.GLM52_EFFORTS, re_.GLM52_OVERRIDES, "high"
    clamped = re_.clamp_effort(effort, efforts, overrides)
    return clamped if clamped in efforts else floor


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — extra_body.thinking on/off + GLM-5.2 reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        is_5_2 = _is_glm_5_2(model)
        if not _model_supports_thinking(model) and not is_5_2:
            return extra_body, top_level
        # Only emit when the user expressed a preference (server default = enabled).
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled") is not False
            if not enabled and _is_glm_5_3(model):
                # GLM-5.3 rejects thinking.type=disabled outright; send the
                # official migration shape and let the effort knob above
                # select the cheapest tier (low).
                extra_body["thinking"] = {"type": "enabled"}
            else:
                extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if is_5_2:
            effort = _glm_native_reasoning_effort(reasoning_config, model=model)
            if effort is not None:
                top_level["reasoning_effort"] = effort
        return extra_body, top_level


zai = ZaiProfile(
    name="zai", aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models", signup_url="https://z.ai/",
    fallback_models=("glm-5.3", "glm-5.2", "glm-5", "glm-4-9b"), base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)
