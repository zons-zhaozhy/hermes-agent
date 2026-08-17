"""adaptive-reasoning plugin — per-turn adaptive reasoning effort.

Implements the "auto reasoning" behaviour requested upstream in
NousResearch/hermes-agent issues #74725 ("Auto/Adaptive reasoning level")
and #13663 ("Smart reasoning_effort routing based on task complexity"),
entirely at the plugin layer — zero core changes.

HOW IT WORKS
============
``llm_request`` middleware fires before EVERY provider call
(agent/conversation_loop.py:2556 ``apply_llm_request_middleware``), with the
final ``api_kwargs`` as ``request``. For models that carry reasoning config,
the transport already emitted ``extra_body.reasoning = {"enabled": True,
"effort": <level>}`` (agent/transports/chat_completions.py:560) or a
top-level ``reasoning_effort`` (Kimi/TokenHub/LM Studio). This middleware
rewrites that effort per call:

    turn complexity → effort level → extra_body.reasoning.effort

Complexity classification (deterministic, no LLM call — the issue's Option C
"LLM-driven auto-classification" costs a round trip per turn; we implement a
signal-based router that is free and reproducible):

  * USER-MESSAGE SIGNALS (evaluated on every API call):
    - complexity cues: message length, code fences, multi-step markers,
      architecture/debug/audit keywords
    - brevity cues: short imperative ("hi", "continue", "ok")
  * IN-TURN ESCALATION (api_call_count > 1, i.e. the tool loop):
    - tool errors accumulate via the post_tool_call hook → effort escalates
      (a failing tool loop IS a hard task, regardless of phrasing)

MEASURED EFFECT (glm-5.3, 2026-08-17, real API A/B)
==================================================
The effort dial is REAL on glm-5.3: ``low`` → reasoning_tokens = 0 (thinking
fully off), ``high`` → 26-395 tokens (server-side). Verified that medium
(default, no plugin) spends 6-183 reasoning tokens even on "What is the
capital of France?"-class turns.

KNOWN LIMIT — measured, not guessed: surface signals (length/keywords) have
a blind spot for SHORT-AND-HARD prompts (e.g. "Sort Mars, Venus, Europa,
Titan by orbital period"): classifier routes low (short, no keywords), but
the task is cognitively hard. On such prompts the plugin can COST accuracy
vs. a static medium (real loss observed: low → wrong, medium/high → right).
Mitigations already built in:
  * tool-error escalation rescues agentic tasks (errors → step up)
  * /reasoning <level> still overrides per-session (plugin clamps within
    floor/ceiling only; raise ``floor`` to ``medium`` in config to disable
    downgrades entirely while keeping escalation)
If 100% accuracy on adversarial short prompts matters more than token
savings, set ``floor: medium``.

The plugin NEVER:
  * exceeds the configured ceiling / drops below the configured floor.

SAFETY
======
* No-op when the request carries no reasoning fields (model doesn't think,
  or the provider profile routes the config elsewhere) — we only rewrite
  what the transport already decided to send.
* Off-switch: ``plugins.entries.adaptive-reasoning.enabled: false`` in
  config.yaml.
* Effort levels validated against the VALID_REASONING_EFFORTS scale.
* Never touches ``messages`` / the prompt-cache prefix — only the effort
  field, which is request-scoped (outside the cached prefix).

CONFIG (~/.hermes/config.yaml)
=============================
    plugins:
      entries:
        adaptive-reasoning:
          enabled: true        # master switch (default: enabled)
    agent:
      adaptive_reasoning:      # optional tuning; defaults shown
        low_max_chars: 120     # shorter user msg → low
        high_min_chars: 600    # longer user msg → high
        ceiling: high          # never exceed this level adaptively
        floor: minimal         # never go below this level adaptively
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Effort scale (mirrors hermes_constants.VALID_REASONING_EFFORTS) ──────

EFFORT_SCALE: List[str] = [
    "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
]

# Complexity keywords → bump effort above what length alone gives.
# Kept deliberately small: false positives on an innocent message raise cost.
# NOTE: two patterns, NOT one — \b word boundaries don't exist for CJK
# (no word characters adjacent to Han), so a single \b(...|排查|...)\b would
# silently never match Chinese keywords (verified: re.findall returned []).
_COMPLEXITY_KEYWORDS_EN = re.compile(
    r"(?ix)\b("
    r"why|debug|root\s*cause|diagnos|investigat|audit|review|architect|"
    r"refactor|redesign|migrat|optimi[sz]e|race\s*condition|deadlock|"
    r"consisten|invariant|regression|failure|trace|analy[sz]e"
    r")\b"
)
_COMPLEXITY_KEYWORDS_ZH = re.compile(
    "为什么|排查|诊断|审计|架构|重构|根因|分析|设计|迁移|优化|死锁|竞态"
)

# Brevity markers → hard-low regardless of keyword presence (short + keyword
# still escalates; short + nothing = trivial continuation turn).
_BREVITY_MARKERS = re.compile(
    r"(?i)^(ok|okay|go|yes|no|continue|cont|继续|好|好的|嗯|行|next|same|"
    r"again|retry|重试|再来)[.!\s]*$"
)

# Code fences / diffs / stack traces are strong "real work" signals.
_TECHNICAL_MARKER = re.compile(r"(```|\bdiff\b|\btraceback\b|Error:|异常|堆栈)")


def _effort_index(level: str) -> int:
    """Return the scale index of *level*; ``medium`` for unknown values.

    Returns:
        Index into EFFORT_SCALE.
    """
    try:
        return EFFORT_SCALE.index(level)
    except ValueError:
        return EFFORT_SCALE.index("medium")


def _clamp(level: str, floor: str, ceiling: str) -> str:
    """Clamp *level* into [floor, ceiling] on the effort scale.

    Returns:
        The clamped effort level string.
    """
    lo, hi = _effort_index(floor), _effort_index(ceiling)
    return EFFORT_SCALE[max(lo, min(hi, _effort_index(level)))]


# ── Config resolution ────────────────────────────────────────────────────

_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_KEY: Optional[tuple] = None


def _default_config() -> Dict[str, Any]:
    return {
        "low_max_chars": 120,
        "high_min_chars": 600,
        "ceiling": "high",
        "floor": "minimal",
    }


def _resolve_config() -> Dict[str, Any]:
    """Load tuning config from config.yaml ``agent.adaptive_reasoning``.

    Cached on (mtime_ns, size) of config.yaml — same pattern the core uses,
    so an on-disk edit takes effect on the next API call without restart.

    Returns:
        Effective config dict with validated floor/ceiling.
    """
    global _CONFIG_CACHE, _CONFIG_CACHE_KEY
    cfg = dict(_default_config())
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        stat = config_path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        return cfg  # no config.yaml on disk — defaults are the config
    except Exception as exc:
        logger.warning(
            "adaptive-reasoning: config.yaml stat failed, using defaults: %s",
            exc, exc_info=True,
        )
        return cfg
    with _CONFIG_LOCK:
        if _CONFIG_CACHE is not None and _CONFIG_CACHE_KEY == key:
            return dict(_CONFIG_CACHE)
    try:
        from hermes_cli.config import read_user_config_raw
        raw = read_user_config_raw() or {}
        section = (raw.get("agent") or {}).get("adaptive_reasoning") or {}
        if isinstance(section, dict):
            for k in cfg:
                if k in section and section[k] is not None:
                    cfg[k] = section[k]
        cfg["ceiling"] = _clamp(str(cfg["ceiling"]).lower(), "minimal", "ultra")
        cfg["floor"] = _clamp(str(cfg["floor"]).lower(), "minimal", "ultra")
        if _effort_index(cfg["floor"]) > _effort_index(cfg["ceiling"]):
            cfg["floor"] = cfg["ceiling"]
    except Exception as exc:
        logger.warning(
            "adaptive-reasoning: config load failed, using defaults: %s",
            exc, exc_info=True,
        )
        return cfg
    with _CONFIG_LOCK:
        _CONFIG_CACHE, _CONFIG_CACHE_KEY = dict(cfg), key
    return dict(cfg)


def _plugin_enabled() -> bool:
    """Master switch: plugins.entries.adaptive-reasoning.enabled (default on).

    Returns:
        False only on an explicit ``enabled: false`` entry.
    """
    try:
        from hermes_cli.config import read_user_config_raw
        raw = read_user_config_raw() or {}
        entries = ((raw.get("plugins") or {}).get("entries") or {})
        entry = entries.get("adaptive-reasoning") or {}
        if isinstance(entry, dict) and entry.get("enabled") is False:
            return False
    except Exception as exc:
        logger.warning(
            "adaptive-reasoning: enabled-check failed, treating as enabled: %s",
            exc, exc_info=True,
        )
    return True


# ── Turn state ───────────────────────────────────────────────────────────
# Per-session in-turn tool-error counters. The post_tool_call hook bumps
# them; the middleware reads them per API call and drops stale turns.

_STATE_LOCK = threading.Lock()
_TURN_ERRORS: Dict[str, int] = {}


def _state_key(session_id: str, turn_id: str) -> str:
    return f"{session_id}:{turn_id}"


def on_post_tool_call(**kwargs: Any) -> None:
    """post_tool_call observer: count failed tool calls this turn."""
    status = str(kwargs.get("status") or kwargs.get("outcome") or "")
    if status.lower() not in {"error", "failed", "exception"}:
        return
    key = _state_key(
        str(kwargs.get("session_id") or ""),
        str(kwargs.get("turn_id") or ""),
    )
    with _STATE_LOCK:
        _TURN_ERRORS[key] = _TURN_ERRORS.get(key, 0) + 1


def _error_count_for_turn(session_id: str, turn_id: str) -> int:
    with _STATE_LOCK:
        return _TURN_ERRORS.get(_state_key(session_id, turn_id), 0)


def _forget_stale_turns(session_id: str, turn_id: str) -> None:
    """Drop state entries older than the current turn (bounded memory)."""
    cur = _state_key(session_id, turn_id)
    with _STATE_LOCK:
        stale = [
            k for k in _TURN_ERRORS if k.startswith(f"{session_id}:") and k != cur
        ]
        for k in stale:
            _TURN_ERRORS.pop(k, None)


# ── Complexity classification ────────────────────────────────────────────

def classify_effort(user_message: str, *, tool_errors: int = 0,
                    cfg: Optional[Dict[str, Any]] = None) -> str:
    """Return the adaptive effort level for this API call.

    Deterministic signal router (no LLM round trip):

    * Base from message shape (length + complexity keywords + technical
      markers vs brevity markers).
    * In-turn escalation: every 2 accumulated tool errors bumps one step
      (a loop that keeps failing is a hard task regardless of phrasing).

    Args:
        user_message: The turn's user message text.
        tool_errors: Failed tool calls accumulated this turn so far.
        cfg: Tuning config (defaults used when None).

    Returns:
        Effort level string, NOT yet clamped to floor/ceiling.
    """
    cfg = cfg or _default_config()
    msg = (user_message or "").strip()
    low_max = int(cfg.get("low_max_chars", 120) or 120)
    high_min = int(cfg.get("high_min_chars", 600) or 600)

    kw_hits = (
        len(set(_COMPLEXITY_KEYWORDS_EN.findall(msg)))
        + len(set(_COMPLEXITY_KEYWORDS_ZH.findall(msg)))
    )
    technical = bool(_TECHNICAL_MARKER.search(msg))
    brevity = bool(_BREVITY_MARKERS.match(msg))
    n = len(msg)

    if brevity and not kw_hits:
        level = "minimal"
    elif n <= low_max and not kw_hits and not technical:
        level = "low"
    elif n >= high_min or kw_hits >= 2 or (kw_hits >= 1 and technical):
        level = "high"
    elif kw_hits == 1 or technical:
        level = "medium"
    else:
        level = "low"

    # In-turn escalation: failing tool loop ⇒ harder than phrasing suggests.
    steps = tool_errors // 2
    if steps > 0:
        idx = min(_effort_index(level) + steps, len(EFFORT_SCALE) - 1)
        level = EFFORT_SCALE[idx]
    return level


# ── Middleware ───────────────────────────────────────────────────────────

def _translate_effort(level: str, provider: str, model: str) -> Optional[str]:
    """Map a Hermes-scale effort onto the provider's native wire value.

    The middleware fires AFTER the transport built the request kwargs
    (chat_completion_helpers.build_kwargs → conversation_loop
    apply_llm_request_middleware), so a top-level ``reasoning_effort`` in
    the request is already in the provider's native scale (zai glm-5.3:
    low/high/max; kimi: low/medium/high — written by the provider profile's
    build_api_kwargs_extras). Writing a raw Hermes level there would send
    values the provider rejects.

    Fix: run the SAME profile mapping the transport used. Whatever the
    profile emits for top_level["reasoning_effort"] is by construction a
    legal wire value for this provider+model.

    Returns:
        Native effort string, or None when the provider has no profile or
        cannot express *level* (caller treats that as no-op).
    """
    if not provider:
        return None
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider)
    except Exception as exc:
        logger.warning(
            "adaptive-reasoning: provider profile lookup failed for %r: %s",
            provider, exc, exc_info=True,
        )
        return None
    if profile is None:
        return None
    try:
        _eb, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": level},
            model=model,
        )
    except Exception as exc:
        logger.warning(
            "adaptive-reasoning: profile mapping failed for %r/%r: %s",
            provider, model, exc, exc_info=True,
        )
        return None
    return top_level.get("reasoning_effort")


def _effort_targets(
    request: Dict[str, Any], level: str, provider: str, model: str,
) -> Dict[str, str]:
    """Compute which wire fields this plugin may rewrite, and their targets.

    Shapes handled (mirroring what the transports emit):
    * ``extra_body.reasoning.effort`` — passthrough shape (the OpenRouter
      profile passes the full reasoning_config dict through), so the raw
      Hermes level is legal there.
    * top-level ``reasoning_effort`` — provider-native scale; rewritten
      ONLY with the profile-translated value (see _translate_effort).
    * top-level ``verbosity`` — OpenRouter's Claude effort lever (the OR
      profile maps effort onto verbosity for reasoning-mandatory Claude
      models); the profile emits it at Hermes scale, so pass through.

    Anything else (Gemini thinking_config, Anthropic thinking budget,
    kimi's extra_body.thinking toggle, unknown providers) is left untouched.

    Returns:
        Mapping of dotted field path → target value; empty = no-op.
    """
    targets: Dict[str, str] = {}
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict):
        reasoning = extra_body.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("enabled") is not False:
            targets["extra_body.reasoning.effort"] = level

    if isinstance(request.get("reasoning_effort"), str):
        native = _translate_effort(level, provider, model)
        if isinstance(native, str):
            targets["reasoning_effort"] = native
    elif isinstance(request.get("verbosity"), str):
        targets["verbosity"] = level
    return targets


def _rewrite_reasoning(
    request: Dict[str, Any], level: str, provider: str, model: str,
) -> Optional[Dict[str, Any]]:
    """Return rewritten request with the adapted effort applied, or None.

    Returns:
        New request dict when any target field actually changes value,
        else None.
    """
    targets = _effort_targets(request, level, provider, model)
    if not targets:
        return None

    new = dict(request)
    extra_target = targets.get("extra_body.reasoning.effort")
    if extra_target is not None:
        extra_body = new.get("extra_body")
        reasoning = (extra_body or {}).get("reasoning") or {}
        merged = dict(reasoning)
        merged["effort"] = extra_target
        eb = dict(extra_body or {})
        eb["reasoning"] = merged
        new["extra_body"] = eb

    if "reasoning_effort" in targets and isinstance(new.get("reasoning_effort"), str):
        new["reasoning_effort"] = targets["reasoning_effort"]
    if "verbosity" in targets and isinstance(new.get("verbosity"), str):
        new["verbosity"] = targets["verbosity"]

    # No-op when nothing actually changed value
    for path, value in targets.items():
        old = request
        for part in path.split("."):
            old = old.get(part) if isinstance(old, dict) else None
        if old != value:
            return new
    return None


def adaptive_llm_request_middleware(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """llm_request middleware: rewrite reasoning effort for this call.

    Returns:
        ``{"request": <rewritten>}`` when the effort changed, else None.
    """
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return None
    if not _plugin_enabled():
        return None

    session_id = str(kwargs.get("session_id") or "")
    turn_id = str(kwargs.get("turn_id") or "")
    api_call_count = int(kwargs.get("api_call_count") or 1)
    provider = str(kwargs.get("provider") or "")
    # The middleware call site passes model=agent.model, but tests / future
    # call sites may omit it — the request payload itself carries the model.
    model = str(kwargs.get("model") or "")
    if not model:
        model = str(request.get("model") or "")
    # The llm_request middleware call site (conversation_loop.py) does NOT
    # pass user_message (only the observer hook pre_api_request does). The
    # turn's user message is recoverable from the request payload itself:
    # the LAST user-role message in api_kwargs["messages"].
    user_message = str(kwargs.get("user_message") or "")
    if not user_message:
        user_message = _last_user_message(request)

    _forget_stale_turns(session_id, turn_id)
    tool_errors = _error_count_for_turn(session_id, turn_id)

    cfg = _resolve_config()
    level = classify_effort(
        user_message,
        tool_errors=tool_errors,
        cfg=cfg,
    )
    level = _clamp(level, str(cfg["floor"]), str(cfg["ceiling"]))

    rewritten = _rewrite_reasoning(request, level, provider, model)
    if rewritten is not None:
        logger.debug(
            "adaptive-reasoning: effort → %s (tool_errors=%d)",
            level, tool_errors,
        )
        return {"request": rewritten}
    return None


def _last_user_message(request: Dict[str, Any]) -> str:
    """Extract the turn's user message text from the request payload.

    The middleware call site doesn't pass ``user_message`` as a kwarg, but
    ``request["messages"]`` is the exact API-bound message list — the last
    ``role == "user"`` entry is this turn's user message (tool results arrive
    as ``role == "tool"``). Multimodal content lists yield their text parts.

    Returns:
        Concatenated text of the last user message, or "" when absent.
    """
    messages = request.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(p.get("text") or "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(x for x in parts if x)
        return ""
    return ""


def register(ctx) -> None:  # noqa: ANN001 — PluginContext from loader
    """Plugin entry point: register middleware + observer hook."""
    ctx.register_middleware("llm_request", adaptive_llm_request_middleware)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    logger.info("adaptive-reasoning registered (issues #74725 / #13663)")
