"""tool-output-trim — keep context growth bounded at the LLM request seam.

Problem: Hermes has no built-in cap on tool-result size. A single verbose
terminal/log read inflates the context permanently; prefill time and
compaction frequency grow for the rest of the session.

Approach (mirrors adaptive-reasoning's proven plugin skeleton):
  * ``llm_request`` middleware inspects request["messages"] right before
    the provider call and truncates *historical* tool results with a
    head+tail window (both ends preserved — errors and summaries tend to
    live at the tail, invocation context at the head).
  * Only messages older than ``keep_recent`` tool results are trimmed;
    the freshest results stay verbatim so the model can act on them.
  * Pure send-side view: session log on disk is untouched; a re-read or
    session_search always recovers the full original text.
  * Idempotent: trimmed blocks are marked and never re-trimmed, so sizes
    are stable across retries/relay replays.

Config under ``agent.tool_output_trim`` in config.yaml (mtime-cached like
adaptive-reasoning, effective on next API call without restart):
  enabled: true        # master switch (plugins.entries also honored)
  max_chars: 3000      # per-tool-result budget in historical messages
  keep_recent: 3       # newest tool results kept verbatim
  head_chars: 1200     # leading slice kept
  tail_chars: 1200     # trailing slice kept
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.plugin.tool-output-trim")

_TRIM_MARKER = "[tool-output-trim "


def _default_config() -> Dict[str, Any]:
    return {
        "max_chars": 3000,
        "keep_recent": 3,
        "head_chars": 1200,
        "tail_chars": 1200,
    }


# ── config (mtime-cached; same contract as adaptive-reasoning) ────────────

_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_KEY: Optional[tuple] = None


def _resolve_config() -> Dict[str, Any]:
    """Load tuning config from ``agent.tool_output_trim``.

    Cached on (mtime_ns, size) of config.yaml; an on-disk edit takes
    effect on the next API call without restart.

    Returns:
        Effective config dict with positive-int coercion.
    """
    global _CONFIG_CACHE, _CONFIG_CACHE_KEY
    cfg = dict(_default_config())
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        stat = config_path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        return cfg
    except Exception as exc:
        logger.warning("tool-output-trim: config stat failed, defaults: %s", exc)
        return cfg
    with _CONFIG_LOCK:
        if _CONFIG_CACHE is not None and _CONFIG_CACHE_KEY == key:
            return dict(_CONFIG_CACHE)
    try:
        from hermes_cli.config import read_user_config_raw
        raw = read_user_config_raw() or {}
        section = (raw.get("agent") or {}).get("tool_output_trim") or {}
        if isinstance(section, dict):
            for k in cfg:
                if section.get(k) is not None:
                    cfg[k] = section[k]
        for k in ("max_chars", "keep_recent", "head_chars", "tail_chars"):
            cfg[k] = max(1, int(cfg[k]))
        if cfg["head_chars"] + cfg["tail_chars"] > cfg["max_chars"]:
            # keep head priority; tail gets the remainder
            cfg["tail_chars"] = max(1, cfg["max_chars"] - cfg["head_chars"])
    except Exception as exc:
        logger.warning("tool-output-trim: config load failed, defaults: %s", exc)
        return cfg
    with _CONFIG_LOCK:
        _CONFIG_CACHE, _CONFIG_CACHE_KEY = dict(cfg), key
    return dict(cfg)


def _plugin_enabled() -> bool:
    """Master switch: explicit ``enabled: false`` in either
    ``plugins.entries.tool-output-trim.enabled`` or
    ``agent.tool_output_trim.enabled`` disables trimming.

    Returns:
        False only on an explicit enabled=false; default on.
    """
    try:
        from hermes_cli.config import read_user_config_raw
        raw = read_user_config_raw() or {}
        entry = ((raw.get("plugins") or {}).get("entries") or {}).get("tool-output-trim") or {}
        if isinstance(entry, dict) and entry.get("enabled") is False:
            return False
        section = (raw.get("agent") or {}).get("tool_output_trim") or {}
        if isinstance(section, dict) and section.get("enabled") is False:
            return False
    except Exception as exc:
        logger.warning("tool-output-trim: enabled-check failed, on: %s", exc)
    return True


def _content_text(content: Any) -> str:
    """Flatten an OpenAI-style content field to text.

    Args:
        content: str, None, or list of {text|...} parts.

    Returns:
        Text or "" — never raises.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    return ""


def _trim_text(text: str, cfg: Dict[str, Any]) -> str:
    """Head+tail window with a size marker.

    Args:
        text: original tool-result text.
        cfg: resolved config.

    Returns:
        Trimmed text with "[tool-output-trim N→M chars]" header when cut.
    """
    if len(text) <= cfg["max_chars"]:
        return text
    head = text[: cfg["head_chars"]]
    tail = text[-cfg["tail_chars"]:]
    return (
        f"{_TRIM_MARKER}{len(text)}→{cfg['head_chars'] + cfg['tail_chars']} chars]\n"
        f"{head}\n…[middle omitted]…\n{tail}"
    )


def trim_outputs_middleware(request: Dict[str, Any], **context: Any) -> Optional[Dict[str, Any]]:
    """llm_request middleware: bound historical tool-result size.

    Returns:
        ``{"request": request}`` per the llm_request middleware contract
        (the payload is replaced only on an explicit ``{"request": ...}``
        return); None leaves the request untouched.
    """
    try:
        if not _plugin_enabled():
            return None
        cfg = _resolve_config()
        messages = request.get("messages")
        if not isinstance(messages, list):
            return None
        tool_idx = [
            i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        # tool_idx is oldest→newest by message index; protect the NEWEST
        # keep_recent results and trim everything older.
        trim_targets = (
            tool_idx[: len(tool_idx) - cfg["keep_recent"]]
            if cfg["keep_recent"]
            else tool_idx
        )
        for i in trim_targets:
            msg = messages[i]
            content = msg.get("content")
            if isinstance(content, str):
                if not content.startswith(_TRIM_MARKER):
                    msg["content"] = _trim_text(content, cfg)
            elif isinstance(content, list):
                # trim each text part; skip non-text (image etc.)
                msg["content"] = [
                    {**p, "text": _trim_text(p.get("text", ""), cfg)}
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                    and not p["text"].startswith(_TRIM_MARKER)
                    else p
                    for p in content
                ]
        return {"request": request}
    except Exception as exc:
        logger.warning("tool-output-trim: middleware failed, passing through: %s", exc)
        return None


def register(ctx) -> None:  # noqa: ANN001 — PluginContext from loader
    """Plugin entry point: register the send-side trimming middleware."""
    ctx.register_middleware("llm_request", trim_outputs_middleware)
    logger.info("tool-output-trim registered")
