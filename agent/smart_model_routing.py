"""Helpers for optional cheap-vs-strong model routing."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, Optional

from utils import is_truthy_value

# Complex task signals: +10 points per signal (English)
_COMPLEX_KEYWORDS_EN = {
    "debug", "debugging", "implement", "implementation", "refactor",
    "patch", "traceback", "stacktrace", "exception", "error",
    "analyze", "analysis", "investigate", "architecture", "design",
    "compare", "benchmark", "optimize", "optimise", "review",
    "terminal", "shell", "tool", "tools",
    "pytest", "test", "tests", "plan", "planning",
    "delegate", "subagent", "cron", "docker", "kubernetes",
}

# Complex task signals: +10 points per signal (Chinese, substring matched)
_COMPLEX_KEYWORDS_ZH = [
    "调试", "重构", "实现", "分析", "架构", "设计",
    "比较", "优化", "审查", "审查", "排查", "调查",
    "实现", "规划", "计划", "测试", "委托",
    "修复", "补丁", "异常", "错误", "报错",
    "性能", "部署", "容器", "迁移",
]

# Simple task signals: -5 points per signal (English, word-boundary matched)
_SIMPLE_KEYWORDS_EN = {
    "what", "why", "how", "where", "when", "which",
    "current", "status", "config", "list", "show",
    "read", "check", "version",
}

# Simple task signals: -5 points per signal (Chinese, substring matched)
_SIMPLE_KEYWORDS_ZH = [
    "是什么", "为什么", "怎么样", "如何", "能否", "可以",
    "有没有", "哪些", "当前", "现在", "目前",
    "状态", "配置", "设置", "版本",
    "列出", "显示", "读取", "查看", "检查", "看看",
]

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_CODE_RE = re.compile(r"```|`|def |class |function |import |from ", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
            0x3400 <= cp <= 0x4DBF or   # CJK Unified Ideographs Extension A
            0x3000 <= cp <= 0x303F or   # CJK Symbols and Punctuation
            0x3040 <= cp <= 0x309F or   # Hiragana
            0x30A0 <= cp <= 0x30FF):    # Katakana
            return True
    return False


def _calculate_complexity_score(text: str) -> int:
    """Calculate task complexity score based on signal presence.

    Positive score = complex task (use primary model).
    Negative or zero score = simple task (use cheap model).

    Signals:
    - Complex keywords (EN word-match / ZH substring): +10 points each
    - Simple keywords (EN word-match / ZH substring): -5 points each
    - Code patterns: +15 points
    - URLs: +10 points
    - Very long text (>500 chars): +10 points
    """
    score = 0
    lowered = text.lower()

    # --- English keyword matching (whitespace-tokenized) ---
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}

    complex_hits = words & _COMPLEX_KEYWORDS_EN
    score += len(complex_hits) * 10

    simple_hits = words & _SIMPLE_KEYWORDS_EN
    score -= len(simple_hits) * 5

    # --- Chinese keyword matching (substring search) ---
    # Only run when text contains CJK characters to avoid false positives
    if _has_cjk(text):
        for kw in _COMPLEX_KEYWORDS_ZH:
            if kw in text:
                score += 10
        for kw in _SIMPLE_KEYWORDS_ZH:
            if kw in text:
                score -= 5

    # --- Structural signals ---
    if _CODE_RE.search(text):
        score += 15

    if _URL_RE.search(text):
        score += 10

    if len(text) > 500:
        score += 10

    return score


def choose_cheap_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple.

    Uses a complexity scoring system instead of simple length thresholds.
    Score > 0 = complex task (primary model)
    Score <= 0 = simple task (cheap model)

    Conservative by design: ambiguous cases default to primary model.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    cheap_model = cfg.get("cheap_model") or {}
    if not isinstance(cheap_model, dict):
        return None
    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    text = (user_message or "").strip()
    if not text:
        return None

    # Calculate complexity score
    score = _calculate_complexity_score(text)

    # Score > 0 means complex task -> use primary model
    if score > 0:
        return None

    # Score <= 0 means simple task -> use cheap model
    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
    return route


def resolve_turn_route(user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Returns a dict with model/runtime/signature/label fields.
    """
    route = choose_cheap_model_route(user_message, routing_config)
    if not route:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
                "command": primary.get("command"),
                "args": list(primary.get("args") or []),
                "credential_pool": primary.get("credential_pool"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
                primary.get("command"),
                tuple(primary.get("args") or ()),
            ),
        }

    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = resolve_runtime_provider(
            requested=route.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=route.get("base_url"),
        )
    except Exception:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
                "command": primary.get("command"),
                "args": list(primary.get("args") or []),
                "credential_pool": primary.get("credential_pool"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
                primary.get("command"),
                tuple(primary.get("args") or ()),
            ),
        }

    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
            "credential_pool": runtime.get("credential_pool"),
        },
        "label": f"smart route -> {route.get('model')} ({runtime.get('provider')})",
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }
