"""Security primitives for the Feishu webhook boundary.

The adapter keeps request routing; these pure, bounded helpers keep parser,
limiter, and anomaly-table policy out of the already-large adapter facade.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

_FEISHU_WEBHOOK_RATE_WINDOW_SECONDS = 60
_FEISHU_WEBHOOK_RATE_LIMIT_MAX = 120
_FEISHU_WEBHOOK_PREAUTH_RATE_LIMIT_MAX = 600
_FEISHU_WEBHOOK_RATE_MAX_KEYS = 4096
_FEISHU_WEBHOOK_MAX_JSON_DEPTH = 64
_FEISHU_WEBHOOK_ANOMALY_THRESHOLD = 25
_FEISHU_WEBHOOK_ANOMALY_TTL_SECONDS = 6 * 60 * 60
_FEISHU_WEBHOOK_ANOMALY_MAX_KEYS = 4096
_FEISHU_WEBHOOK_ANOMALY_SWEEP_INTERVAL_SECONDS = 60


def json_containers_within_depth(value: Any, max_depth: int) -> bool:
    """Return whether JSON containers stay below ``max_depth`` without recursion."""

    stack: list[tuple[Iterator[Any], int]] = [(iter((value,)), 0)]
    while stack:
        values, parent_depth = stack[-1]
        try:
            item = next(values)
        except StopIteration:
            stack.pop()
            continue
        if not isinstance(item, (dict, list)):
            continue
        depth = parent_depth + 1
        if depth > max_depth:
            return False
        children = item.values() if isinstance(item, dict) else item
        stack.append((iter(children), depth))
    return True


def parse_webhook_envelope(
    body_bytes: bytes, *, max_depth: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode one object envelope and its mapping header, or raise ``ValueError``."""

    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid webhook JSON") from exc
    if not isinstance(payload, dict) or not json_containers_within_depth(payload, max_depth):
        raise ValueError("invalid webhook envelope")
    header = payload.get("header")
    if header is None:
        return payload, {}
    if not isinstance(header, dict):
        raise ValueError("invalid webhook header")
    return payload, header


def verification_token_matches(incoming_token: Any, expected_token: str) -> bool:
    """Compare remote and configured tokens without accepting invalid Unicode."""

    try:
        return hmac.compare_digest(
            str(incoming_token or "").encode("utf-8"),
            expected_token.encode("utf-8"),
        )
    except UnicodeEncodeError:
        return False


def check_rate_limit(
    counts: dict[str, tuple[int, float]],
    rate_key: str,
    *,
    limit: int,
    window_seconds: float,
    max_keys: int,
) -> bool:
    """Update a capped fixed-window limiter; deny untracked keys when it is full."""

    now = time.time()
    entry = counts.get(rate_key)
    if entry is not None:
        count, window_start = entry
        if now - window_start < window_seconds:
            if count >= limit:
                return False
            counts[rate_key] = (count + 1, window_start)
            return True
    if len(counts) >= max_keys:
        for key in [
            key
            for key, (_count, started_at) in counts.items()
            if now - started_at >= window_seconds
        ]:
            del counts[key]
        if rate_key not in counts and len(counts) >= max_keys:
            logger.warning(
                "Feishu webhook rate-limit table at capacity (%d keys); denying %r",
                max_keys,
                rate_key,
            )
            return False
    counts[rate_key] = (1, now)
    return True


def record_webhook_anomaly(
    counts: dict[str, tuple[int, str, float]],
    remote_ip: str,
    status: str,
    *,
    next_sweep_at: float,
) -> float:
    """Record one rejected source in a bounded table and return the next sweep time."""

    now = time.time()
    entry = counts.get(remote_ip)
    if entry is not None:
        count, _last_status, first_seen = entry
        if now - first_seen < _FEISHU_WEBHOOK_ANOMALY_TTL_SECONDS:
            count += 1
            if count % _FEISHU_WEBHOOK_ANOMALY_THRESHOLD == 0:
                logger.warning(
                    "Feishu webhook anomaly: %d rejected requests (%s) from %s over %.0fs",
                    count,
                    status,
                    remote_ip,
                    now - first_seen,
                )
            counts[remote_ip] = (count, status, first_seen)
            return next_sweep_at
        del counts[remote_ip]

    if len(counts) >= _FEISHU_WEBHOOK_ANOMALY_MAX_KEYS:
        if now < next_sweep_at:
            return next_sweep_at
        next_sweep_at = now + _FEISHU_WEBHOOK_ANOMALY_SWEEP_INTERVAL_SECONDS
        for ip in [
            ip
            for ip, (_count, _status, first_seen) in counts.items()
            if now - first_seen >= _FEISHU_WEBHOOK_ANOMALY_TTL_SECONDS
        ]:
            del counts[ip]
        if len(counts) >= _FEISHU_WEBHOOK_ANOMALY_MAX_KEYS:
            return next_sweep_at

    counts[remote_ip] = (1, status, now)
    return next_sweep_at


class FeishuWebhookSecurityMixin:
    """Bounded parser, limiter, and anomaly state for ``FeishuAdapter``."""

    def _init_webhook_security_state(self) -> None:
        self._webhook_rate_counts: dict[str, tuple[int, float]] = {}
        self._webhook_preauth_counts: dict[str, tuple[int, float]] = {}
        self._webhook_anomaly_counts: dict[str, tuple[int, str, float]] = {}
        self._webhook_anomaly_next_sweep_at = 0.0

    def _record_webhook_anomaly(self, remote_ip: str, status: str) -> None:
        """Count rejected responses per source in a bounded anomaly table."""
        self._webhook_anomaly_next_sweep_at = record_webhook_anomaly(
            self._webhook_anomaly_counts,
            remote_ip,
            status,
            next_sweep_at=self._webhook_anomaly_next_sweep_at,
        )

    def _clear_webhook_anomaly(self, remote_ip: str) -> None:
        """Reset the anomaly counter for remote_ip after a successful request."""
        self._webhook_anomaly_counts.pop(remote_ip, None)

    def _parse_webhook_envelope(self, body_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
        return parse_webhook_envelope(
            body_bytes, max_depth=_FEISHU_WEBHOOK_MAX_JSON_DEPTH
        )

    def _verification_token_matches(self, incoming_token: Any, expected_token: str) -> bool:
        return verification_token_matches(incoming_token, expected_token)

    def _check_webhook_rate_limit(self, rate_key: str) -> bool:
        """Charge the authenticated delivery quota after caller authentication."""
        return check_rate_limit(
            self._webhook_rate_counts,
            rate_key,
            limit=_FEISHU_WEBHOOK_RATE_LIMIT_MAX,
            window_seconds=_FEISHU_WEBHOOK_RATE_WINDOW_SECONDS,
            max_keys=_FEISHU_WEBHOOK_RATE_MAX_KEYS,
        )

    def _check_webhook_preauth_limit(self, remote_ip: str) -> bool:
        """Bound parser work separately from authenticated delivery quota."""
        return check_rate_limit(
            self._webhook_preauth_counts,
            remote_ip,
            limit=_FEISHU_WEBHOOK_PREAUTH_RATE_LIMIT_MAX,
            window_seconds=_FEISHU_WEBHOOK_RATE_WINDOW_SECONDS,
            max_keys=_FEISHU_WEBHOOK_RATE_MAX_KEYS,
        )
