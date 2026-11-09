"""API server metric snapshot publication kept off the owner event loop."""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Optional


API_SERVER_HEARTBEAT_SECONDS = 30.0
API_SERVER_LATENCY_SAMPLE_LIMIT = 512


class APIServerStatusMixin:
    """Metric snapshot helpers used by ``APIServerAdapter``."""

    @staticmethod
    def _metrics_day_key(timestamp: Optional[float] = None) -> str:
        """Return the UTC day bucket for daily API metrics."""
        return time.strftime("%Y-%m-%d", time.gmtime(timestamp or time.time()))

    @staticmethod
    def _iso_timestamp(timestamp: Optional[float]) -> Optional[str]:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, UTC).isoformat()

    def _reset_metrics_if_needed(self) -> None:
        current_day = self._metrics_day_key()
        if self._metrics_day == current_day:
            return
        self._metrics_day = current_day
        self._metrics_requests_today = 0
        self._metrics_messages_today = 0
        self._metrics_tokens_today = 0
        self._metrics_latency_ms.clear()

    @staticmethod
    def _total_tokens_from_usage(usage: Optional[dict[str, Any]]) -> int:
        if not isinstance(usage, dict):
            return 0
        try:
            return max(0, int(usage.get("total_tokens") or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _latency_p95_ms(samples: list[float]) -> Optional[float]:
        if not samples:
            return None
        ordered = sorted(samples)
        index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999999) - 1))
        return round(ordered[index], 2)

    def _api_server_status_payload(self, *, heartbeat_at: Optional[float] = None) -> dict[str, Any]:
        self._reset_metrics_if_needed()
        heartbeat = self._metrics_last_heartbeat_at if heartbeat_at is None else heartbeat_at
        return {
            "host": self._host,
            "port": self._port,
            "active_runs": self._active_structured_run_count() + self._inflight_agent_runs,
            "stored_runs": len(self._run_statuses),
            "last_request_at": self._iso_timestamp(self._metrics_last_request_at),
            "last_heartbeat": self._iso_timestamp(heartbeat),
            "metrics_today": {
                "day": self._metrics_day,
                "requests": self._metrics_requests_today,
                "messages": self._metrics_messages_today,
                "tokens": self._metrics_tokens_today,
                "latency_p95_ms": self._latency_p95_ms(self._metrics_latency_ms),
            },
        }

    def _publish_runtime_status(self) -> None:
        self._metrics_last_heartbeat_at = time.time()
        self._write_runtime_status_safe(
            "api_server_heartbeat",
            platform_state="connected" if self.is_connected else "disconnected",
            platform_metrics=self._api_server_status_payload(),
        )

    def _record_api_metrics(self, usage: Optional[dict[str, Any]], latency_seconds: float) -> None:
        self._reset_metrics_if_needed()
        self._metrics_requests_today += 1
        self._metrics_messages_today += 1
        self._metrics_tokens_today += self._total_tokens_from_usage(usage)
        self._metrics_last_request_at = time.time()
        self._metrics_latency_ms.append(max(0.0, latency_seconds * 1000.0))
        if len(self._metrics_latency_ms) > API_SERVER_LATENCY_SAMPLE_LIMIT:
            del self._metrics_latency_ms[:-API_SERVER_LATENCY_SAMPLE_LIMIT]
        self._publish_runtime_status()

    async def _publish_runtime_status_off_loop(self) -> None:
        await asyncio.to_thread(self._publish_runtime_status)

    async def _record_api_metrics_off_loop(
        self, usage: Optional[dict[str, Any]], latency_seconds: float,
    ) -> None:
        await asyncio.to_thread(self._record_api_metrics, usage, latency_seconds)

    async def _heartbeat_loop(self) -> None:
        while self.is_connected:
            await self._publish_runtime_status_off_loop()
            if not self.is_connected:
                break
            await asyncio.sleep(API_SERVER_HEARTBEAT_SECONDS)
