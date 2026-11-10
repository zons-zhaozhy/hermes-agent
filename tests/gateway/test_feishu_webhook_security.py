"""Pre-authentication security contract for the Feishu webhook boundary.

Regression for NousResearch/hermes-agent#29154: unauthenticated traffic must
not consume the authenticated delivery quota or force unbounded parser work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

try:
    import lark_oapi

    _HAS_LARK_OAPI = True
except ImportError:
    _HAS_LARK_OAPI = False


class _FakeRequestContent:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.read_sizes: list[int] = []

    async def readexactly(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if len(self.body) < size:
            raise asyncio.IncompleteReadError(self.body, size)
        return self.body[:size]


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestWebhookSecurity(unittest.TestCase):
    """Tests for webhook signature verification, rate limiting, and body size limits."""

    def _make_adapter(self, encrypt_key: str = "") -> Any:
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        with patch.dict(
            os.environ,
            {
                "FEISHU_APP_ID": "cli",
                "FEISHU_APP_SECRET": "sec",
                "FEISHU_ENCRYPT_KEY": encrypt_key,
            },
            clear=True,
        ):
            return FeishuAdapter(PlatformConfig())

    def test_signature_valid_passes(self) -> None:
        encrypt_key = "test_secret"
        adapter = self._make_adapter(encrypt_key)
        body = b'{"type":"event"}'
        timestamp = "1700000000"
        nonce = "abc123"
        content = f"{timestamp}{nonce}{encrypt_key}" + body.decode("utf-8")
        sig = hashlib.sha256(content.encode("utf-8")).hexdigest()
        headers = {
            "x-lark-request-timestamp": timestamp,
            "x-lark-request-nonce": nonce,
            "x-lark-signature": sig,
        }
        self.assertTrue(adapter._is_webhook_signature_valid(headers, body))

    def test_rate_limit_resets_after_window_expires(self) -> None:
        from plugins.platforms.feishu.adapter_webhook_security import (
            _FEISHU_WEBHOOK_RATE_LIMIT_MAX,
            _FEISHU_WEBHOOK_RATE_WINDOW_SECONDS,
        )

        adapter = self._make_adapter()
        ip = "10.0.0.3"
        for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX):
            adapter._check_webhook_rate_limit(ip)
        self.assertFalse(adapter._check_webhook_rate_limit(ip))
        # Simulate window expiry by backdating the stored entry.
        count, window_start = adapter._webhook_rate_counts[ip]
        adapter._webhook_rate_counts[ip] = (
            count,
            window_start - _FEISHU_WEBHOOK_RATE_WINDOW_SECONDS - 1,
        )
        self.assertTrue(adapter._check_webhook_rate_limit(ip))

    def test_webhook_request_rejects_oversized_chunked_body_while_reading(self) -> None:
        from gateway.config import PlatformConfig
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from plugins.platforms.feishu.adapter import FeishuAdapter, _FEISHU_WEBHOOK_MAX_BODY_BYTES

        with tempfile.TemporaryDirectory() as tmpdir:
            token = set_hermes_home_override(tmpdir)
            try:
                adapter = FeishuAdapter(PlatformConfig())
            finally:
                reset_hermes_home_override(token)
            content = _FakeRequestContent(b"A" * (_FEISHU_WEBHOOK_MAX_BODY_BYTES + 2))
            request = SimpleNamespace(
                remote="127.0.0.1",
                content_length=None,
                headers={},
                content=content,
            )

            response = asyncio.run(adapter._handle_webhook_request(request))

            self.assertEqual(response.status, 413)
            self.assertEqual(content.read_sizes, [_FEISHU_WEBHOOK_MAX_BODY_BYTES + 1])

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_connect_requires_inbound_auth_secret(self) -> None:
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "app_id": "cli_app",
                    "app_secret": "secret_app",
                    "connection_mode": "webhook",
                },
            )
        )
        self.assertFalse(asyncio.run(adapter.connect()))

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_loads_auth_secrets_from_platform_extra(self) -> None:
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "app_id": "cli_app",
                    "app_secret": "secret_app",
                    "connection_mode": "webhook",
                    "verification_token": "token_from_extra",
                    "encrypt_key": "encrypt_from_extra",
                },
            )
        )
        self.assertEqual(adapter._verification_token, "token_from_extra")
        self.assertEqual(adapter._encrypt_key, "encrypt_from_extra")


class TestFeishuWebhookPreAuthSecurity(unittest.TestCase):
    def _request(self, body: bytes, *, remote: str, headers: dict[str, str] | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            remote=remote,
            content_length=len(body),
            headers={"Content-Type": "application/json", **(headers or {})},
            content=_FakeRequestContent(body),
        )

    def _adapter(self, extra: dict[str, str] | None = None):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        return FeishuAdapter(PlatformConfig(extra=extra or {}))

    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_verification_token_does_not_consume_delivery_quota(self) -> None:
        adapter = self._adapter({"verification_token": "expected"})

        def request(token: str) -> SimpleNamespace:
            body = json.dumps({"header": {"token": token, "event_type": "unknown"}}).encode()
            return self._request(body, remote="203.0.113.10")

        with patch(
            "plugins.platforms.feishu.adapter_webhook_security._FEISHU_WEBHOOK_RATE_LIMIT_MAX",
            2,
        ):
            self.assertEqual(asyncio.run(adapter._handle_webhook_request(request("invalid"))).status, 401)
            self.assertEqual(asyncio.run(adapter._handle_webhook_request(request("invalid"))).status, 401)
            response = asyncio.run(adapter._handle_webhook_request(request("expected")))

        self.assertEqual(response.status, 200)
        self.assertEqual([count for count, _started in adapter._webhook_rate_counts.values()], [1])

    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_signature_does_not_consume_delivery_quota(self) -> None:
        encrypt_key = "expected-key"
        adapter = self._adapter({"encrypt_key": encrypt_key})
        body = json.dumps({"header": {"event_type": "unknown"}}).encode()
        headers = {
            "x-lark-request-timestamp": "1700000000",
            "x-lark-request-nonce": "abc123",
        }
        valid_signature = hashlib.sha256(
            f"1700000000abc123{encrypt_key}".encode() + body
        ).hexdigest()

        def request(signature: str) -> SimpleNamespace:
            return self._request(
                body,
                remote="203.0.113.11",
                headers={**headers, "x-lark-signature": signature},
            )

        with patch(
            "plugins.platforms.feishu.adapter_webhook_security._FEISHU_WEBHOOK_RATE_LIMIT_MAX",
            2,
        ):
            self.assertEqual(asyncio.run(adapter._handle_webhook_request(request("invalid"))).status, 401)
            self.assertEqual(asyncio.run(adapter._handle_webhook_request(request("invalid"))).status, 401)
            response = asyncio.run(adapter._handle_webhook_request(request(valid_signature)))

        self.assertEqual(response.status, 200)
        self.assertEqual([count for count, _started in adapter._webhook_rate_counts.values()], [1])

    @patch.dict(os.environ, {}, clear=True)
    def test_deep_json_is_rejected_before_quota_or_dispatch(self) -> None:
        adapter = self._adapter({"verification_token": "expected"})
        body = (
            b'{"header":{"token":"expected","event_type":"unknown"},"nested":'
            + b"[" * 65
            + b"0"
            + b"]" * 65
            + b"}"
        )

        with patch.object(adapter, "_on_message_event") as dispatch:
            response = asyncio.run(
                adapter._handle_webhook_request(
                    self._request(body, remote="203.0.113.12")
                )
            )

        self.assertEqual(response.status, 400)
        self.assertEqual(adapter._webhook_rate_counts, {})
        dispatch.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_unsigned_url_verification_requires_a_configured_token(self) -> None:
        adapter = self._adapter()
        body = json.dumps(
            {
                "type": "url_verification",
                "token": "unconfigured",
                "challenge": "attacker-controlled-challenge",
            }
        ).encode()

        response = asyncio.run(
            adapter._handle_webhook_request(
                self._request(body, remote="203.0.113.13")
            )
        )

        self.assertEqual(response.status, 401)
        self.assertNotIn(b"attacker-controlled-challenge", response.body)
        self.assertEqual(adapter._webhook_rate_counts, {})

    @patch.dict(os.environ, {}, clear=True)
    def test_preauth_limit_is_separate_from_delivery_quota(self) -> None:
        adapter = self._adapter({"verification_token": "expected"})
        body = json.dumps({"header": {"token": "invalid"}}).encode()
        remote = "203.0.113.14"

        with patch(
            "plugins.platforms.feishu.adapter_webhook_security._FEISHU_WEBHOOK_PREAUTH_RATE_LIMIT_MAX",
            2,
            create=True,
        ), patch(
            "plugins.platforms.feishu.adapter_webhook_security.json.loads",
            wraps=json.loads,
        ) as loads:
            statuses = [
                asyncio.run(
                    adapter._handle_webhook_request(self._request(body, remote=remote))
                ).status
                for _ in range(3)
            ]

        self.assertEqual(statuses, [401, 401, 429])
        self.assertEqual(loads.call_count, 2)
        self.assertEqual(adapter._webhook_rate_counts, {})

    @patch.dict(os.environ, {}, clear=True)
    def test_anomaly_tracking_is_bounded(self) -> None:
        adapter = self._adapter()

        with patch(
            "plugins.platforms.feishu.adapter_webhook_security._FEISHU_WEBHOOK_ANOMALY_MAX_KEYS",
            2,
            create=True,
        ):
            for remote in ("203.0.113.20", "203.0.113.21", "203.0.113.22"):
                adapter._record_webhook_anomaly(remote, "401-token")

        self.assertLessEqual(len(adapter._webhook_anomaly_counts), 2)
