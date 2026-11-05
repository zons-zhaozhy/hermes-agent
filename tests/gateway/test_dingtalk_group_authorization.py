"""DingTalk chat-scoped group authorization contracts."""

import logging
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig


def _adapter(extra=None):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    adapter = object.__new__(DingTalkAdapter)
    adapter.platform = Platform.DINGTALK
    adapter.config = PlatformConfig(enabled=True, extra=extra or {})
    adapter._allowed_users = {str(item).lower() for item in (extra or {}).get("allowed_users", [])}
    adapter._mention_patterns = []
    return adapter


class TestGroupScopedAuthorization:
    def test_allowed_group_authorizes_member_without_widening_dm(self):
        adapter = _adapter({
            "allowed_users": ["owner"],
            "group_allowed_chats": ["group-1"],
        })

        assert adapter._is_user_allowed("guest", "guest-staff", chat_id="group-1", is_group=True)
        assert not adapter._is_user_allowed("guest", "guest-staff", chat_id="group-2", is_group=True)
        assert not adapter._is_user_allowed("guest", "guest-staff", chat_id="group-1", is_group=False)

    def test_group_authorization_does_not_bypass_mention_gate(self):
        adapter = _adapter({
            "allowed_users": ["owner"],
            "group_allowed_chats": ["group-1"],
            "require_mention": True,
        })
        mentioned = SimpleNamespace(is_in_at_list=True)
        unmentioned = SimpleNamespace(is_in_at_list=False)

        assert adapter._should_process_message(mentioned, "hello", True, "group-1")
        assert not adapter._should_process_message(unmentioned, "hello", True, "group-1")


class TestDingTalkGroupConfigChain:
    def test_yaml_bridge_seeds_group_allowed_chats(self, monkeypatch):
        from plugins.platforms.dingtalk.adapter import _apply_yaml_config

        monkeypatch.delenv("DINGTALK_GROUP_ALLOWED_CHATS", raising=False)
        seeded = _apply_yaml_config({}, {"group_allowed_chats": ["group-1", "group-2"]})

        assert seeded == {"group_allowed_chats": ["group-1", "group-2"]}

    def test_registry_declares_dingtalk_group_chat_env(self):
        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
        entry = platform_registry.get("dingtalk")

        assert entry is not None
        assert entry.group_allowed_chats_env == "DINGTALK_GROUP_ALLOWED_CHATS"


class TestAuthorizationRejectionVisibility:
    @pytest.mark.asyncio
    async def test_mentioned_unauthorized_group_sender_logs_info_without_full_ids(self, caplog):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter

        adapter = object.__new__(DingTalkAdapter)
        adapter.platform = Platform.DINGTALK
        adapter.config = PlatformConfig(enabled=True)
        adapter._allowed_users = {"owner"}
        adapter._dedup = SimpleNamespace(is_duplicate=lambda _message_id: False)
        adapter._dingtalk_group_allowed_chats = lambda: set()
        message = SimpleNamespace(
            message_id="message-1",
            conversation_id="group-1",
            conversation_type="2",
            sender_id="secret-sender-id",
            sender_staff_id="secret-staff-id",
            sender_nick="guest",
            is_in_at_list=True,
        )

        with caplog.at_level(logging.INFO, logger="plugins.platforms.dingtalk.adapter"):
            await adapter._on_message(message)

        assert any("non-allowlisted" in record.message for record in caplog.records)
        assert not any("secret-sender-id" in record.message for record in caplog.records)
        assert not any("secret-staff-id" in record.message for record in caplog.records)
