#!/usr/bin/env python3
"""
Tests for subagent tier resolution (P2).

Tests tier config loading, credential overlay, and fallback behavior.
"""

import unittest
from unittest.mock import patch, MagicMock


class TestTierResolution(unittest.TestCase):
    """Test the tier-based model resolution."""

    def test_no_tier_returns_base_creds(self):
        """No tier should return base creds unchanged."""
        from tools.subagent_tiers import resolve_tier_credentials

        base = {
            "model": "parent-model",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-xxx",
            "api_mode": "chat_completions",
        }
        result = resolve_tier_credentials(None, base)
        self.assertEqual(result, base)

        result = resolve_tier_credentials("", base)
        self.assertEqual(result, base)

    def test_unknown_tier_returns_base_creds(self):
        """Unknown tier string should fall back to base creds."""
        from tools.subagent_tiers import resolve_tier_credentials

        base = {"model": "parent-model", "provider": None}
        result = resolve_tier_credentials("ultra", base)
        self.assertEqual(result, base)

    @patch("tools.subagent_tiers._load_tier_config")
    def test_configured_tier_overlays_model(self, mock_tiers):
        """A configured tier should override model and provider."""
        from tools.subagent_tiers import resolve_tier_credentials

        mock_tiers.return_value = {
            "cheap": {
                "model": "deepseek/deepseek-chat",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": None,
                "api_mode": None,
            }
        }

        base = {
            "model": "parent-model",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-parent",
            "api_mode": "chat_completions",
        }

        result = resolve_tier_credentials("cheap", base)
        self.assertEqual(result["model"], "deepseek/deepseek-chat")
        # Fields not in tier config should be inherited from base
        self.assertEqual(result["api_key"], "sk-parent")
        self.assertEqual(result["api_mode"], "chat_completions")

    @patch("tools.subagent_tiers._load_tier_config")
    def test_tier_with_all_fields(self, mock_tiers):
        """A tier with all fields set should override everything."""
        from tools.subagent_tiers import resolve_tier_credentials

        mock_tiers.return_value = {
            "capable": {
                "model": "anthropic/claude-opus-4",
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com",
                "api_key": "sk-ant-xxx",
                "api_mode": "anthropic_messages",
            }
        }

        base = {
            "model": "parent-model",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-parent",
            "api_mode": "chat_completions",
        }

        result = resolve_tier_credentials("capable", base)
        self.assertEqual(result["model"], "anthropic/claude-opus-4")
        self.assertEqual(result["provider"], "anthropic")
        self.assertEqual(result["api_key"], "sk-ant-xxx")
        self.assertEqual(result["api_mode"], "anthropic_messages")

    @patch("tools.subagent_tiers._load_tier_config")
    def test_unconfigured_tier_falls_back(self, mock_tiers):
        """Tier requested but not configured should fall back to base."""
        from tools.subagent_tiers import resolve_tier_credentials

        mock_tiers.return_value = {}  # No tiers configured

        base = {"model": "parent-model", "provider": None}
        result = resolve_tier_credentials("cheap", base)
        self.assertEqual(result, base)

    def test_per_task_tier_extraction(self):
        """Per-task tier should override batch default."""
        from tools.subagent_tiers import resolve_per_task_tier

        # Task with its own tier
        task = {"goal": "test", "tier": "capable"}
        self.assertEqual(resolve_per_task_tier(task, "cheap"), "capable")

        # Task without tier falls back to default
        task = {"goal": "test"}
        self.assertEqual(resolve_per_task_tier(task, "cheap"), "cheap")

        # No default
        task = {"goal": "test"}
        self.assertIsNone(resolve_per_task_tier(task, None))

    def test_per_task_tier_invalid(self):
        """Invalid per-task tier should be ignored."""
        from tools.subagent_tiers import resolve_per_task_tier

        task = {"goal": "test", "tier": "ultra"}
        self.assertIsNone(resolve_per_task_tier(task, None))


class TestReviewGating(unittest.TestCase):
    """Test the review gating logic (P1)."""

    def test_should_review_skips_non_completed(self):
        """Non-completed tasks should not be reviewed."""
        from tools.subagent_review import should_review

        self.assertFalse(should_review({"status": "failed", "summary": "err"}))
        self.assertFalse(should_review({"status": "interrupted", "summary": ""}))
        self.assertFalse(should_review({"status": "timeout"}))

    def test_should_review_skips_no_files(self):
        """Tasks without file modifications should not be reviewed."""
        from tools.subagent_review import should_review

        self.assertFalse(
            should_review({"status": "completed", "summary": "research only", "files_written": []})
        )
        self.assertFalse(
            should_review({"status": "completed", "summary": "research only"})
        )

    def test_should_review_accepts_file_modifying_tasks(self):
        """Completed tasks with file writes should be reviewed."""
        from tools.subagent_review import should_review

        self.assertTrue(
            should_review({
                "status": "completed",
                "summary": "implemented feature X",
                "files_written": ["src/feature.py"],
            })
        )


if __name__ == "__main__":
    unittest.main()
