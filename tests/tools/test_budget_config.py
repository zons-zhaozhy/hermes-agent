"""Unit tests for tools/budget_config.py.

Covers default values, resolve_threshold() priority chain
(pinned > tool_overrides > registry > default), immutability,
and the PINNED_THRESHOLDS escape-hatch for read_file.
"""

from unittest.mock import patch


from tools.budget_config import (
    DEFAULT_BUDGET,
    DEFAULT_RESULT_SIZE_CHARS,
    PINNED_THRESHOLDS,
    BudgetConfig,
    budget_for_context_window,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# BudgetConfig defaults
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Custom construction
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# resolve_threshold() priority chain
# ---------------------------------------------------------------------------


class TestResolveThreshold:
    """Priority: pinned > tool_overrides > registry > default."""

    def test_pinned_wins_over_override(self):
        """Even if tool_overrides contains read_file, pinned value wins."""
        cfg = BudgetConfig(tool_overrides={"read_file": 1})
        result = cfg.resolve_threshold("read_file")
        assert result == float("inf")

    def test_tool_override_wins_over_default(self):
        """tool_overrides should be returned before falling back to registry."""
        cfg = BudgetConfig(tool_overrides={"my_tool": 42})
        result = cfg.resolve_threshold("my_tool")
        assert result == 42


    @patch("tools.registry.registry")
    def test_registry_value_capped_at_default(self, mock_registry):
        """A scaled-down budget caps an oversized registry value (#23767).

        web/terminal/x_search register max_result_size_chars=100_000; a small
        model's scaled budget must not be re-inflated by that.
        """
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig(default_result_size=30_000)
        assert cfg.resolve_threshold("web_search") == 30_000


    @patch("tools.registry.registry")
    def test_default_budget_unchanged_for_100k_tool(self, mock_registry):
        """Default budget keeps 100K registry tools at 100K (no behavior change)."""
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig()  # default_result_size == 100_000
        assert cfg.resolve_threshold("web_search") == 100_000


# ---------------------------------------------------------------------------
# budget_for_context_window() — context-aware scaling (#23767)
# ---------------------------------------------------------------------------


class TestBudgetForContextWindow:
    """Scaling the tool-output budget to the active model's context window."""

    def test_none_returns_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # isolate from the operator's real config
        assert budget_for_context_window(None) is DEFAULT_BUDGET

    def test_zero_or_negative_returns_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # isolate from the operator's real config
        assert budget_for_context_window(0) is DEFAULT_BUDGET
        assert budget_for_context_window(-5) is DEFAULT_BUDGET


    def test_scaled_budget_constrains_oversized_result(self):
        """A 279K-char result against a 65K model exceeds the scaled per-result
        threshold, so it will be persisted/truncated rather than sent whole."""
        cfg = budget_for_context_window(65_536)
        huge_len = 279_549
        threshold = cfg.resolve_threshold("mcp_firecrawl_firecrawl_search")
        assert threshold < huge_len
        assert cfg.default_result_size < huge_len


# ---------------------------------------------------------------------------
# MCP-prefix threshold (mcp_result_size)
# ---------------------------------------------------------------------------


class TestMcpPrefixThreshold:
    """mcp_* tools get the tighter 50K default, config-overridable."""

    def test_default_mcp_threshold_is_tighter_than_generic(self):
        from tools.budget_config import DEFAULT_MCP_RESULT_SIZE_CHARS
        assert DEFAULT_MCP_RESULT_SIZE_CHARS < DEFAULT_RESULT_SIZE_CHARS
        assert DEFAULT_BUDGET.resolve_threshold("mcp_composio_search_tools") == DEFAULT_MCP_RESULT_SIZE_CHARS

    def test_non_mcp_tools_keep_generic_default(self):
        assert DEFAULT_BUDGET.resolve_threshold("some_random_tool") == DEFAULT_RESULT_SIZE_CHARS

    def test_pinned_wins_over_mcp_prefix(self):
        with patch.dict(PINNED_THRESHOLDS, {"mcp_pinned_tool": float("inf")}):
            assert DEFAULT_BUDGET.resolve_threshold("mcp_pinned_tool") == float("inf")

    def test_tool_override_wins_over_mcp_prefix(self):
        cfg = BudgetConfig(tool_overrides={"mcp_special": 75_000})
        assert cfg.resolve_threshold("mcp_special") == 75_000

    def test_mcp_threshold_capped_by_scaled_default(self):
        """On a small model the scaled default_result_size caps the MCP value."""
        cfg = BudgetConfig(default_result_size=20_000, mcp_result_size=50_000)
        assert cfg.resolve_threshold("mcp_anything") == 20_000


    def test_config_override_via_hermes_home(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  mcp_result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_composio_multi_execute") == 30_000
        # Generic tools are untouched by the MCP knob.
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS

    def test_config_override_survives_window_scaling(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  mcp_result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(200_000)
        assert cfg.mcp_result_size == 30_000

    def test_malformed_config_falls_back_to_default(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text("tool_budget: not-a-mapping\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.budget_config import DEFAULT_MCP_RESULT_SIZE_CHARS
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_x_y") == DEFAULT_MCP_RESULT_SIZE_CHARS

    def test_scaled_small_window_caps_mcp_threshold(self, tmp_path, monkeypatch):
        """A tiny model's scaled default_result_size caps even the MCP value."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no config.yaml
        cfg = budget_for_context_window(16_384)  # scaled default < 50K
        assert cfg.default_result_size < 50_000
        assert cfg.resolve_threshold("mcp_tool") == cfg.default_result_size


# ---------------------------------------------------------------------------
# tool_budget: config keys (#23767 / #80508) — tighten-only, clamped, warned
# ---------------------------------------------------------------------------


class TestToolBudgetConfigKeys:
    """result_size_chars / turn_budget_chars / tool_overrides tighten the budget.

    Invariants: values equal to the built-in defaults keep the DEFAULT_BUDGET
    singleton (byte-identical history for untouched configs); values outside
    [floor, default] warn once and clamp.
    """

    def test_result_and_turn_keys_tighten(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  result_size_chars: 30000\n  turn_budget_chars: 90000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.default_result_size == 30_000
        assert cfg.turn_budget == 90_000

    def test_defaults_written_explicitly_keep_singleton(self, tmp_path, monkeypatch):
        """The seeded DEFAULT_CONFIG values must not change budget identity."""
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  result_size_chars: 100000\n  turn_budget_chars: 200000\n"
            "  mcp_result_size_chars: 50000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert budget_for_context_window(None) is DEFAULT_BUDGET

    def test_below_floor_clamps_and_warns(self, tmp_path, monkeypatch, caplog):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  result_size_chars: 4000\n  turn_budget_chars: 9000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with caplog.at_level("WARNING", logger="tools.budget_config"):
            cfg = budget_for_context_window(None)
        assert cfg.default_result_size == 8_000   # clamped up to the floor
        assert cfg.turn_budget == 16_000          # clamped up to the floor
        assert "result_size_chars" in caplog.text
        assert "turn_budget_chars" in caplog.text

    def test_above_default_clamps_to_default_and_warns(self, tmp_path, monkeypatch, caplog):
        """Loosening past the default caps back to the default: these keys tighten only,
        and the applied!=configured gap is warned (never silently swallowed)."""
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  result_size_chars: 500000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with caplog.at_level("WARNING", logger="tools.budget_config"):
            cfg = budget_for_context_window(None)
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS  # 期望: 上限即默认值, 钳制回 100000
        assert "result_size_chars" in caplog.text  # 期望: 生效值≠配置值必须告知

    def test_configured_value_also_caps_window_scaling(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(1_000_000)  # scaling alone would give 100K
        assert cfg.default_result_size == 30_000

    def test_tool_overrides_keep_patch_diff_inline(self, tmp_path, monkeypatch):
        """patch/write_file diffs stay fully inline under a tighter generic budget."""
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  result_size_chars: 30000\n"
            "  tool_overrides:\n    patch: 100000\n    write_file: 100000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("patch") == 100_000
        assert cfg.resolve_threshold("write_file") == 100_000
        assert cfg.default_result_size == 30_000  # generic tools stay tightened

    def test_tool_overrides_drop_invalid_entries(self, tmp_path, monkeypatch, caplog):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  tool_overrides:\n    good_tool: 40000\n"
            "    bad_name: not-a-number\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with caplog.at_level("WARNING", logger="tools.budget_config"):
            cfg = budget_for_context_window(None)
        assert cfg.tool_overrides == {"good_tool": 40_000}
        assert "bad_name" in caplog.text

    def test_tightened_generic_caps_mcp_threshold(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_anything") == 30_000  # capped by the tightened generic
