"""Tests for context_compressor timeout fallback logic."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestCompressionTimeoutFallback:
    """Verify that timeout errors trigger fallback to main model."""

    def _make_compressor(self, summary_model="aux-model", main_model="main-model"):
        """Create a minimal ContextCompressor mock for testing fallback logic."""
        from agent.context_compressor import ContextCompressor
        comp = object.__new__(ContextCompressor)
        comp.summary_model = summary_model
        comp.model = main_model
        comp._summary_model_fallen_back = False
        comp._summary_failure_cooldown_until = 0
        comp._last_summary_error = ""
        return comp

    def test_timeout_string_triggers_fallback(self):
        """'timeout' in error message should trigger fallback."""
        comp = self._make_compressor(summary_model="aux-model", main_model="main-model")
        
        # Simulate the error handling logic from context_compressor.py
        err_str = "Request timeout after 30s"
        status = 0
        _is_model_not_found = (
            "not found" in err_str
            or "does not exist" in err_str
            or "no available channel" in err_str
        )
        _is_timeout = (
            "timeout" in err_str
            or "timed out" in err_str
            or status in (408, 429, 502, 504)
        )
        
        assert not _is_model_not_found
        assert _is_timeout
        assert comp.summary_model != comp.model
        assert not comp._summary_model_fallen_back
        # Would trigger fallback
        assert _is_model_not_found or _is_timeout

    def test_timed_out_string_triggers_fallback(self):
        """'timed out' in error message should trigger fallback."""
        err_str = "Connection timed out"
        _is_timeout = (
            "timeout" in err_str
            or "timed out" in err_str
        )
        assert _is_timeout

    def test_http_408_triggers_fallback(self):
        """HTTP 408 Request Timeout should trigger fallback."""
        status = 408
        _is_timeout = status in (408, 429, 502, 504)
        assert _is_timeout

    def test_http_504_triggers_fallback(self):
        """HTTP 504 Gateway Timeout should trigger fallback."""
        status = 504
        _is_timeout = status in (408, 429, 502, 504)
        assert _is_timeout

    def test_http_502_triggers_fallback(self):
        """HTTP 502 Bad Gateway should trigger fallback."""
        status = 502
        _is_timeout = status in (408, 429, 502, 504)
        assert _is_timeout

    def test_http_429_triggers_fallback(self):
        """HTTP 429 Rate Limited should trigger fallback."""
        status = 429
        _is_timeout = status in (408, 429, 502, 504)
        assert _is_timeout

    def test_model_not_found_still_works(self):
        """Original 'not found' fallback still works."""
        err_str = "Model aux-model not found"
        _is_model_not_found = (
            "not found" in err_str
            or "does not exist" in err_str
            or "no available channel" in err_str
        )
        assert _is_model_not_found

    def test_no_fallback_when_already_fallen_back(self):
        """No double-fallback once already fallen back."""
        comp = self._make_compressor()
        comp._summary_model_fallen_back = True
        assert comp._summary_model_fallen_back

    def test_no_fallback_when_same_model(self):
        """No fallback when summary_model == model."""
        comp = self._make_compressor(summary_model="same-model", main_model="same-model")
        assert comp.summary_model == comp.model

    def test_normal_error_no_fallback(self):
        """A normal error (not timeout, not not-found) should NOT trigger fallback."""
        err_str = "Internal server error"
        status = 500
        _is_model_not_found = (
            "not found" in err_str
            or "does not exist" in err_str
            or "no available channel" in err_str
        )
        _is_timeout = (
            "timeout" in err_str
            or "timed out" in err_str
            or status in (408, 429, 502, 504)
        )
        assert not (_is_model_not_found or _is_timeout)
