"""Tests for redact_sensitive_text code_file parameter."""

import pytest
from agent.redact import redact_sensitive_text, _mask_token


class TestRedactCodeFileFlag:
    """Verify code_file=True skips ENV/JSON patterns but keeps dangerous ones."""

    # -- ENV patterns --

    def test_env_assignment_redacted_by_default(self):
        """Normal mode: OPENAI_API_KEY=sk-abc... should be redacted."""
        text = "OPENAI_API_KEY=sk-abcdef1234567890"
        result = redact_sensitive_text(text)
        assert "sk-abcdef1234567890" not in result
        assert "***" in result

    def test_env_assignment_preserved_in_code_file(self):
        """code_file=True: ENV assignments like _TOKENS=... are left alone."""
        # A legitimate code pattern: TOKEN_COUNT = 42
        text = "MAX_TOKENS=100"
        result = redact_sensitive_text(text, code_file=True)
        assert "MAX_TOKENS=100" == result

    def test_env_secret_name_still_redacted_in_code_file(self):
        """code_file=True: but known prefix patterns still apply."""
        text = "MY_API_KEY=sk-abcdef1234567890abcdef"
        result = redact_sensitive_text(text, code_file=True)
        # The sk- prefix pattern should still catch it
        assert "sk-abcdef1234567890abcdef" not in result

    def test_env_fallback_redacted_in_code_file(self):
        """code_file=True: os.getenv fallback value with known prefix still caught."""
        text = 'token = os.getenv("API_KEY", "sk-proj-abcdef1234567890")'
        result = redact_sensitive_text(text, code_file=True)
        assert "sk-proj-abcdef1234567890" not in result

    # -- JSON field patterns --

    def test_json_field_redacted_by_default(self):
        """Normal mode: "apiKey": "secret123" should be redacted."""
        text = '"apiKey": "my-secret-value-12345"'
        result = redact_sensitive_text(text)
        assert "my-secret-value-12345" not in result

    def test_json_field_preserved_in_code_file(self):
        """code_file=True: JSON-like patterns in code are left alone."""
        text = '"key": "value"'
        result = redact_sensitive_text(text, code_file=True)
        assert '"key": "value"' == result

    def test_json_api_key_name_preserved_in_code_file(self):
        """code_file=True: "apiKey": "..." in code is not redacted."""
        text = '"apiKey": "test-value"'
        result = redact_sensitive_text(text, code_file=True)
        assert '"apiKey": "test-value"' == result

    # -- Patterns that ALWAYS apply, even with code_file=True --

    def test_bearer_token_always_redacted(self):
        """Authorization headers always redacted, even for code files."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test.sig"
        result = redact_sensitive_text(text, code_file=True)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

    def test_private_key_always_redacted(self):
        """Private key blocks always redacted, even for code files."""
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----"
        result = redact_sensitive_text(text, code_file=True)
        assert "[REDACTED PRIVATE KEY]" in result
        assert "MIIEow" not in result

    def test_db_connection_string_always_redacted(self):
        """DB connstring passwords always redacted, even for code files."""
        text = "postgresql://admin:secretpass@db.example.com:5432/mydb"
        result = redact_sensitive_text(text, code_file=True)
        assert "secretpass" not in result
        assert "***" in result

    def test_known_prefix_always_redacted(self):
        """Known prefixes (sk-, ghp_) always redacted, even for code files."""
        text = "key = sk-abcdef1234567890abcdef1234567890"
        result = redact_sensitive_text(text, code_file=True)
        assert "sk-abcdef1234567890abcdef1234567890" not in result

    def test_ghp_prefix_always_redacted(self):
        text = "token = ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh"
        result = redact_sensitive_text(text, code_file=True)
        assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh" not in result

    # -- Edge cases --

    def test_none_input(self):
        result = redact_sensitive_text(None)
        assert result is None

    def test_empty_string(self):
        result = redact_sensitive_text("")
        assert result == ""

    def test_code_file_false_explicit(self):
        """Explicitly passing code_file=False should behave like default."""
        text = "OPENAI_API_KEY=sk-abcdef1234567890abcdef"
        result = redact_sensitive_text(text, code_file=False)
        assert "sk-abcdef1234567890abcdef" not in result

    def test_realistic_python_code(self):
        """Simulated Python source code: ENV-like patterns preserved, secrets caught."""
        code = """# Config
MAX_TOKENS = 4096
DEFAULT_MODEL = "gpt-4"

# This is a real secret that should still be caught
api_key = "sk-proj-abcdefghijklmnopqrstuv"

# DB connection
db_url = "postgresql://user:password123@localhost:5432/testdb"
"""
        result = redact_sensitive_text(code, code_file=True)
        # ENV-like assignments preserved
        assert "MAX_TOKENS = 4096" in result
        assert '"gpt-4"' in result
        # But real secrets still caught
        assert "sk-proj-abcdefghijklmnopqrstuv" not in result
        assert "password123" not in result
