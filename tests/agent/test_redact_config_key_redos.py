"""Redactor ReDoS resistance for dotted config-key scans (split from test_redact)."""

import pytest

from agent.redact import redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure HERMES_REDACT_SECRETS is not disabled by prior test imports."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


class TestConfigKeyRedosResistance:
    """The dotted-key patterns must not backtrack exponentially (ReDoS).

    Before the possessive-quantifier rewrite, a non-matching run of ~40
    dotted segments took ~30ms and doubled every ~4 segments; 100 segments
    would effectively hang the redactor (it runs on every log line).
    """

    def test_long_dotted_run_completes_fast(self):
        import time

        # 100 dotted segments with no '=' — worst case for the old pattern.
        text = ".".join(["segment"] * 100) + " end"
        t0 = time.perf_counter()
        assert redact_sensitive_text(text) == text
        assert time.perf_counter() - t0 < 2.0

    def test_long_dotted_run_with_keyword_completes_fast(self):
        """Exercise _CFG_DOTTED_RE directly (bypasses the keyword pre-gate).

        The pre-gate skips the regex when no secret keyword is present, so
        test_long_dotted_run_completes_fast only guards the pre-gate.  This
        test includes a keyword but no '=' so the regex runs and must still
        complete quickly thanks to the possessive quantifiers.
        """
        import time

        text = ".".join(["segment"] * 100) + ".token end"
        t0 = time.perf_counter()
        assert redact_sensitive_text(text) == text
        assert time.perf_counter() - t0 < 2.0

    def test_long_dotted_secret_still_redacted(self):
        # Possessive quantifiers must not change matching behavior.
        text = ".".join(["seg"] * 50) + ".password=Sup3rS3cret!"
        result = redact_sensitive_text(text)
        assert "Sup3rS3cret!" not in result
        assert ".password=" in result

    def test_long_opaque_assignment_run_completes_fast(self):
        """Lowercase env scanning stays linear on compaction payload blobs."""
        import time

        # A serialized tool payload can contain a long opaque alphanumeric
        # value followed by '=' without containing a secret-key suffix.  The
        # old unanchored lowercase-env pattern retried its greedy prefix from
        # every byte, making this quadratic while holding the GIL.
        text = "a" * 20_000 + "=value"
        t0 = time.perf_counter()
        assert redact_sensitive_text(text, force=True) == text
        assert time.perf_counter() - t0 < 2.0

    def test_dotted_cfg_scan_stays_linear_with_keyword_elsewhere(self):
        """_CFG_DOTTED_RE must stay linear once the pre-gate passes.

        The ``_CFG_SECRET_WORD_RE`` pre-gate only skips secret-FREE text, so a
        payload that contains a real secret assignment AND a long opaque
        dotted run still reaches the backtrackable ``*`` prefix. Without the
        run-start lookbehind the sub retries that prefix from every byte of
        the run (quadratic while holding the GIL).
        """
        import time

        text = "password=hunter2\n" + "a." * 15_000 + "=value"
        t0 = time.perf_counter()
        result = redact_sensitive_text(text, force=True)
        assert "hunter2" not in result
        assert time.perf_counter() - t0 < 2.0

    def test_yaml_assign_redos_resistance(self):
        """_YAML_ASSIGN_RE must not backtrack excessively on long inputs."""
        import time

        # 100 lines of a long dotted key with a secret keyword but no
        # matching colon-value form — stresses the regex without matching.
        line = "a." * 50 + "token not_an_assignment"
        text = "\n".join([line] * 100)
        t0 = time.perf_counter()
        redact_sensitive_text(text)
        assert time.perf_counter() - t0 < 2.0

    def test_yaml_assign_secret_still_redacted(self):
        # Possessive quantifiers must not change YAML matching behavior.
        text = "spring.datasource.password: hunter2"
        result = redact_sensitive_text(text)
        assert "hunter2" not in result
        assert "password:" in result
