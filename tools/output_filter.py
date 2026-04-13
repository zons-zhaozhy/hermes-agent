"""Output filter for AI context optimization across multiple tool types.

Inspired by RTK (Rust Token Killer), applies 4-layer purification to tool
output before it enters the model's context window:

  1. Smart Filter  — strip noise (progress bars, warnings, blank lines, comments)
  2. Group Aggregate — merge similar lines by category
  3. Smart Truncate  — keep valuable head/tail, drop repetitive middle
  4. Dedup Merge     — collapse consecutive identical lines with count

Supported tool types:
  - terminal: command output (cargo test, git status, npm install, etc.)
  - code_execution: Python script stdout/stderr
  - browser: page snapshots, vision analysis, console output
  - web_extract: web page content extraction results
  - search: file search results (ripgrep output)
  - generic: any text output

Command-aware filtering (terminal tool):
  Certain commands produce output where every line matters. The filter
  recognizes these and adjusts behavior:
  - "preserve" : git diff, git show — keep ALL lines (no truncation, no dedup)
  - "compress" : git status, git log, git branch — allow aggressive dedup
  - "default"  : all other commands — normal 4-layer pipeline

Filter levels:
  - "none"      : passthrough (no filtering)
  - "light"     : ANSI already stripped + blank line collapse + progress bar removal
  - "moderate"  : light + dedup + group aggregate + smart truncate (DEFAULT)
  - "aggressive": moderate + strip comments + strip warnings + heavier truncation

Statistics:
  Every filter invocation records original/filtered sizes and savings.
  Access via get_filter_stats() for optimization insights.

Config via config.yaml:
  output_filter:
    enabled: true
    level: "moderate"
    max_output_chars: 30000   # post-filter cap (down from 50000 pre-filter)
    tools:                    # Per-tool enable/disable
      terminal: true
      code_execution: true
      browser: true
      web_extract: true
      search: true
      generic: true
"""

import os
import re
import time
import logging
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_FILTER_ENABLED = os.getenv("HERMES_OUTPUT_FILTER", "").lower() not in (
    "0", "false", "no", "off",
)

# Default filter level — can be overridden via config.yaml
_DEFAULT_LEVEL = "moderate"

# Valid levels in order of aggressiveness
VALID_LEVELS = ("none", "light", "moderate", "aggressive")

# Post-filter output cap (terminal_tool already caps at 50000, this is a
# secondary cap applied AFTER filtering to ensure the model never gets
# excessive output even if the filter doesn't compress enough)
_DEFAULT_MAX_OUTPUT = 30000

# ---------------------------------------------------------------------------
# ILM: Per-tool-type hard limits (Information Lifecycle Management)
# ---------------------------------------------------------------------------
# These enforce context budget discipline at the code level (strongest
# enforcement tier).  Each tool type has a calibrated cap based on how
# much information the model actually needs from that output type.
#
# Rationale:
#   - web_search: titles + snippets, never need full content → tight
#   - search (ripgrep): structured matches, summaries suffice → tight
#   - browser: accessibility tree needs structure → moderate
#   - terminal: varies wildly by command → handled by command_class
#   - code_execution: tracebacks need full context → moderate-high
#   - web_extract: already LLM-summarized → tight
#   - generic: fallback → use global default
#
# Overrides: config.yaml output_filter.tool_max_chars.{tool_type}
# ---------------------------------------------------------------------------
_TOOL_MAX_CHARS: dict[str, int] = {
    "web_search": 8000,       # Search results: titles + snippets only
    "search": 10000,          # Ripgrep matches: structured, dedup-friendly
    "web_extract": 8000,      # Already LLM-summarized page content
    "browser": 15000,         # Accessibility tree: preserve structure
    "code_execution": 20000,  # Tracebacks + stdout need context
    "terminal": 20000,        # Default terminal; command_class may override
    "generic": 15000,         # Fallback
}


# ---------------------------------------------------------------------------
# Command-aware rules — commands that need special filter behavior
# ---------------------------------------------------------------------------

# Commands whose output should be preserved verbatim (every line matters).
# These produce structured diff/patch content where truncation or dedup
# would destroy information the model needs.
_PRESERVE_COMMANDS = re.compile(
    r'^\s*(?:'
    r'git\s+(?:diff|show|format-patch|log\s+--(?:patch|stat|diff))'
    r'|diff\b'
    r'|patch\b'
    r'|sd\b'  # modern diff alternative
    r')',
    re.IGNORECASE,
)

# Commands whose output is safe to compress aggressively.
# These produce lots of repetitive lines where the model only needs
# the summary or unique entries.
_COMPRESS_COMMANDS = re.compile(
    r'^\s*(?:'
    r'git\s+(?:status|log|branch|tag|stash\s+list)'
    r'|ls\b'
    r'|find\b'
    r'|tree\b'
    r'|docker\s+images'
    r'|docker\s+ps'
    r'|kubectl\s+get'
    r'|pip\s+(?:list|freeze|show)'
    r'|npm\s+ls'
    r'|mvn\s+dependency'
    r'|mvn\s+(?:clean|compile|install|package|build)'
    r')',
    re.IGNORECASE,
)


def _classify_command(command: str) -> str:
    """Classify a terminal command for filter behavior.

    Returns:
        "preserve" — keep all lines (no truncation, no dedup, no aggregate)
        "compress" — allow aggressive dedup and truncation
        "default"  — normal 4-layer pipeline
    """
    if not command:
        return "default"
    if _PRESERVE_COMMANDS.match(command):
        return "preserve"
    if _COMPRESS_COMMANDS.match(command):
        return "compress"
    return "default"


# ---------------------------------------------------------------------------
# Filter statistics — per-tool, per-command savings tracking
# ---------------------------------------------------------------------------

# In-memory stats store.  Keyed by tool_type, each entry is a list of
# individual filter call records.  Bounded to prevent unbounded growth
# (FIFO eviction at 500 records per tool type).
_MAX_STATS_PER_TOOL = 500

_filter_stats: dict[str, list[dict]] = defaultdict(list)


def _record_stats(
    tool_type: str,
    context: str,
    original_len: int,
    filtered_len: int,
    level: str,
    command_class: str = "",
) -> None:
    """Record a single filter invocation for statistics tracking."""
    record = {
        "ts": time.time(),
        "context": context[:80] if context else "",
        "original": original_len,
        "filtered": filtered_len,
        "saved_pct": round((1 - filtered_len / original_len) * 100, 1) if original_len > 0 else 0.0,
        "level": level,
        "command_class": command_class,
    }
    stats = _filter_stats[tool_type]
    stats.append(record)
    # FIFO eviction to prevent unbounded memory growth
    if len(stats) > _MAX_STATS_PER_TOOL:
        del stats[: len(stats) - _MAX_STATS_PER_TOOL]


def get_filter_stats(tool_type: str | None = None) -> dict:
    """Retrieve filter statistics, optionally filtered by tool type.

    Returns a dict with:
      - total_calls: total number of filter invocations
      - total_original: sum of original sizes (chars)
      - total_filtered: sum of filtered sizes (chars)
      - overall_saved_pct: overall compression percentage
      - by_tool: per-tool breakdown (dict of tool_type -> stats dict)
      - recent: last 20 records across all tool types

    If tool_type is specified, returns only that tool's stats.
    """
    tools = {tool_type} if tool_type else set(_filter_stats.keys())
    result = {
        "total_calls": 0,
        "total_original": 0,
        "total_filtered": 0,
        "by_tool": {},
    }

    all_records = []
    for t in sorted(tools):
        records = _filter_stats.get(t, [])
        if not records:
            continue
        tool_total_orig = sum(r["original"] for r in records)
        tool_total_filt = sum(r["filtered"] for r in records)
        tool_saved = round((1 - tool_total_filt / tool_total_orig) * 100, 1) if tool_total_orig > 0 else 0.0
        result["by_tool"][t] = {
            "calls": len(records),
            "total_original": tool_total_orig,
            "total_filtered": tool_total_filt,
            "saved_pct": tool_saved,
            "avg_saved_pct": round(sum(r["saved_pct"] for r in records) / len(records), 1),
        }
        result["total_calls"] += len(records)
        result["total_original"] += tool_total_orig
        result["total_filtered"] += tool_total_filt
        all_records.extend(records)

    if result["total_original"] > 0:
        result["overall_saved_pct"] = round(
            (1 - result["total_filtered"] / result["total_original"]) * 100, 1
        )
    else:
        result["overall_saved_pct"] = 0.0

    # Recent records (last 20, newest first)
    all_records.sort(key=lambda r: r["ts"], reverse=True)
    result["recent"] = all_records[:20]

    return result


def reset_filter_stats() -> None:
    """Clear all accumulated filter statistics."""
    _filter_stats.clear()


def _get_filter_config() -> dict:
    """Load filter config from Hermes config.yaml, with env overrides."""
    config = {
        "enabled": _FILTER_ENABLED,
        "level": _DEFAULT_LEVEL,
        "max_output_chars": _DEFAULT_MAX_OUTPUT,
        "tool_max_chars": dict(_TOOL_MAX_CHARS),  # ILM per-tool hard limits
        "tools": {
            "terminal": True,
            "code_execution": True,
            "browser": True,
            "web_extract": True,
            "search": True,
            "generic": True,
        },
    }
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        fc = cfg.get("output_filter", {})
        if isinstance(fc, dict):
            config["enabled"] = fc.get("enabled", config["enabled"])
            config["level"] = fc.get("level", config["level"])
            config["max_output_chars"] = fc.get("max_output_chars", config["max_output_chars"])
            # ILM per-tool max chars override from config
            tool_max_cfg = fc.get("tool_max_chars", {})
            if isinstance(tool_max_cfg, dict):
                for tool_name, limit in tool_max_cfg.items():
                    if isinstance(limit, int) and limit > 0:
                        config["tool_max_chars"][tool_name] = limit
            # Per-tool configuration
            tools_cfg = fc.get("tools", {})
            if isinstance(tools_cfg, dict):
                for tool in config["tools"]:
                    if tool in tools_cfg:
                        config["tools"][tool] = bool(tools_cfg[tool])
    except Exception:
        pass

    # Env override for enabled (HERMES_OUTPUT_FILTER=false disables globally)
    env_enabled = os.getenv("HERMES_OUTPUT_FILTER", "").lower()
    if env_enabled in ("false", "0", "no", "off"):
        config["enabled"] = False

    # Env override for level
    env_level = os.getenv("HERMES_OUTPUT_FILTER_LEVEL", "").lower()
    if env_level in VALID_LEVELS:
        config["level"] = env_level

    return config


# ---------------------------------------------------------------------------
# Layer 1: Smart Filter — strip noise
# ---------------------------------------------------------------------------

# Progress bar patterns: [=   ], [###>    ], 100%, etc.
_PROGRESS_BAR_RE = re.compile(
    r'^.*\[(?:[#*=\s]{3,}|\.+|[=>#\s]{3,})\]\s*\d*%?\s*$',
    re.MULTILINE,
)

# Spinner patterns
_SPINNER_RE = re.compile(r'^[|/\-\x5c][\s]*$', re.MULTILINE)

# Carriage return lines (overwritten terminal output)
_CR_LINE_RE = re.compile(r'^.*\r(?!\n)', re.MULTILINE)

# Excessive blank lines (3+ consecutive → 2)
_MULTI_BLANK_RE = re.compile(r'\n{4,}')

# Common warning patterns that are typically noise in dev output
_NOISE_WARNING_RE = re.compile(
    r'^(?:'
    r'(?:npm|yarn|pnpm)\s+warn\s+(?!deprecated|error|EBAD)'  # npm warnings except critical ones
    r'|warning:\s*(?:unused|implicit|deprecated|overridden|duplicate)'
    r'|(?:note|info):\s*.*(?:recompile|restart|rerun)'
    r'|BUILD SUCCESSFUL\b'  # Gradle noise
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# Comment lines in build output (NOT source code — we don't touch source code)
_BUILD_COMMENT_RE = re.compile(r'^\s*#[^\n]*$', re.MULTILINE)


def smart_filter(output: str, level: str = "moderate") -> str:
    """Layer 1: Remove noise from terminal output.

    Applies: progress bars, spinners, carriage returns, blank line collapse.
    At 'aggressive' level also strips: build comments, noise warnings.
    """
    if not output:
        return output

    # Always apply (even 'light' level):
    # Remove progress bars (terminal animation artifacts)
    result = _PROGRESS_BAR_RE.sub('', output)
    # Remove spinners
    result = _SPINNER_RE.sub('', result)

    if level in ("moderate", "aggressive"):
        # Collapse excessive blank lines to exactly 2 (one visual gap)
        result = _MULTI_BLANK_RE.sub('\n\n', result)

    if level == "aggressive":
        # Strip build-system comments (not source code!)
        result = _BUILD_COMMENT_RE.sub('', result)
        # Strip noise warnings
        result = _NOISE_WARNING_RE.sub('', result)
        # Re-collapse blanks after stripping
        result = _MULTI_BLANK_RE.sub('\n\n', result)

    return result.strip()


# ---------------------------------------------------------------------------
# Layer 2: Group Aggregate — merge similar lines by category
# ---------------------------------------------------------------------------

# Test output patterns: group "test XYZ ... ok/FAILED" lines
_TEST_RESULT_RE = re.compile(
    r'^(test\s+\S+\s+\.\.\.\s+(?:ok|FAILED|ignored|bench))$',
    re.MULTILINE,
)

# Log timestamp prefix pattern
_LOG_TS_RE = re.compile(r'^\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}')

# Maven/Gradle download progress
_DOWNLOAD_RE = re.compile(
    r'^(?:Download|Downloading|Downloading from)[\s:].*?\s+\d+%',
    re.MULTILINE | re.IGNORECASE,
)


def group_aggregate(output: str, level: str = "moderate") -> str:
    """Layer 2: Merge similar output lines into grouped summaries.

    Currently handles:
    - Test result lines (Rust cargo test style)
    - Download progress lines

    When grouped, shows a count + one example line.
    """
    if not output or level == "none":
        return output

    lines = output.split('\n')
    result_lines = []
    test_pass_count = 0
    test_fail_lines = []
    download_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result_lines.append(line)
            continue

        # Aggregate test results
        m = _TEST_RESULT_RE.match(stripped)
        if m:
            if 'ok' in stripped:
                test_pass_count += 1
                continue
            elif 'FAILED' in stripped:
                test_fail_lines.append(stripped)
                continue
            else:
                # ignored/bench — keep
                result_lines.append(line)
                continue

        # Aggregate download progress
        if _DOWNLOAD_RE.match(stripped):
            download_count += 1
            continue

        result_lines.append(line)

    # Emit aggregated test summary
    if test_pass_count > 0 or test_fail_lines:
        summary_parts = []
        if test_pass_count > 0:
            summary_parts.append(f"{test_pass_count} passed")
        if test_fail_lines:
            summary_parts.append(f"{len(test_fail_lines)} FAILED")
        result_lines.append(f"[Test results: {', '.join(summary_parts)}]")
        # Always show failures in detail
        result_lines.extend(test_fail_lines)

    if download_count > 0:
        result_lines.append(f"[{download_count} download progress lines omitted]")

    return '\n'.join(result_lines)


# ---------------------------------------------------------------------------
# Layer 3: Smart Truncate — keep head/tail, summarize middle
# ---------------------------------------------------------------------------

def smart_truncate(output: str, level: str = "moderate") -> str:
    """Layer 3: Intelligently truncate long output.

    Keeps first N lines (where errors/headers appear) and last M lines
    (where final results/summary appear).  Middle is collapsed to a
    line count notice.

    Thresholds:
      moderate:    >500 lines triggers truncation
      aggressive:  >200 lines triggers truncation
    """
    if not output:
        return output

    threshold = 500 if level == "moderate" else (200 if level == "aggressive" else 10000)
    lines = output.split('\n')

    if len(lines) <= threshold:
        return output

    # Keep 40% head (errors, headers, initial context) + 50% tail (results)
    head_count = int(threshold * 0.4)
    tail_count = int(threshold * 0.5)
    omitted = len(lines) - head_count - tail_count

    head = lines[:head_count]
    tail = lines[-tail_count:]

    return (
        '\n'.join(head)
        + f'\n\n... [{omitted} lines omitted out of {len(lines)} total] ...\n\n'
        + '\n'.join(tail)
    )


# ---------------------------------------------------------------------------
# Layer 4: Dedup Merge — collapse consecutive identical lines
# ---------------------------------------------------------------------------

def dedup_merge(output: str, level: str = "moderate") -> str:
    """Layer 4: Collapse consecutive identical lines with occurrence count.

    Example:
      Connection timeout
      Connection timeout
      Connection timeout
      → Connection timeout (×3)
    """
    if not output or level == "none":
        return output

    lines = output.split('\n')
    if len(lines) < 3:
        return output

    result = []
    prev = None
    count = 0

    for line in lines:
        if line == prev:
            count += 1
        else:
            if prev is not None:
                if count > 2:
                    result.append(f"{prev} (×{count})")
                elif count == 2:
                    result.append(prev)
                    result.append(prev)
                else:
                    result.append(prev)
            prev = line
            count = 1

    # Flush last group
    if prev is not None:
        if count > 2:
            result.append(f"{prev} (×{count})")
        elif count == 2:
            result.append(prev)
            result.append(prev)
        else:
            result.append(prev)

    return '\n'.join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_generic_output(
    output: str,
    tool_type: str = "generic",
    context: str = "",
    level: Optional[str] = None,
) -> str:
    """Apply all 4 filter layers to any tool output.

    This is the main entry point for all tool types.

    Args:
        output: Raw output (should already be ANSI-stripped and redacted).
        tool_type: Type of tool ("terminal", "code_execution", "browser",
                   "web_extract", "search", "generic").
        context: Tool-specific context (e.g., command string for terminal).
        level: Override filter level. None = use config default.

    Returns:
        Filtered output string. Original structure preserved, noise removed.
    """
    config = _get_filter_config()

    # Quick exit if disabled globally or for this tool type
    if not config["enabled"]:
        return output
    if not config["tools"].get(tool_type, True):
        return output

    filter_level = level or config["level"]
    if filter_level not in VALID_LEVELS:
        filter_level = _DEFAULT_LEVEL
    if filter_level == "none":
        return output

    if not output or len(output) < 50:
        # Too short to benefit from filtering
        return output

    original_len = len(output)

    # Command-aware classification for terminal tool
    command_class = ""
    if tool_type == "terminal" and context:
        command_class = _classify_command(context)
        if command_class == "preserve":
            # git diff, diff, patch — only strip ANSI noise, preserve all content
            result = smart_filter(output, filter_level)
            # Skip group_aggregate, dedup_merge, smart_truncate
            # Only apply final cap (ILM per-tool limit)
            tool_hard_limit = config["tool_max_chars"].get(tool_type, config["max_output_chars"])
            max_chars = min(config["max_output_chars"], tool_hard_limit)
            if len(result) > max_chars:
                head = int(max_chars * 0.4)
                tail = max_chars - head
                omitted = len(result) - head - tail
                result = (
                    result[:head]
                    + f"\n\n... [{omitted} chars omitted (preserve mode, ILM cap: {tool_type}={max_chars})] ...\n\n"
                    + result[-tail:]
                )
            if result != output:
                saved_pct = round((1 - len(result) / original_len) * 100, 1) if original_len > 0 else 0
                logger.info(
                    "Output filter [%s]: %d → %d chars (%.1f%% saved, level=%s, class=%s, ctx=%s)",
                    tool_type, original_len, len(result), saved_pct, filter_level,
                    command_class, context[:50],
                )
                _record_stats(tool_type, context, original_len, len(result), filter_level, command_class)
            return result

    # Apply 4 layers in sequence
    result = smart_filter(output, filter_level)
    result = group_aggregate(result, filter_level)
    result = dedup_merge(result, filter_level)

    # Compress mode: use a more aggressive truncation threshold
    # (e.g., ls, pip list — the user just needs a summary, not every line)
    if command_class == "compress":
        # For compress commands, always apply aggressive truncation
        # even if under threshold, to provide summary view
        result = smart_truncate(result, "aggressive")
        # Additionally, if output has many lines, apply summary logic
        lines = result.split('\n')
        if len(lines) > 30:
            # Show first 15, last 5, with count summary
            head = lines[:15]
            tail = lines[-5:] if len(lines) > 20 else []
            omitted = len(lines) - len(head) - len(tail)
            result = '\n'.join(head)
            if tail:
                result += f'\n\n... [{omitted} lines omitted] ...\n\n' + '\n'.join(tail)
            result += f'\n\nTotal: {len(lines)} lines'
    else:
        result = smart_truncate(result, filter_level)

    # Tool-specific adjustments
    if tool_type == "browser":
        # Browser snapshots need to preserve accessibility tree structure
        # Don't dedup or aggregate too aggressively
        pass
    elif tool_type == "code_execution":
        # Python script output may have tracebacks - preserve structure
        pass
    elif tool_type == "web_extract":
        # Web content is already processed by LLM, just cap size
        pass
    elif tool_type == "search":
        # Search results are structured JSON — dedup is the main win
        pass

    # Final cap — ensure we never exceed post-filter max
    # ILM enforcement: use per-tool-type hard limit (stronger than global cap)
    tool_hard_limit = config["tool_max_chars"].get(tool_type, config["max_output_chars"])
    max_chars = min(config["max_output_chars"], tool_hard_limit)
    if len(result) > max_chars:
        head = int(max_chars * 0.4)
        tail = max_chars - head
        omitted = len(result) - head - tail
        result = (
            result[:head]
            + f"\n\n... [{omitted} chars omitted after filtering (ILM cap: {tool_type}={max_chars})] ...\n\n"
            + result[-tail:]
        )

    # Log compression stats + record for statistics
    if result != output:
        saved_pct = round((1 - len(result) / original_len) * 100, 1) if original_len > 0 else 0
        logger.info(
            "Output filter [%s]: %d → %d chars (%.1f%% saved, level=%s, ctx=%s)",
            tool_type, original_len, len(result), saved_pct, filter_level,
            context[:50] if context else "<no context>",
        )
        _record_stats(tool_type, context, original_len, len(result), filter_level, command_class)

    return result


def filter_terminal_output(
    output: str,
    command: str = "",
    level: Optional[str] = None,
) -> str:
    """Apply all 4 filter layers to terminal output.

    This is the single entry point called from terminal_tool.py.

    Args:
        output: Raw terminal output (already ANSI-stripped and redacted).
        command: The original command string (for command-aware filtering).
        level: Override filter level. None = use config default.

    Returns:
        Filtered output string. Original structure preserved, noise removed.
    """
    return filter_generic_output(
        output=output,
        tool_type="terminal",
        context=command,
        level=level,
    )


def filter_code_execution_output(
    output: str,
    script_info: str = "",
    level: Optional[str] = None,
) -> str:
    """Apply all 4 filter layers to code execution output.

    This is called from code_execution_tool.py.

    Args:
        output: Python script stdout/stderr (already ANSI-stripped and redacted).
        script_info: Brief description of the script (e.g., "web scraping").
        level: Override filter level. None = use config default.

    Returns:
        Filtered output string.
    """
    return filter_generic_output(
        output=output,
        tool_type="code_execution",
        context=script_info,
        level=level,
    )


def filter_browser_output(
    output: str,
    browser_action: str = "",
    level: Optional[str] = None,
) -> str:
    """Apply all 4 filter layers to browser tool output.

    This is called from browser_tool.py for snapshots, vision analysis, etc.

    Args:
        output: Browser snapshot or analysis text.
        browser_action: Type of browser action ("snapshot", "vision", "console").
        level: Override filter level. None = use config default.

    Returns:
        Filtered output string.
    """
    return filter_generic_output(
        output=output,
        tool_type="browser",
        context=browser_action,
        level=level,
    )


def filter_web_extract_output(
    output: str,
    url: str = "",
    level: Optional[str] = None,
) -> str:
    """Apply output filtering to web_extract results.

    Web content is typically already LLM-summarized, so filtering focuses on:
    - Removing excessive blank lines and boilerplate
    - Capping size (large page extractions can be huge)
    - NOT dedup/aggregating (content structure matters)

    This is called from web_tools.py.

    Args:
        output: Extracted page content (markdown).
        url: Source URL for context.
        level: Override filter level. None = use config default.

    Returns:
        Filtered output string.
    """
    return filter_generic_output(
        output=output,
        tool_type="web_extract",
        context=url[:80] if url else "",
        level=level,
    )


def filter_search_output(
    output: str,
    pattern: str = "",
    level: Optional[str] = None,
) -> str:
    """Apply output filtering to search_files results.

    Search output is structured (ripgrep JSON-like), so filtering focuses on:
    - Dedup of repeated patterns (e.g., same match across many files)
    - Size capping for broad searches with many results

    This is called from file_tools.py.

    Args:
        output: Search results text.
        pattern: Search pattern for context.
        level: Override filter level. None = use config default.

    Returns:
        Filtered output string.
    """
    return filter_generic_output(
        output=output,
        tool_type="search",
        context=pattern[:80] if pattern else "",
        level=level,
    )
