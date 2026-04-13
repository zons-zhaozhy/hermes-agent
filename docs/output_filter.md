# Output Filter for AI Context Optimization

Inspired by RTK (Rust Token Killer), this filter applies 4-layer purification to tool output before it enters the model's context window, reducing token waste by 50-90% on noisy output.

## Overview

The filter intercepts output from three key tools:
- **terminal_tool.py**: Command output (cargo test, git status, npm install, etc.)
- **code_execution_tool.py**: Python script stdout/stderr
- **browser_tool.py**: Page snapshots and vision analysis

It applies four purification layers:
1. **Smart Filter** — strip noise (progress bars, warnings, blank lines, comments)
2. **Group Aggregate** — merge similar lines by category
3. **Smart Truncate** — keep valuable head/tail, drop repetitive middle
4. **Dedup Merge** — collapse consecutive identical lines with count

## Architecture

The filter is integrated **inside each tool handler** (after `strip_ansi` + `redact`, before JSON return). This follows Hermes's existing pattern — `redact.py` is also called inside individual tool handlers.

Key integration points:
- `terminal_tool.py` L1550: `output = filter_terminal_output(output, command=command)`
- `code_execution_tool.py` L863: `stdout_text = filter_code_execution_output(stdout_text, script_info="execute_code")`
- `browser_tool.py` L1365: `snapshot_text = filter_browser_output(snapshot_text, browser_action="snapshot")`
- `browser_tool.py` L1940: `analysis = filter_browser_output(analysis, browser_action="vision")`

## Configuration

### config.yaml
```yaml
output_filter:
  enabled: true                    # Global enable/disable
  level: "moderate"                # none, light, moderate, aggressive
  max_output_chars: 30000          # Post-filter cap (down from 50000 pre-filter)
  tools:                           # Per-tool enable/disable
    terminal: true
    code_execution: true
    browser: true
    generic: true
```

### Environment Variables
```bash
export HERMES_OUTPUT_FILTER=true          # Enable filter
export HERMES_OUTPUT_FILTER_LEVEL=aggressive  # Override level
```

### Filter Levels
- **none**: Passthrough (no filtering)
- **light**: ANSI already stripped + blank line collapse + progress bar removal
- **moderate** (default): light + dedup + group aggregate + smart truncate
- **aggressive**: moderate + strip comments + strip warnings + heavier truncation

## Token Savings Examples

### Terminal Output (cargo test)
```
Original: 1663 chars
Filtered: 157 chars
Savings: 90.6%
```

Before:
```
Compiling myproject v0.1.0
[========================================] 100%
test test_001 ... ok
test test_002 ... ok
... (78 more test lines) ...
test test_080 ... ok
test test_edge ... FAILED
warning: unused import: `std::collections::HashMap`
test result: FAILED. 80 passed; 1 failed;
```

After:
```
Compiling myproject v0.1.0

warning: unused import:  (×1)

test result: FAILED. 80 passed; 1 failed;
[Test results: 80 passed, 1 FAILED]
test test_edge ... FAILED
```

### Code Execution Output
```
Original: 4440 chars (100 warnings + traceback)
Filtered: ~1500 chars (dedup + truncate)
Savings: ~66%
```

### Browser Output
- Clean text (vision analysis): Passes through unchanged (0% savings)
- Noisy snapshots: Blank line collapse, dedup where applicable

## Implementation Details

### File Structure
```
tools/output_filter.py              # Main filter implementation
tests/tools/test_output_filter.py   # Unit tests (45 tests)
tests/tools/test_output_filter_integration.py  # Integration tests
```

### Key Functions
- `filter_generic_output()`: Main entry point for all tool types
- `filter_terminal_output()`: Terminal-specific wrapper
- `filter_code_execution_output()`: Code execution wrapper
- `filter_browser_output()`: Browser tool wrapper

### Filter Layers
1. **smart_filter()**: Removes progress bars, spinners, excessive blanks
2. **group_aggregate()**: Aggregates test results, download progress
3. **dedup_merge()**: Collapses consecutive identical lines
4. **smart_truncate()**: Keeps head/tail, summarizes middle

## Performance Impact

- **CPU**: Minimal (regex matching, line processing)
- **Memory**: Processes output in-place
- **Latency**: <1ms for typical output, <10ms for 10k+ lines
- **Logging**: Logs compression stats when savings > 0%

## Testing

Run unit tests:
```bash
cd ~/.hermes/hermes-agent
python -m pytest tests/tools/test_output_filter.py -v
```

Run integration verification:
```bash
cd ~/.hermes/hermes-agent
python tests/tools/test_output_filter_integration.py
```

## Comparison with RTK

| Feature | RTK (Rust Token Killer) | Hermes Output Filter |
|---------|-------------------------|----------------------|
| **Architecture** | External CLI proxy | Internal tool integration |
| **Installation** | Separate binary | Built-in |
| **Configuration** | Command-line flags | config.yaml + env vars |
| **Integration** | Manual hook setup | Automatic for all tools |
| **Token Savings** | 80-96% | 50-90% (similar) |
| **Dependencies** | Rust runtime | Pure Python (no deps) |

## Best Practices

1. **Start with "moderate" level**: Balanced savings without losing important info
2. **Monitor logs**: Filter logs compression stats to identify high-savings commands
3. **Use "aggressive" for CI/CD**: When you only care about pass/fail, not details
4. **Disable per-tool**: If a specific tool's output needs full fidelity
5. **Test before deployment**: Verify critical commands still show necessary info

## Troubleshooting

**Problem**: Filter not working
**Solution**: Check `HERMES_OUTPUT_FILTER` env var and config.yaml

**Problem**: Too much information lost
**Solution**: Reduce level to "light" or disable for specific tool

**Problem**: Performance issues with huge output
**Solution**: Reduce `max_output_chars` or disable filter

**Problem**: JSON structure broken
**Solution**: Filter runs after redact, before JSON serialization - structure preserved

## Future Enhancements

1. **Command-aware filtering**: `git diff` (preserve all) vs `git status` (compress)
2. **Output type detection**: Auto-detect test output, build logs, etc.
3. **Statistical reporting**: Dashboard of token savings per tool/command
4. **Dynamic level adjustment**: Based on conversation length/remaining context
5. **Plugin system**: Custom filters for specific workflows

## Credits

Inspired by [RTK (Rust Token Killer)](https://github.com/rtk-ai/rtk), an open-source CLI tool that reduces AI programming token waste by 80-96%.

## License

Part of Hermes Agent. See main project license.
