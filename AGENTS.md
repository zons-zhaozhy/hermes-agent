# Hermes Agent - Development Guide (精简版)

Instructions for AI coding assistants and developers working on the hermes-agent codebase.
**Never give up on the right solution.**

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across a CLI, a messaging gateway, a TUI, and an Electron desktop app. It learns across sessions, delegates to subagents, runs scheduled jobs, and drives a real terminal and browser. It is extended primarily through plugins and skills, not by growing the core.

Two load-bearing design properties:
- **Per-conversation prompt caching is sacred**. Anything that mutates past context invalidates the cached prefix.
- **The core is a narrow waist; capability lives at the edges**. Every model tool is sent on every API call — but peripheral tools (browser, vision, TTS, smart-home, kanban) are **deferrable** behind `tool_search` bridge tools, reducing schema overhead ~50% (see `_TOOL_SEARCH_PROTECTED_TOOLS` in `toolsets.py`).

## Important Policies

- **No `HERMES_*` env vars for non-secret config** — `.env` is for secrets only. Behavioral settings go in `config.yaml`.
- **No new core tool when terminal+file already do the job, or when a skill/MCP server would.**
- **No outbound telemetry without opt-in gating.**
- **Behavior contracts over snapshots** — tests assert invariants, not enumerations.
- **Contributor credit preserved** — cherry-pick (rebase-merge) to retain authorship.
- **Fix the bug class, not just the site.**

## Contribution Rubric — What We Want / What We Don't

### What we want
- Fix real bugs, well — reproduce, point to the exact line, fix the whole bug class.
- Expand reach at the edges — new platforms, channels, providers, models, desktop/TUI features.
- Refactor god-files into clean modules — extracting 1000+ line clumps from cli.py/run_agent.py.
- Keep the core narrow — extend existing code → CLI+skill → service-gated tool → plugin → MCP server → new core tool.
- Extend, don't duplicate — before adding a module, check existing infrastructure.
- E2E validation over green unit mocks — exercise real paths with real imports.
- Cache-, alternation-, and invariant-safe.

### What we don't want
- Speculative infrastructure — hooks/callbacks with no concrete consumer.
- Lazy-reading escape hatches — no offset/limit pagination on instructional tools.
- "Fixes" that destroy the feature they secure — read the original commit's intent.
- Change-detector tests — code freeze snapshots that break on every model release.
- Plugins that touch core files — live in their own directory.

### The Footprint Ladder (new capability decision)
1. Extend existing code — zero new surface
2. CLI command + skill — zero model-tool footprint
3. Service-gated tool (check_fn) — only appears when configured
4. Plugin — lives in `~/.hermes/plugins/`
5. MCP server (in catalog) — zero permanent core-schema footprint
6. New core tool — only when fundamental, broadly useful, unreachable via terminal+file or MCP

When 3+ PRs integrate the same category, design an ABC + orchestrator instead of merging one at a time.

## Development Environment

```bash
source .venv/bin/activate   # or: source venv/bin/activate
scripts/run_tests.sh        # probes .venv → venv → ~/.hermes/hermes-agent/venv
```

## Testing

### Running Tests
The **only** supported invocation is `scripts/run_tests.sh`:
```bash
cd hermes-agent
./scripts/run_tests.sh             # hermetic suite
./scripts/run_tests.sh tests/...   # one file
```

### Why the wrapper
The wrapper closes 5 sources of local-vs-CI drift: provider API keys, HOME/~/.hermes, timezone (UTC), locale (C.UTF-8), xdist workers (-n auto with subprocess isolation).

### Without the wrapper (IDE debugging only)
```bash
source .venv/bin/activate
python -m pytest tests/ -q
python -m pytest tests/agent/test_foo.py -q --no-isolate  # bypass isolation for speed
```

### Migration Testing
- Full test suite before pushing.
- Run diff tests against migration scripts before writing them.
- Validate in CI (testcontainers) before any production apply.
- Mock the production DB with test data, not schema stubs.

### Don't write change-detector tests
A test is a change-detector if it fails when data expected to change gets updated — model catalogs, config version numbers, enumeration counts. Do write: behavior assertions (does the plumbing work?), invariants (every model has a context length), relationship checks (no plan-only model leaks into legacy lists). Reviewers: reject new change-detector tests.

## Profiles: Multi-Instance Support

- Each profile is an **independent island** — config/skills/plugins/memories/cron/state are fully isolated.
- Profile directory: `~/.hermes/profiles/<name>/` — clone of `~/.hermes/`.
- CLI: `hermes --profile <name> ...` or `HERMES_PROFILE=<name>` env var.
- Gateway profiles are managed via `config.profile` or `hermes.profile` settings per platform.
- Desktop profiles via `--profile` or profile-selector in GUI.
- CWD-based detection: when an `AGENTS.md` contains a `profile:` key, the agent can auto-switch.
- Profiles are not a security boundary — same user, same file system.

## Project Structure (abridged)

```
hermes-agent/
├── run_agent.py          # AIAgent class (~12K LOC)
├── model_tools.py        # Tool orchestration
├── toolsets.py           # Toolset definitions, _HERMES_CORE_TOOLS
├── cli.py                # HermesCLI class (~11K LOC)
├── agent/                # Agent internals (providers, memory, caching, compression)
├── hermes_cli/           # CLI subcommands, setup wizard, plugins loader, skin engine
├── tools/                # Tool implementations (auto-discovered)
│   └── environments/     # Terminal backends (local, docker, ssh, modal, etc.)
├── gateway/              # Messaging gateway
│   └── platforms/        # Adapters (telegram, discord, slack, etc.)
├── apps/                 # Desktop app (Electron), CLI TUI
├── plugins/              # Plugin type definitions
├── tests/                # Test suite
└── scripts/              # Run, build, install scripts
```

## Key Architecture Notes

- **AIAgent (run_agent.py ~12K LOC)**: Core loop — tool dispatch, skill loading, memory, session tracking, context management, compression, error handling, provider abstraction.
- **cli.py (~11K LOC)**: CLI orchestrator — input processing, rendering, multi-line editing, conversation history, tmux, streaming, file upload.
- **Context compression** (`agent/context_compressor.py`): Summarizes middle turns while protecting head (system prompt) and tail (recent messages). Default threshold: 66% of context window. Tail-zone tool outputs are age-decayed (Pass 4: age 0 full → age 3+ truncated to 800 chars). After 2 ineffective compressions (<10% savings), compression is skipped (anti-thrashing). Escape hatch: 2x threshold triggers forced compression.
- **prompt_builder.py**: Injects AGENTS.md/CLAUDE.md/.cursorrules from workdir as context. Skills index is ranked by `state.db` historical `skill_view` frequency (top-5 get ★ marker). Terminal state snapshot (`~/.hermes/.terminal_state.json`) injected into environment hints for cross-session continuity.
- **toolsets.py**: Defines tool groups. `_HERMES_CORE_TOOLS` is the always-loaded set (48 tools). `_TOOL_SEARCH_PROTECTED_TOOLS` is the strict subset (18 tools) that `tool_search` will never defer — peripheral tools (browser, vision, TTS, etc.) become deferrable behind bridge tools.
- **model_tools.py**: `discover_builtin_tools()`, `handle_function_call()` — tool schema generation + dispatch.
- **gateway/**: Each platform adapter inherits `BasePlatformAdapter`. Extracted media via `extract_media()` returns `(path, is_voice)` tuples. Background tasks use `_run_background_task()`.

## Known Pitfalls

- **MCP tool deep trees**: `read_file` + MCP schemas dominate prompt tokens. Use `enabled_toolsets` to limit. `tool_search` auto-defers MCP + peripheral tools when schema overhead exceeds 10% of context (see `_TOOL_SEARCH_PROTECTED_TOOLS`).
- **macOS `/private/var`**: macOS `/var` is a symlink to `/private/var` — path comparisons that don't resolve symlinks will fail.
- **shoelace component a11y**: Always use `role="img"` + `aria-label="..."` on `sl-icon` for accessibility.
- **platform-only file**: Writing platform-only-changes to files that are imported by tests for other platforms = import time explosion.
- **Electron `git 2.39+`**: Uses `safe.directory` — without it, Desktop auto-updater crashes.
- **Windows PS5.1 UTF-8**: Without BOM, PS5.1 mangles UTF-8 in Write-Host and here-strings.
- **Windows no-console**: `prompt_toolkit.create_output()` crashes in no-console Electron subprocess — needs try/except.
- **Bot tokens in git history**: Don't commit, don't read, don't print.
