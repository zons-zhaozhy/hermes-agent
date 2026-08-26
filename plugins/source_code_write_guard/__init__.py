"""source-code-write-guard plugin — intercept terminal commands that silently write source files.

Motivation: terminal tool output is summary-only in CLI. Any file-write channel
(cat >, tee, python3 -c, perl -i, sed -i, echo >) silently modifies files with
no visible diff. patch tool returns diff as its result, visible in CLI.

Detection layers (narrow → wide):
  1. Python inline: python[23]? -c / << / /dev/stdin + write operations
  2. Shell redirect: cat > / tee / echo > / printf > targeting source extensions
  3. In-place edit: perl -pi / sed -i / awk -i targeting source extensions

Source file extensions: .py .pyi .ts .tsx .js .mjs .cjs .java .yaml .yml
  .json .toml .md .sh .rs .go .rb .php .css .html .sql .vue .svelte

Exclusions (always allow):
  - Non-source targets: /tmp, /dev/null, node_modules, .git, __pycache__, build, dist
  - Read-only commands: sed -n, grep, git, docker, npm, pip, pytest, find, ls
  - Escape hatch: guard/infra file path keywords

Contract:
  Preconditions: tool_name == "terminal", command is non-empty string
  Postconditions: returns None (allow) or {"action":"block","message":str} (block)
"""

from __future__ import annotations

import logging
import os
import re  # noqa: scanner plugin — regex is the implementation, not an analysis shortcut
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Source file extension whitelist ──────────────────────────────────────

_SOURCE_EXTENSIONS = frozenset({
    ".py", ".pyi", ".pyw",
    ".ts", ".tsx", ".js", ".mjs", ".cjs",
    ".java", ".kt", ".scala",
    ".yaml", ".yml", ".json", ".toml",
    ".md", ".rst", ".txt",
    ".sh", ".bash", ".zsh",
    ".rs", ".go", ".rb", ".php",
    ".css", ".scss", ".html", ".vue", ".svelte",
    ".sql",
    ".proto", ".graphql",
})

# ── Non-source path exclusions ──────────────────────────────────────────

_SAFE_PREFIXES = (
    "/tmp/",
    "/var/",
    "/dev/null",
)

_SAFE_DIRNAMES = (
    "node_modules/",
    "__pycache__/",
    ".git/",
    "build/",
    "dist/",
    ".next/",
    ".nuxt/",
    "vendor/",
    "target/",  # Rust/Java build output
)

# ── Layer 1: Python inline write detection ────────────────────────────────

_RE_PYTHON_INLINE = re.compile(
    r"python[23]?\s+-c\b"
    r"|python[23]?\s+/dev/stdin"
    r"|python[23]?\s*<<",
    re.MULTILINE,
)

_WRITE_INDICATORS = (
    ".write(", ".write_text(", ".write_bytes(",
    "write_file(", "patch(",
)

_RE_OPEN_WRITE = re.compile(r"open\s*\([^)]*['\"]w['\"]")

def _is_python_inline_write(command: str) -> bool:
    """Check if command is a python3 inline script with file writes.

    Contract:
      Preconditions: command is non-empty string
      Postconditions: True iff python3 inline AND has write indicator
    """
    if not _RE_PYTHON_INLINE.search(command):
        return False
    for indicator in _WRITE_INDICATORS:
        if indicator in command:
            return True
    if _RE_OPEN_WRITE.search(command):
        return True
    return False


# ── Layer 2: Shell redirect to source file ───────────────────────────────

def _extract_redirect_target(command: str) -> Optional[str]:
    """Extract the target file path from a shell redirect (> or tee).

    Uses str methods only — no regex (R2 safety).

    Contract:
      Preconditions: command is non-empty string
      Postconditions: returns file path string or None
    """
    # Find ">" or "tee" and extract the next token
    lower = command.lower()
    # Strategy 1: find > and extract path after it
    pos = 0
    while True:
        idx = lower.find(">", pos)
        if idx < 0:
            break
        # Skip >> (append redirect)
        if idx + 1 < len(lower) and lower[idx + 1] == ">":
            pos = idx + 2
            continue
        # Extract token after >
        rest = command[idx + 1:].lstrip()
        path = _extract_first_token(rest)
        if path and not path.startswith("-"):
            return path
        pos = idx + 1

    # Strategy 2: find "tee" and extract path after it
    idx = lower.find("tee")
    if idx >= 0:
        rest = command[idx + 3:].lstrip()
        path = _extract_first_token(rest)
        if path and not path.startswith("-"):
            return path
    return None


def _extract_first_token(s: str) -> Optional[str]:
    """Extract first non-empty token from a string, stopping at shell metachar.

    Contract:
      Preconditions: s is non-empty string
      Postconditions: returns first token or None
    """
    # Stop at shell metacharacters: space, &, |, ;, >, newline
    for i, ch in enumerate(s):
        if ch in " \t\n&|;>\n":
            if i == 0:
                continue
            return s[:i].strip("'\"")
    return s.strip("'\"") if s else None


_RE_CAT_HEREDOC_WRITE = re.compile(
    r"cat\s*>",
    re.IGNORECASE,
)

_RE_TEE_WRITE = re.compile(
    r"\btee\b",
    re.IGNORECASE,
)


# ── Layer 3: In-place edit ───────────────────────────────────────────────

_RE_INPLACE_EDIT = re.compile(
    r"(?:perl\s+-pi|sed\s+-i|awk\s+-i)",
    re.IGNORECASE,
)

def _extract_inplace_target(command: str) -> Optional[str]:
    """Extract target file from an in-place edit command.

    Contract:
      Preconditions: command matches in-place edit pattern
      Postconditions: returns file path or None
    """
    # perl -pi -e 's/old/new/g' file.py
    # sed -i 's/old/new/' file.py
    # Handle: trailing filename after the last quoted/flag argument
    tokens = command.split()
    # Walk from end to find a file-like token
    for token in reversed(tokens):
        t = token.strip("'\"")
        if t.endswith("/") or t.startswith("-") or t.startswith("'"):
            continue
        # Could be a flag value or part of the script
        if any(ext in t.lower() for ext in (".py", ".ts", ".js", ".yaml", ".yml", ".json", ".md", ".sh")):
            return t
        # Check if the remaining tokens after -i flags form a plausible path
    # Fallback: try regex
    m = re.search(r"(?:perl|sed|awk)\s+(?:-\w+\s+)*['\"][^'\"]*['\"]?\s+(\S+)", command)
    if m:
        return m.group(1).strip("'\"")
    return None


# ── Path classification ──────────────────────────────────────────────────

def _is_source_file(path: str) -> bool:
    """Check if a path points to a source file by extension.

    Contract:
      Preconditions: path is non-empty string
      Postconditions: True iff path ends with a known source extension
    """
    _, ext = os.path.splitext(path)
    return ext.lower() in _SOURCE_EXTENSIONS


def _is_safe_path(path: str) -> bool:
    """Check if path is explicitly excluded from protection.

    Contract:
      Preconditions: path is non-empty string
      Postconditions: True iff path is a temp/build/cache path
    """
    lower = path.lower().replace("\\", "/")
    for prefix in _SAFE_PREFIXES:
        if lower.startswith(prefix):
            return True
    for dirname in _SAFE_DIRNAMES:
        if dirname in lower:
            return True
    return False


# ── Escape hatch ─────────────────────────────────────────────────────────

def _extract_all_write_targets(command: str) -> list:
    """Collect every detectable write target in a compound command.

    Contract:
      Preconditions: command is non-empty string
      Postconditions: returns list of target path strings (may be empty)
    """
    targets = []
    for part in _split_compound(command):
        t = _extract_redirect_target(part)
        if t:
            targets.append(t)
    return targets


def _split_compound(command: str) -> list:
    """Split a shell command on && ; | (best-effort, no shell exec).

    Contract:
      Preconditions: command is non-empty string
      Postconditions: returns list of sub-command strings
    """
    parts = []
    buf = []
    quote = None
    for ch in command:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            continue
        if ch in "&|;":
            if ch == "&" and buf and buf[-1] == "&":
                buf.pop()
                parts.append("".join(buf))
                buf = []
                continue
            if ch in ";|":
                parts.append("".join(buf))
                buf = []
                continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p for p in (s.strip() for s in parts) if p]


def _is_guard_owned_path(path: str) -> bool:
    """Check if a write target belongs to the guard itself (escape scope).

    Contract:
      Preconditions: path is non-empty string
      Postconditions: True iff path points inside the guard's own files
                      (read_think_gate / tool_executor / plugins dir /
                      .hermes config / hermes_constants)
    """
    p = path.replace("\\", "/")
    for keyword in _ESCAPE_KEYWORDS:
        if keyword in p:
            return True
    return False


_ESCAPE_KEYWORDS = (
    "agent/read_think_gate.py",
    "agent/tool_executor.py",
    "plugins/",
    ".hermes/config.yaml",
    "hermes_constants.py",
)


def _is_escape_hatch(command: str) -> bool:
    """Check if the *write target* is a guard-owned file (guard self-reference).

    2026-08-26 修复：旧逻辑只查命令里是否包含 _ESCAPE_KEYWORDS 任意字样——
    `ls plugins/` 出现在命令任意位置即整条命令放行，重定向写无关源码文件
    也被放走（进程内实测实锤）。修复后语义收窄：仅当每个被检测到的写入
    目标都落在护栏自身路径下才放行；无写入目标时才退回关键词全文匹配
    （兼容原「护栏自指」注释场景）。

    Contract:
      Preconditions: command is non-empty string
      Postconditions: True iff every detected write target is guard-owned
                      (or, when no target detectable, a keyword appears)
    """
    targets = _extract_all_write_targets(command)
    if targets:
        # 有可提取的写入目标：全部须为护栏自身文件才放行
        return all(_is_guard_owned_path(t) for t in targets)
    for keyword in _ESCAPE_KEYWORDS:
        if keyword in command:
            return True
    return False


# ── Combined detection ───────────────────────────────────────────────────

def _detect_source_write(command: str) -> Tuple[bool, str]:
    """Detect terminal command writing to a source file.

    Returns (blocked: bool, reason: str).

    Contract:
      Preconditions: command is non-empty string, tool_name == "terminal"
      Postconditions: (True, reason) iff source file write detected;
                       (False, "") iff safe or no write detected
    """
    # Layer 1: Python inline writes
    if _is_python_inline_write(command):
        return True, "python3 内联脚本含写文件操作"

    # Layer 2: Shell redirect (cat > / tee / echo > / printf >)
    if _RE_CAT_HEREDOC_WRITE.search(command):
        target = _extract_redirect_target(command)
        if target and _is_source_file(target) and not _is_safe_path(target):
            return True, f"shell 重定向写入源码文件 ({target})"

    if _RE_TEE_WRITE.search(command):
        target = _extract_redirect_target(command)
        if target and _is_source_file(target) and not _is_safe_path(target):
            return True, f"tee 写入源码文件 ({target})"
        # tee without explicit target but writing to source
        for ext in _SOURCE_EXTENSIONS:
            if ext in command.lower() and not _is_safe_path(command):
                return True, f"tee 写入源码文件"

    # Generic > redirect (echo, printf, etc.)
    target = _extract_redirect_target(command)
    if target and _is_source_file(target) and not _is_safe_path(target):
        return True, f"shell 重定向写入源码文件 ({target})"

    # Layer 3: In-place edit (sed -i, perl -pi)
    if _RE_INPLACE_EDIT.search(command):
        target = _extract_inplace_target(command)
        if target and _is_source_file(target) and not _is_safe_path(target):
            return True, f"in-place 编辑源码文件 ({target})"
        # If can't extract target but pattern matches, block conservatively
        # UNLESS the command is clearly non-source (e.g., sed -i on /tmp)
        if not any(_is_safe_path(t) for t in command.split()):
            return True, f"in-place 编辑 (无法提取目标文件，保守拦截)"

    return False, ""


# ── Hook entry ────────────────────────────────────────────────────────────

_BLOCK_MESSAGE = (
    "[SourceCodeWriteGuard] terminal 中检测到源码文件写操作，已拦截。\n\n"
    "请在 CLI 中可见 diff 的工具完成源码修改：\n"
    "  - 修改文件 → 用 patch 工具（返回值就是 diff）\n"
    "  - 新建文件 → 用 write_file 工具\n\n"
    "如果 patch 被护栏拦住（护栏自指等），在 command 中\n"
    "包含被拦文件的路径关键词（如 plugins/）即可自动放行。"
)


def on_pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """pre_tool_call hook: block terminal commands that write source files.

    Contract:
      Preconditions: kwargs contains tool_name (str) and args (dict with command key)
      Postconditions: returns None (allow) or {"action":"block","message":str} (block)
    """
    tool_name = kwargs.get("tool_name", "")
    if tool_name != "terminal":
        return None

    args = kwargs.get("args") or {}
    command = str(args.get("command", ""))
    if not command:
        return None

    if _is_escape_hatch(command):
        logger.info(
            "source-code-write-guard: escape hatch triggered — "
            "guard file modification allowed via terminal"
        )
        return None

    blocked, reason = _detect_source_write(command)
    if not blocked:
        return None

    logger.warning(
        "source-code-write-guard: blocked terminal source write — %s "
        "(command length=%d)",
        reason,
        len(command),
    )
    return {"action": "block", "message": _BLOCK_MESSAGE}


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("source-code-write-guard plugin registered")
