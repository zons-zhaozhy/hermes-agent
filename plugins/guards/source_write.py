"""guards.source_write — intercept terminal commands that silently write source files.

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

import ast
import logging
import os
import re  # re-ok: shell命令扫描器,正则即本体实现 # noqa: scanner plugin
from typing import Any, Optional

from plugins.guards.duplicate_check import _execute_code_write_targets  # 写形态识别单一事实源

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
# 2026-10-19 根修（守卫误拦同型缺陷第四轮）：旧实现 _RE_PYTHON_INLINE 对整条
# 命令全文子串匹配（引号盲），载荷字符串数据里的 python3 -c "open('x.py','w')"
# 字样被当 shell 命令——守卫验证命令（外层 python heredoc，open 形态仅存在于
# 字符串字面量）被整条拦截（实测）。新实现只取 shell 层 python 调用（段头）的
# 真实载荷（-c 参数 / heredoc 体），AST 识别真实写调用；动态路径写保守拦截
# （旧版提取伪目标 "w" 后早退放行=旁路漏报，一并收紧）。

_PY_HEADS = frozenset({"python", "python3"})

_WRITE_INDICATORS = (
    ".write(", ".write_text(", ".write_bytes(",
    "write_file(", "patch(",
)

_RE_OPEN_WRITE = re.compile(r"open\s*\([^)]*['\"]w['\"]")  # re-ok: shell命令扫描器本体,保守回退分支专用


def _head_tokens(head_line: str) -> list:
    """首行 token 化（shlex 引号感知；不闭合退化空格切分）。

    Contract:
      Postconditions: 恒返回列表（可能为空表），不抛异常。
    """
    try:
        import shlex

        return shlex.split(head_line, posix=True)
    except ValueError:
        return head_line.split()


def _strip_to_marker(body: str, marker: str) -> str:
    """截掉 heredoc 终止标记行及其后内容。

    Contract:
      Postconditions: 无匹配标记行时原样返回（best-effort，多余文本由
        AST 解析失败分支回退保守判据兜住）。
    """
    lines = body.split("\n")
    for idx, line in enumerate(lines):
        if line.strip() == marker:
            return "\n".join(lines[:idx])
    return body


def _flag_c_payload(toks: list) -> Optional[str]:
    """-c 参数 token（无 -c 返回 None）。

    Contract:
      Postconditions: 返回紧跟 -c 的 token 或 None。
    """
    for k, t in enumerate(toks):
        if t == "-c" and k + 1 < len(toks):
            return toks[k + 1]
    return None


def _segment_python_payload(seg: str) -> Optional[tuple]:
    """单段 shell 层 python 载荷（非 python 段头返回 None）。

    Contract:
      Postconditions: 返回 (payload, trusted)——trusted=True=载荷文本可靠
        （-c 参数 / heredoc 体），False=载荷不可取（/dev/stdin）须保守回退；
        echo 等非 python 段头 → None。
    """
    head_line, _nl, tail = seg.partition("\n")
    toks = _head_tokens(head_line)
    if not toks or toks[0].rsplit("/", 1)[-1] not in _PY_HEADS:
        return None
    payload = _flag_c_payload(toks)
    if payload is not None:
        return payload, True
    if "<<" in head_line and tail:
        marker = head_line.split("<<", 1)[1].strip().strip("'\"")
        return _strip_to_marker(tail, marker), True
    if "/dev/stdin" in head_line:
        return seg, False
    return None


def _python_payloads(command: str) -> list:
    """shell 层 python 调用的真实载荷列表（含可靠性标记）。

    Contract:
      Postconditions: 仅 python 段头的段产出条目；无 python 段头 → 空表。
    """
    out: list = []
    for seg in _split_compound(command):
        item = _segment_python_payload(seg)
        if item is not None:
            out.append(item)
    return out


def _is_write_mode(node: ast.AST) -> bool:
    """open 第二实参是否写模式字面量（语义与 duplicate_check._target_from_open 一致）。"""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and any(m in node.value for m in ("w", "a", "x"))
    )


def _is_write_call(node: ast.Call) -> bool:
    """ast.Call 是否写形态（open 写模式 / write·write_text·write_bytes / write_file）。"""
    fn = node.func
    if isinstance(fn, ast.Name) and fn.id == "open" and len(node.args) > 1:
        return _is_write_mode(node.args[1])
    if isinstance(fn, ast.Attribute) and fn.attr in ("write_text", "write_bytes", "write"):
        return True
    return isinstance(fn, ast.Name) and fn.id == "write_file"


def _ast_write_signals(payload: str) -> tuple:
    """AST 级写信号：真实写调用判定 + 字面量写目标提取。

    Contract:
      Postconditions: 返回 (parse_ok, has_write_call, 字面量目标列表)；
        parse_ok=False=载荷非合法 Python 源（调用方回退保守判据）；
        has_write_call=存在写形态 ast.Call（含动态路径形态——保守拦截依据）；
        目标提取委托 duplicate_check._execute_code_write_targets（单一事实源）。
    """
    try:
        tree = ast.parse(payload)
    except SyntaxError:
        return False, False, []
    targets = [p for p, _c in _execute_code_write_targets(payload, tuple(_SOURCE_EXTENSIONS))]
    has = any(isinstance(n, ast.Call) and _is_write_call(n) for n in ast.walk(tree))
    return True, has or bool(targets), targets


def _payload_write_verdict(payload: str, trusted: bool) -> bool:
    """单载荷写判定（True=拦截）。

    Contract:
      Postconditions: trusted 载荷 AST 可解析时按（字面量目标 / 写调用保守）
        判定; 不可解析或非受信载荷回退旧保守判据（写指示词/open 写模式正则）。
    """
    if trusted:
        parse_ok, has_write, targets = _ast_write_signals(payload)
        if parse_ok:
            if any(_is_source_file(t) and not _is_safe_path(t) for t in targets):
                return True
            if not targets and has_write:
                return True
            return False
    if any(ind in payload for ind in _WRITE_INDICATORS):
        return True
    return bool(_RE_OPEN_WRITE.search(payload))  # re-ok: shell命令扫描器本体,保守回退分支


def _is_python_inline_write(command: str) -> bool:
    """Check if command is a python inline script with file writes.

    只拦「写到源码路径」的内联写——open('/tmp/..','w') 等安全路径放行
    （对齐插件头 Exclusions 契约与 Layer 2/3 的 _is_safe_path 判定）。
    载荷字符串字面量里的写形态字样（测试数据/文档样例）不是写动作。

    Contract:
      Preconditions: command is non-empty string
      Postconditions: True iff 任一 shell 层 python 载荷判定为写（见
        _payload_write_verdict）。
    """
    return any(_payload_write_verdict(payload, trusted) for payload, trusted in _python_payloads(command))


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

# shell 求值标志：其后的引号段会被 shell 当命令执行（bash -c "..." / find -exec）
_EXEC_EVAL_FLAGS = frozenset({"-c", "-exec", "-execdir", "eval", "--command", "-command"})

def _executable_text(command: str) -> str:
    """取出命令中「会被 shell 执行」的文本，供 in-place 模式匹配使用。

    引号外文本原样保留；引号段仅当其前一个 token 是 shell 求值标志（-c/-exec/eval）
    时保留，其余引号段（提交信息、grep 模式等数据）丢弃。否则
    `git commit -m "... sed -i ..." -- x.py` 会把消息文本当成真的 sed 命令，
    并把 in-place 目标错误归属到命令行里另一处的源码路径（实测误拦）。

    Contract:
      Preconditions: command 为 str。
      Postconditions: 返回 str；保留片段与原文逐字一致；不抛异常。
      Invariants: 只用 str 方法（无正则）；只改引号段的取舍，不改其余字符。
    """
    out: list[str] = []
    quote = ""
    token = ""
    prev = ""
    keep = False
    for ch in command:
        if quote:
            if ch == quote:
                quote = ""
                continue
            if keep:
                out.append(ch)
            continue
        if ch in ("'", '"'):
            quote = ch
            keep = prev in _EXEC_EVAL_FLAGS
            continue
        if ch in " \t\n":
            if token:
                prev = token
                token = ""
            out.append(" ")
            continue
        if ch in "&|;":
            prev = ""
            token = ""
            out.append(" ")
            continue
        token += ch
        out.append(ch)
    return "".join(out)

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
    m = re.search(r"(?:perl|sed|awk)\s+(?:-\w+\s+)*['\"][^'\"]*['\"]?\s+(\S+)", command)  # re-ok: shell命令扫描器本体
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

def _detect_source_write(command: str) -> tuple[bool, str]:
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
    if _RE_CAT_HEREDOC_WRITE.search(command):  # re-ok: shell命令扫描器本体
        target = _extract_redirect_target(command)
        if target and _is_source_file(target) and not _is_safe_path(target):
            return True, f"shell 重定向写入源码文件 ({target})"

    if _RE_TEE_WRITE.search(command):  # re-ok: shell命令扫描器本体
        target = _extract_redirect_target(command)
        if target and _is_source_file(target) and not _is_safe_path(target):
            return True, f"tee 写入源码文件 ({target})"
        # tee without explicit target but writing to source
        for ext in _SOURCE_EXTENSIONS:
            if ext in command.lower() and not _is_safe_path(command):
                return True, "tee 写入源码文件"

    # Generic > redirect (echo, printf, etc.)
    target = _extract_redirect_target(command)
    if target and _is_source_file(target) and not _is_safe_path(target):
        return True, f"shell 重定向写入源码文件 ({target})"

    # Layer 3: In-place edit (sed -i, perl -pi)
    # 只在「会被 shell 执行」的文本上匹配：引号外部分 + shell 求值标志（-c/-exec/eval）
    # 后的引号段。否则提交信息/grep 模式里出现该字样即拦住整条无关命令。
    if _RE_INPLACE_EDIT.search(_executable_text(command)):  # re-ok: shell命令扫描器本体
        target = _extract_inplace_target(command)
        if target and _is_source_file(target) and not _is_safe_path(target):
            return True, f"in-place 编辑源码文件 ({target})"
        # If can't extract target but pattern matches, block conservatively
        # UNLESS the command is clearly non-source (e.g., sed -i on /tmp)
        if not any(_is_safe_path(t) for t in command.split()):
            return True, "in-place 编辑 (无法提取目标文件，保守拦截)"

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


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
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
            "guards.source_write: escape hatch triggered — "
            "guard file modification allowed via terminal"
        )
        return None

    blocked, reason = _detect_source_write(command)
    if not blocked:
        return None

    logger.warning(
        "guards.source_write: blocked terminal source write — %s "
        "(command length=%d)",
        reason,
        len(command),
    )
    return {"action": "block", "message": _BLOCK_MESSAGE}


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.source_write registered")
