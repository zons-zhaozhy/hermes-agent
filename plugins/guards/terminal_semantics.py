"""terminal_semantics —— terminal 命令写文件语义的唯一事实源（C3 收敛）。

背景（2026-11-03 反方审查定性的架构债）: 全库曾散布六套 terminal 写意图
解析器（gate._terminal_writes_file 字符扫描 / source_write._extract_
redirect_target / casebook._redirect_targets token 级 / patch-first sed
检测 / block_escalation._redirects_write / diff_debt 工具参数账本）,各持
一份启发式=各有一份误报面,同一命令六处判定可互不相同:
  - gate 曾要求目标含 "." 才判写（`echo hi > out` 漏判为只读）;
  - source_write 曾跳过 ">>" append 形态;
  - casebook 曾把 "2>/dev/null" 当写目标（E2 根修）。

本模块收敛「重定向流」与「目标路径形态」两个原子,语义基底取六套中最全
的 block_escalation._next_redirect_sink（append/fd_dup/进程替换/sink 四
形态全覆盖）,消费方一律 import 本模块,禁止再持私有重定向解析。

Contract:
  Preconditions: 输入为 str（可为任意残缺 shell 片段,含未闭合引号——
    字符扫描天然容错,不用 shlex）。
  Postconditions: 全部函数纯函数、零正则、永不抛异常。
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

# 与 shell 语义对应的 sink 目标: 重定向到它们不构成文件写。
HARMLESS_SINKS = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout"})

# 系统暂存目录前缀——OS 约定的临时区（日志采集/跑批输出落点）。本模块只
# 提供判定原子，不改变任何既有写语义；消费方按自身策略决定是否豁免
# （read_think_gate 豁免远端跑批日志，source_write/approval 不豁免）。
SCRATCH_PATH_PREFIXES: tuple[str, ...] = (
    "/tmp/", "/private/tmp/", "/var/tmp/", "/dev/shm/",  # no-tmp: ok — 只识别 shell scratch 前缀，不创建或访问路径。
)


# 段首命令名判定为只读的基础集合。具体命令仍需先经过重定向/专用语义
# （例如 sed 写形态）排除；这不是“任意参数都安全”的完整命令分析器。
READ_COMMANDS: frozenset[str] = frozenset({
    "ls", "stat", "file", "find", "grep", "rg", "ag", "cat", "head", "tail",
    "wc", "echo", "printf", "date", "pwd", "whoami", "uname", "ps", "pgrep",
    "lsof", "df", "du", "which", "type", "printenv", "id", "groups",
    "hostname", "sw_vers", "sysctl", "uptime", "jq", "awk", "sed", "cut",
    "sort", "uniq", "tr", "column", "diff", "cmp", "comm", "md5", "md5sum",
    "shasum", "sha1sum", "sha256sum", "base64", "xxd", "strings", "nm",
    "otool", "exiftool", "sleep", "true", "false", "test", "[", "cd",
    "export", "set", "unset", "alias", "unalias", "wait", "mdfind", "mdls",
    "pbpaste", "tree", "history", "vm_stat", "iostat", "netstat",
})

# 目标 token 扫描的字符上限（防超长串把整段命令吞成"目标"）。
_MAX_TARGET_CHARS = 4096


@dataclass(frozen=True)
class Redirect:
    """一次重定向流的解析结果。

    Contract:
      Postconditions: append=">>" 形态; fd_dup=">&n" 描述符复制（非文件写）;
        is_procsub=">(cmd)" 进程替换（非文件写）; target=目标 token 原样
        （可为空串——行尾裸 ">"）。writes_file 属性给出「是否构成文件写」
        的综合判定（sink 排除+数字字面量排除下沉到 looks_like_path）。
    """

    append: bool
    fd_dup: bool
    is_procsub: bool
    target: str

    @property
    def writes_file(self) -> bool:
        """是否构成文件写: 非 fd 复制、非进程替换、非 sink、目标非空
        且非数字字面量。append 同样是文件写。"""
        if self.fd_dup or self.is_procsub:
            return False
        tok = self.target.strip("\"'")
        return bool(tok) and tok not in HARMLESS_SINKS and not _is_numeric(tok)


def _is_numeric(tok: str) -> bool:
    """纯数字字面量（含小数）——SQL 比较符 `WHERE price > 9.99` 的 9.99
    不是重定向目标（2026-11-03 D2 根修,从 gate 下沉到唯一事实源）。"""
    return tok.replace(".", "", 1).isdigit()


def is_read_command(command_name: str) -> bool:
    """命令名（允许绝对/相对可执行路径）是否属于基础只读命令集。"""
    name = command_name.strip("\"'").rsplit("/", 1)[-1]
    return name in READ_COMMANDS


def _words(seg: str) -> list[str]:
    """把 shell/Python 片段容错拆成语义词。

    shlex 是首选；残缺引号时退回空白拆分。再剥掉 Python 容器与 shell
    连接符，使 ``["sed", "-i"]`` 这类嵌入形态也能进入同一判据。
    """
    try:
        raw_words = shlex.split(seg, posix=True)
    except ValueError:
        raw_words = seg.split()
    separators = " \t\r\n()[]{}\"'`,;|&<>"
    table = str.maketrans({char: " " for char in separators})
    return [word for raw in raw_words for word in raw.translate(table).split()]


def _is_sed_substitution(word: str) -> bool:
    """是否为 ``s<delim>...<delim>...<delim>`` 替换脚本。"""
    if len(word) < 4 or not word.startswith("s"):
        return False
    delimiter = word[1]
    if delimiter.isalnum():
        return False
    return word.count(delimiter) >= 3


def sed_writes_file(command: str) -> bool:
    """sed 是否构成 patch-first 语义上的源文件写。

    ``-i`` 是实际写文件；非 quiet 替换即使只输出到 stdout，也属于
    patch-first 明确禁止的批量替换形态。``sed -n ...p`` 是只读查看。
    本函数只做文本分类，不执行 shell，残缺输入返回保守/只读结论。
    """
    words = _words(command)
    for index, word in enumerate(words):
        if word != "sed":
            continue
        arguments = words[index + 1:index + 9]
        if any(arg == "-i" or arg.startswith("-i") for arg in arguments):
            return True
        if any(arg in {"-n", "--quiet", "--silent"} for arg in arguments):
            return False
        scripts = (
            arg.removeprefix("--expression=").removeprefix("-e")
            for arg in arguments
        )
        if any(_is_sed_substitution(script) for script in scripts):
            return True
    return False


def looks_like_path(tok: str) -> bool:
    """token 是否具备文件路径形态（消费方按需选用的窄口径判定）。

    Contract:
      Postconditions: 剥引号后含 "/" 前缀（绝对/相对路径）、"." 开头、
        含 "." 扩展名、或常见源码/配置后缀 → True; 纯数字/裸词/空 → False。
    """
    t = tok.strip("\"'")
    if not t or t.startswith("-"):
        return False
    if _is_numeric(t):
        return False
    if t.startswith(("/", "./", "../", "~")):
        return True
    return "." in t


def parse_redirects(seg: str) -> list[Redirect]:
    """解析一段 shell 文本里全部重定向流（引号感知字符扫描,容错残缺引号）。

    Contract:
      Preconditions: seg 为 str。
      Postconditions: 按出现顺序返回; ">" 字符逐个考察, ">>" 合并为一条;
        fd 前缀（"2>"）与 "&>" 不产生额外条目（fd_dup 语义并入该条）;
        永不抛异常。
    """
    out: list[Redirect] = []
    positions = _unquoted_gt_positions(seg)
    position_index = 0
    n = len(seg)
    while True:
        if position_index >= len(positions):
            return out
        gt = positions[position_index]
        position_index += 1
        append = seg[gt + 1: gt + 2] == ">"
        nxt = gt + (2 if append else 1)
        is_procsub = seg[gt + 1: gt + 2] == "("
        j = nxt
        while j < n and seg[j] == " ":
            j += 1
        sink_end = j
        while sink_end < n and seg[sink_end] not in " ;|&\t" and sink_end - j < _MAX_TARGET_CHARS:
            sink_end += 1
        fd_dup = (gt > 0 and seg[gt - 1] == "&") or seg[nxt: nxt + 1] == "&"
        out.append(Redirect(append, fd_dup, is_procsub, seg[j:sink_end]))
        while position_index < len(positions) and positions[position_index] < sink_end:
            position_index += 1


def _unquoted_gt_positions(seg: str) -> tuple[int, ...]:
    """Return ``>`` character positions that shell interpretation sees as syntax.

    Quoted program text (including awk comparisons) and quoted literals are data,
    not redirection. An unclosed quote makes the remainder data, preserving the
    malformed-input contract without raising.
    """
    positions: list[int] = []
    quote: str | None = None
    escaped = False
    for index, char in enumerate(seg):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == ">":
            positions.append(index)
    return tuple(positions)


def redirect_writes_file(seg: str) -> bool:
    """段内是否存在构成文件写的重定向（宽口径: 非 sink 目标即写,
    `> out` 无扩展名同样是写——gate 旧窄口径的漏判修正）。"""
    return any(r.writes_file for r in parse_redirects(seg))


def redirect_file_targets(seg: str) -> list[str]:
    """提取段内构成文件写的重定向目标列表（消费方做归属判定用）。"""
    return [r.target for r in parse_redirects(seg) if r.writes_file]
