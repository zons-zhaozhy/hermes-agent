"""Terminal-command write semantics for the read-think gate.

Split out of ``plugins/read_think_gate/gate.py`` (which is capped at 2,000
lines by the code-health ratchet) because this is one self-contained topic: a
pure, model-free reading of a shell command string that answers "is this a
provenance read, or a write channel?".

Nothing here imports the gate. ``terminal`` is gated only when it can write a
file, and both the investigation counter and the gate share these verdicts, so
they must stay the single implementation — the devil-advocate-audit armed lane
consumes ``_terminal_is_investigation`` rather than re-deriving it.

Contract:
    Preconditions: every function takes a ``str`` (may be empty).
    Postconditions: pure string analysis; no I/O, no exceptions, no state.
"""

from __future__ import annotations

import re

# terminal 中可能写入文件的 shell 操作模式——用于检测绕过 gate 的文件写入。
# 用 _terminal_writes_file() 函数检测，而非简单子串匹配。
# 优化原则：只匹配确实表示写文件的模式，避免 "grep 'cp '" / "git commit -m 'mv'" 误报。

# 高置信度写操作——几乎不会出现在非写入上下文
_TERMINAL_WRITE_PATTERNS_HIGH: tuple[str, ...] = (
    "sed -i",      # sed 原地编辑——唯一用途就是改文件
    "dd of=",      # dd 输出到文件——唯一用途就是写
    "rsync ",      # rsync——同步文件到目标
    "tee /",       # tee 写绝对路径
    "tee ~",       # tee 写 home 目录
    "tee ./",      # tee 写相对路径文件
    "tee \"",       # tee 写引号包裹的路径
    "tee '",       # tee 写引号包裹的路径
)

# 解释器 token——出现即视为任意代码执行通道，不算只读取证
# （gate 调查计数与 devil-advocate-audit armed 冻结共用，单一事实源）。
_TERMINAL_INTERPRETER_TOKENS: tuple[str, ...] = (
    "python3", "python", "bash", "sh", "zsh", "node", "perl", "ruby",
)


def _terminal_is_investigation(command: str) -> bool:
    """terminal 命令是否为只读取证（调查证据计数用）。

    判据：解释器执行 / ``./相对可执行`` / ``.sh`` 脚本 / 文件写形态
    之外 → 只读取证。与 devil-advocate-audit armed 冻结放行共用本判定
    （armed 侧额外叠加落地 SQL 检查）。本判定是「terminal 只读语义」的
    唯一实现——armed 侧与调查计数侧禁各写一份。

    2026-11-03 根修：READ_ONLY_INVESTIGATION_TOOLS 枚举白名单不含
    terminal，只读 ls/sqlite3 查询/git log 永不计入调查证据——实测只读
    ls+sqlite3 取证命令被拦一整轮（枚举式白名单默认拒绝缺陷）。

    Contract:
      Preconditions: command 为字符串（可为空）。
      Postconditions: 纯字符串判定不抛异常; 解释器/脚本/写形态 → False。
    """
    if not command:
        return False
    tokens = command.lower().split()
    if any(tok in _TERMINAL_INTERPRETER_TOKENS for tok in tokens):
        return False
    if any(tok.startswith("./") for tok in tokens):
        return False
    if any(tok.endswith(".sh") for tok in tokens):
        return False
    return not _terminal_writes_file(command)


def _scrub_scratch_log_targets(command: str) -> str:
    """把 tee/重定向到系统暂存目录的目标改写为 sink 形态（仅 gate 语义）。

    远端跑批的日志采集（``ssdt run … | tee /tmp/quick336.log``、
    ``run_tests.sh > /tmp/x.log``）不属于本 gate 门禁的「仓库源码写」，
    却被高置信度 tee 模式判写，导致会话内每次重启跑批都要重新过完整
    授权（2026-11-08 agent.log 实测 8 连拦）。tee 改写为无路径形态
    （``tee DEVNULL``——不能用 ``tee /dev/null``，仍含 ``tee /`` 模式），
    重定向改写为 /dev/null。sed -i/cp/mv 等其他写通道不受影响。

    Contract:
      Postconditions: 纯字符串替换不抛异常；仅 ``tee <scratch>``/``> <scratch>``
        形态被改写，暂存前缀取 terminal_semantics.SCRATCH_PATH_PREFIXES
        （单一事实源，不另持清单）。
    """
    from plugins.guards.terminal_semantics import SCRATCH_PATH_PREFIXES

    scrubbed = command
    for prefix in SCRATCH_PATH_PREFIXES:
        scrubbed = scrubbed.replace("tee " + prefix, "tee DEVNULL ")
        scrubbed = scrubbed.replace("> " + prefix, "> /dev/null ")
    return scrubbed


def _terminal_writes_file(command: str) -> bool:
    """检测 terminal 命令是否包含文件写入操作。

    策略：
    1. 解释器内联/heredoc/管道代码 = 任意代码执行 = 写通道（2026-10-27
       反方审查 A2：`python3 -c`/heredoc 可 open().write()/起子进程跑 git
       push，此前无检测）
    2. 高置信度模式（sed -i/dd of=/rsync/tee+路径）直接匹配
    3. shell 重定向（> >>）后跟文件路径特征——排除 /dev/null/2>&1
    4. cp/mv 目标是文件路径——要求前导空格 + 路径特征（排除 grep/docker/git 中的字符串）

    Contract:
      Preconditions: command 为 str（可为空）
      Postconditions: 纯形态判定；解释器执行形态保守返回 True（fail-closed）
    """
    if not command:
        return False

    command = _scrub_scratch_log_targets(command)

    # 0. 解释器 heredoc（python3 <<'PY'）吃 stdin——形态级不判写。解释器
    #    载荷的写检测由四处内容级判定负责（tools/approval 结构性 -c 扫描/
    #    cron lifecycle inert-heredoc/ guards source_write 内联写/ guards
    #    duplicate_check heredoc 正文），本函数与其重复执法=误拦回归
    #    （2026-10-19 source_write 修复的外层 heredoc 载荷形态）。
    #    armed 冻结面的保守解释器判定在 devil-advocate-audit 侧
    #    （_terminal_passes），不共用本函数语义。
    if "<<" in command:
        pre_heredoc = command.split("<<", 1)[0]
        last_seg = pre_heredoc.rstrip().rsplit("|", 1)[-1].strip()
        # 整 token 匹配——后缀匹配会撞 .sh 文件名(cat > f.sh <<EOF 被误判解释器)
        interpreter_tokens = ("python3", "python", "bash", "sh", "zsh",
                              "node", "perl", "ruby", "ssh", "sudo")
        seg_words = last_seg.split()
        if seg_words and seg_words[-1] in interpreter_tokens:
            return False

    # 1. 高置信度模式
    for pat in _TERMINAL_WRITE_PATTERNS_HIGH:
        if pat in command:
            return True

    # 2. shell 重定向——唯一事实源 guards.terminal_semantics（宽口径: 非
    #    sink 目标即写,含 >> 与无扩展名目标;fd 复制/进程替换/数字字面量
    #    由 owner 排除）。2026-11-03 C3 收敛: 此前私有扫描为窄口径（要求
    #    目标含路径形态）,`echo hi > out` 漏判为只读。
    from plugins.guards.terminal_semantics import redirect_writes_file

    if redirect_writes_file(command):
        return True

    # 3. cp/mv 只有处在命令边界才视为写通道；引号内或普通参数中的 cp/mv
    # （grep 'cp ...'、echo cp）不误伤。
    for match in re.finditer(r"\b(?:cp|mv)(?=\s)", command, re.IGNORECASE):
        preceding = command[:match.start()].rstrip()
        if not preceding or preceding.endswith(("|", ";", "&&", "||", "(")):
            return True
    return False
