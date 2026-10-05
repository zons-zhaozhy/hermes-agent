"""block-escalation plugin.

拦截置信度升级：拦截器只认「单次调用」，不认「意图」——被拦后换一条
通道重试同一意图时，各拦截器互不知情，每次都只看到第 1 次。本插件在
post_tool_call 观察点收敛「意图指纹」：对 status="blocked" 的调用按
被改写目标（路径/命令文本）计算指纹，同一指纹在时间窗内第 2 次被拦
即升级——transform_llm_output 注入用户可见警示强制停下，禁止第三次
尝试。

持久化：指纹计数落 HERMES_HOME/block_escalation.db（专用库，范式同
outcome-collector 的 outcomes.db，WAL + 进程内锁）。跨会话、跨进程
共享计数——换会话重试同一意图同样在第 2 次被拦时升级。

指纹设计（第一性：意图 = 对什么目标做什么写操作）：
  - terminal: 被拦命令中出现的文件路径（改写目标）
  - write_file/patch: 目标 path
  - execute_code 写操作: code 中出现的文件路径
  - 无路径可提取时退化为命令文本的 token 集合（排序去重），忽略通道差异
"""

import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger("hermes.plugin.block-escalation")

_WINDOW_SECS = 1800          # 同意图重复被拦的判定时间窗
_STREAK_LIMIT = 2            # 窗口内第 2 次即升级（用户拍板：连续 2 次同因=停）
_STATE_CAP = 512             # 库内指纹行数上限（防膨胀，驱逐最旧）

# 只读命令白名单：段首词命中且无落盘重定向时，该段判为只读。
# 未列出的命令一律按写意图计数（保守方向：漏计升级 < 误伤防线）。
_READ_CMDS = frozenset({
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
_WRAPPERS = frozenset({"sudo", "nohup", "command", "time", "nice", "env", "exec"})
_GIT_READ_SUBCMDS = frozenset({
    "status", "log", "diff", "show", "rev-list", "branch", "blame",
    "shortlog", "describe", "ls-files", "ls-remote", "remote", "tag",
    "cat-file", "name-only", "reflog", "whatchanged", "grep", "config",
})
_SQL_WRITE_VERBS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "REPLACE",
    "ATTACH", "DETACH", "VACUUM", "PRAGMA",
)
_HARMLESS_SINKS = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout"})

# 升级终止提示前缀。模块级常量供 cron/scheduler.py 探测：含此标记的
# final_response 说明 agent 被拦截升级终止而非正常完成，scheduler 须按
# 失败记账（last_status=error），禁假绿（与 issue #17855 同一不变量）。
ESCALATION_MARKER = "[拦截升级]"

_db_lock = threading.Lock()
_db_path = None
_escalated: set = set()      # 本进程已升级指纹（提示单次消费）


def _get_db_path() -> Path:
    """Contract: Preconditions: 无；Postconditions: 返回活跃 HERMES_HOME 下的
    block_escalation.db 路径，解析失败时回退 HERMES_HOME 环境变量。"""
    global _db_path
    if _db_path is not None:
        return _db_path
    try:
        from hermes_constants import get_hermes_home
        _db_path = get_hermes_home() / "block_escalation.db"
    except Exception:  # get_hermes_home 不可用时回退环境变量，行为不中断
        import os
        home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
        _db_path = Path(home) / "block_escalation.db"
    return _db_path


def _get_conn() -> sqlite3.Connection:
    """Contract: Preconditions: db 路径可写；Postconditions: 返回 WAL 模式
    连接（库不存在则创建），Row 工厂已设。"""
    conn = sqlite3.connect(str(_get_db_path()), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Contract: Preconditions: conn 为可写连接；Postconditions:
    block_streaks 表与指纹索引存在（幂等）。"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS block_streaks (
            fingerprint TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0,
            last_ts REAL NOT NULL,
            tools TEXT NOT NULL DEFAULT '',
            sample TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_bs_last_ts ON block_streaks(last_ts);
    """)
    conn.commit()


def _next_redirect_sink(seg: str, gt: int) -> tuple:
    """Contract: Preconditions: gt 为 seg 中 '>' 下标；Postconditions: 返回
    (append, fd_dup, is_procsub, sink, scan_from)——sink 为目标 token，
    scan_from 为继续扫描起点。"""
    append = seg[gt + 1:gt + 2] == ">"
    nxt = gt + (2 if append else 1)
    is_procsub = seg[gt + 1:gt + 2] == "("
    j = nxt
    while j < len(seg) and seg[j] == " ":
        j += 1
    sink_end = j
    while sink_end < len(seg) and seg[sink_end] not in " ;|&\t":
        sink_end += 1
    fd_dup = (gt > 0 and seg[gt - 1] == "&") or seg[nxt:nxt + 1] == "&"
    return append, fd_dup, is_procsub, seg[j:sink_end], max(sink_end, gt + 1)


def _redirects_write(seg: str) -> bool:
    """Contract: Preconditions: seg 为 str；Postconditions: 存在落盘重定向
    （> file / >> file，丢弃 >/dev/null 族与进程替换 >(）时返回 True。
    零正则，纯 str 方法。"""
    i = 0
    while True:
        gt = seg.find(">", i)
        if gt == -1:
            return False
        append, fd_dup, is_procsub, sink, i = _next_redirect_sink(seg, gt)
        if (append or not fd_dup) and not is_procsub \
                and sink not in _HARMLESS_SINKS:
            return True


def _head_index(words: list) -> int:
    """Contract: Preconditions: words 非空列表；Postconditions: 返回段首
    命令词（剥 wrapper 与带路径前缀白名单词后）的下标。"""
    wi = 0
    while wi < len(words):
        w = words[wi]
        if w in _WRAPPERS:
            wi += 1
            continue
        if "/" in w:
            base = w.rsplit("/", 1)[-1]
            if base in _READ_CMDS or base in _WRAPPERS:
                wi += 1
                continue
        return wi
    return wi


def _sql_is_readonly(words: list, wi: int) -> bool:
    """Contract: Preconditions: words[wi] 为 sqlite3/psql；Postconditions:
    命令串无 SQL 写动词（含 PRAGMA）时返回 True。"""
    quoted = " ".join(words[wi:]).upper()
    return not any(v in quoted for v in _SQL_WRITE_VERBS)


def _git_is_readonly(words: list, wi: int) -> bool:
    """Contract: Preconditions: words[wi] == 'git'；Postconditions: 跳过
    全局选项（-C 带参数）后，子命令属读集合时返回 True。"""
    rest = words[wi + 1:]
    ri = 0
    while ri < len(rest) and rest[ri].startswith("-"):
        if rest[ri] == "-C" and ri + 1 < len(rest):
            ri += 2
        else:
            ri += 1
    sub = rest[ri] if ri < len(rest) else ""
    return sub in _GIT_READ_SUBCMDS


def _segment_is_readonly(seg: str) -> bool:
    """Contract: Preconditions: seg 为命令段（已 strip）；Postconditions:
    返回该段是否可判定为只读——无落盘重定向，且段首词在白名单 / git 读
    子命令 / sqlite3·psql 纯查询。零正则。"""
    seg = seg.strip()
    if not seg:
        return True
    if _redirects_write(seg):
        return False
    words = seg.split()
    if not words:
        return True
    if "curl" in words or "wget" in words:
        return _fetch_is_readonly(words)
    wi = _head_index(words)
    head = words[wi] if wi < len(words) else ""
    if head == "git":
        return _git_is_readonly(words, wi)
    if head in ("sqlite3", "psql"):
        return _sql_is_readonly(words, wi)
    return head in _READ_CMDS


def _fetch_is_readonly(words: list) -> bool:
    """Contract: Preconditions: words 含 curl/wget；Postconditions: 所有
    -o/--output 目标均为 /dev/null（探活形态）时返回 True，否则 False。"""
    out_targets = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("-o", "--output"):
            i += 1
            if i < len(words):
                out_targets.append(words[i])
        elif w.startswith("--output="):
            out_targets.append(w[len("--output="):])
        i += 1
    return all(t.strip('\'"') == "/dev/null" for t in out_targets)


def _command_is_readonly(command: str) -> bool:
    """Contract: Preconditions: command 为 str；Postconditions: 命令所有非空
    段均只读时返回 True（&& || ; | 分隔符拆段）；任一段为写即 False。"""
    for raw in command.replace("&&", ";").replace("||", ";").split(";"):
        for pipe_seg in raw.split("|"):
            if not _segment_is_readonly(pipe_seg):
                return False
    return True


def _reset_for_test() -> None:
    """测试专用：清空进程内升级标记。"""
    _escalated.clear()


def _extract_paths(text: str) -> tuple:
    """Contract: Preconditions: text 为 str；Postconditions: 返回 text 中
    疑似文件路径的元组（优先带扩展名 token，否则含斜杠 token，排序）。纯 str 方法，零正则。"""
    if not text:
        return ()
    hits, loose = [], []
    for tok in text.replace('"', " ").replace("'", " ").split():
        tok = tok.strip("`,;()")
        if tok.endswith((".py", ".md", ".yaml", ".yml", ".json", ".ts", ".js",
                ".sh", ".sql", ".toml", ".txt", ".java", ".vue", ".xml")):
            hits.append(tok)
        elif "/" in tok and len(tok) > 3:
            loose.append(tok)
    pool = hits if hits else loose
    return tuple(sorted(set(pool))[:8])


def _intent_fingerprint(tool_name: str, args: dict) -> tuple:
    """Contract: Preconditions: args 为 dict；Postconditions: 返回意图指纹
    （目标路径元组，或退化 token 元组）；无内容时返回 ()（不参与计数）。"""
    if tool_name == "terminal":
        cmd = str(args.get("command") or "")
        return _extract_paths(cmd) or tuple(sorted(set(cmd.split()))[:12])
    if tool_name in ("write_file", "patch", "edit"):
        return (str(args.get("path") or ""),)
    if tool_name == "execute_code":
        return _extract_paths(str(args.get("code") or ""))
    return ()


def _fp_key(fp: tuple) -> str:
    """Contract: Preconditions: fp 为非空元组；Postconditions: 返回可作
    SQLite 主键的稳定字符串（'\x1f' 连接，避免与路径分隔符冲突）。"""
    return "\x1f".join(fp)


def _note_blocked(tool_name: str, args: dict) -> None:
    """Contract: Preconditions: 已发生一次 status=blocked 的调用；
    Postconditions: 写意图指纹计数持久化入库（只读命令豁免不计），窗口
    过期行重置、超上限驱逐最旧，计数达阈值时置进程内升级标记；任何库
    异常降级为 ERROR 日志不中断。"""
    fp = _intent_fingerprint(tool_name, args)
    if not fp:
        return
    if tool_name == "terminal" and _command_is_readonly(
            str(args.get("command") or "")):
        logger.info(
            "block_escalation: 只读命令被拦（指纹=%s），豁免升级计数",
            _fp_key(fp))
        return
    now = time.time()
    key = _fp_key(fp)
    try:
        with _db_lock:
            conn = _get_conn()
            try:
                _ensure_schema(conn)
                conn.execute("DELETE FROM block_streaks WHERE last_ts < ?", (now - _WINDOW_SECS,))
                row = conn.execute(
                    "SELECT count, tools FROM block_streaks WHERE fingerprint = ?",
                    (key,)).fetchone()
                if row is None:
                    while conn.execute("SELECT COUNT(*) FROM block_streaks").fetchone()[0] >= _STATE_CAP:
                        oldest = conn.execute(
                            "SELECT fingerprint FROM block_streaks "
                            "ORDER BY last_ts LIMIT 1").fetchone()
                        if oldest is None:
                            break
                        conn.execute("DELETE FROM block_streaks WHERE fingerprint = ?",
                                     (oldest[0],))
                    conn.execute(
                        "INSERT INTO block_streaks (fingerprint, count, last_ts, tools, sample) "
                        "VALUES (?, 1, ?, ?, ?)",
                        (key, now, tool_name, str(args)[:200]))
                    count, tools = 1, {tool_name}
                else:
                    count = row["count"] + 1
                    tools = set(filter(None, row["tools"].split(","))) | {tool_name}
                    conn.execute(
                        "UPDATE block_streaks SET count = ?, last_ts = ?, tools = ? "
                        "WHERE fingerprint = ?",
                        (count, now, ",".join(sorted(tools)), key))
                conn.commit()
            finally:
                conn.close()
    except Exception as err:
        logger.error("block_escalation: 持久化失败（不中断拦截链）: %s", err)
        return
    if count >= _STREAK_LIMIT:
        _escalated.add(fp)
        logger.warning(
            "block_escalation: 同一意图第 %d 次被拦（通道=%s，跨会话持久计数），升级终止",
            count, sorted(tools))


def _overlapping_keys(conn: sqlite3.Connection, fp: tuple) -> list:
    """Contract: Preconditions: conn 为已建 schema 的连接，fp 为非空指纹；
    Postconditions: 返回库内与 fp 存在元素交集的指纹键列表。"""
    sep = "\x1f"
    keys = []
    for row in conn.execute(
            "SELECT fingerprint FROM block_streaks").fetchall():
        if any(e and e in fp for e in row[0].split(sep)):
            keys.append(row[0])
    return keys


def _clear_streak(tool_name: str, args: dict) -> None:
    """Contract: Preconditions: 一次非 blocked 状态的工具调用已完成；Postconditions:
    该调用指纹（及库内任一元素与成功路径相同的行为——跨工具形态）的 streak
    清零并撤销升级标记（意图已成功落地，惩罚作废）；库异常降级为 ERROR 日志
    不中断。"""
    fp = _intent_fingerprint(tool_name, args)
    if not fp:
        return
    try:
        with _db_lock:
            conn = _get_conn()
            try:
                _ensure_schema(conn)
                for key in _overlapping_keys(conn, fp):
                    conn.execute(
                        "DELETE FROM block_streaks WHERE fingerprint = ?",
                        (key,))
                conn.commit()
            finally:
                conn.close()
    except Exception as err:
        logger.error("block_escalation: 清账失败（不中断）: %s", err)
    for stale in [f for f in list(_escalated) if any(
            e and e in f for e in fp)]:
        _escalated.discard(stale)


def _on_post_tool_call(tool_name: str = "", status: str = "", error_type: str = "",
                       args: dict | None = None, **kwargs) -> None:
    """Contract: Preconditions: 核心以 status/error_type 复播工具结果；
    Postconditions: status=blocked 时记录意图指纹；status=ok 时同指纹
    清账（成功落地=streak 作废，防误判残留）；纯观察无返回值。"""
    if status == "blocked":
        _note_blocked(tool_name, args or {})
    elif status == "ok":
        _clear_streak(tool_name, args or {})


def register(ctx):
    """Contract: Preconditions: ctx 暴露 register_hook(hook_name, callback)；
    Postconditions: post_tool_call/transform_llm_output 两钩子注册成功并落日志。"""
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    logger.info("block-escalation plugin registered (post_tool_call + transform_llm_output, sqlite-persisted)")


def _on_transform_llm_output(response_text: str = "", **kwargs) -> str:
    """Contract: Preconditions: response_text 为本回合 LLM 输出（核心键名）；
    Postconditions: 存在已升级指纹（说明被拦后仍在尝试或继续输出）时追加
    带用户可见终止提示（ESCALATION_MARKER 前缀，cron scheduler 据此按
    失败记账），否则原样返回。"""
    output = response_text or ""
    if not _escalated:
        return output
    logger.warning("block_escalation: 升级指纹仍活跃，追加用户可见终止提示")
    _escalated.clear()  # 单次提示，避免重复堆叠
    return output + (

        "\n\n" + ESCALATION_MARKER + " 同一写操作意图已连续 2 次被拦且发生通道切换（跨会话持久计数）。"
        "本轮必须立即停止第 3 次尝试：①逐字重读两次拦截信息原文②按拦截信息指明的合法通道"
        "补齐证据原路重试；若指引不可行，停下向用户呈报拦截原文与已尝试通道清单，等待人工"
        "裁决。禁止再换任何新通道、禁止换会话规避。")
