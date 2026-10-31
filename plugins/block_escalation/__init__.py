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

import json
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
    "rev-parse", "merge-base", "for-each-ref", "name-rev", "verify",
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
_escalated: dict[str, set] = {}  # 本进程已升级指纹，键控 session_id（提示单次消费，禁跨会话污染）


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
    block_streaks 表与指纹索引存在（幂等）；kind 列（'blocked'=护栏拦截/
    'error'=执行失败）缺失时经 PRAGMA table_info 判定后追加，追加同时清空
    全部历史行——streak 窗口仅 1800s 短命，旧行 kind 不可判定，保留会让
    执行失败行继续保有升级权（10-31 deputy-patrol 三次误杀链）。跨进程
    并发迁移撞 duplicate column 时由调用方 except 记日志，下次调用即正常。"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS block_streaks (
            fingerprint TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0,
            last_ts REAL NOT NULL,
            tools TEXT NOT NULL DEFAULT '',
            sample TEXT,
            kind TEXT NOT NULL DEFAULT 'blocked'
        );
        CREATE INDEX IF NOT EXISTS idx_bs_last_ts ON block_streaks(last_ts);
    """)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(block_streaks)")}
    if "kind" not in cols:
        conn.execute("ALTER TABLE block_streaks ADD COLUMN kind TEXT NOT NULL DEFAULT 'blocked'")
        conn.execute("DELETE FROM block_streaks")
        _escalated.clear()
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
    if tool_name in ("read_file", "search_files"):
        # 读工具指纹=path（或 pattern+path）——error/blocked 循环的观测键。
        # 曾恒返回 ()，读循环从不参与计数，失败观测层对读通道结构性失明。
        target = str(args.get("path") or "")
        pattern = str(args.get("pattern") or "")
        return (f"{tool_name}:{target or pattern}",)
    if tool_name == "execute_code":
        return _extract_paths(str(args.get("code") or ""))
    return ()


def _fp_key(fp: tuple) -> str:
    """Contract: Preconditions: fp 为非空元组；Postconditions: 返回可作
    SQLite 主键的稳定字符串（'\x1f' 连接，避免与路径分隔符冲突）。"""
    return "\x1f".join(fp)


def _note_blocked(tool_name: str, args: dict, session_id: str = "",
                  kind: str = "blocked") -> None:
    """Contract: Preconditions: 一次「未成功落地」的工具调用已发生，kind
    区分语义（'blocked'=护栏拦截/'error'=执行失败）；
    Postconditions: 只读命令豁免不计；写意图指纹计数经 _bump_streak 持久化，
    达阈值时按 kind 分流（blocked 升级终止/error 仅观测告警）。"""
    fp = _intent_fingerprint(tool_name, args)
    if not fp:
        return
    if tool_name == "terminal" and _command_is_readonly(
            str(args.get("command") or "")):
        logger.info(
            "block_escalation: 只读命令被拦（指纹=%s），豁免升级计数",
            _fp_key(fp))
        return
    bumped = _bump_streak(fp, tool_name, args, kind)
    if bumped is None:
        return
    _streak_threshold_action(bumped[0], bumped[1], bumped[2], session_id, fp)


def _bump_streak(fp: tuple, tool_name: str, args: dict, kind: str) -> "tuple[int, set, str] | None":
    """Contract: Preconditions: fp 为非空意图指纹，kind 为本次语义；
    Postconditions: 指数计数持久化入库（窗口过期行重置、超上限驱逐最旧，
    INSERT/UPDATE 显式写 kind；同指纹已有 blocked 史时行 kind 不因 error
    降级——拦截挣扎不弱化升级权）；任何库异常降级为 ERROR 日志并返回
    None（不中断拦截链）。"""
    now = time.time()
    key = _fp_key(fp)
    try:
        with _db_lock:
            conn = _get_conn()
            try:
                _ensure_schema(conn)
                conn.execute("DELETE FROM block_streaks WHERE last_ts < ?", (now - _WINDOW_SECS,))
                row = conn.execute(
                    "SELECT count, tools, kind FROM block_streaks WHERE fingerprint = ?",
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
                        "INSERT INTO block_streaks (fingerprint, count, last_ts, tools, sample, kind) "
                        "VALUES (?, 1, ?, ?, ?, ?)",
                        (key, now, tool_name, str(args)[:200], kind))
                    count, tools, kind_now = 1, {tool_name}, kind
                else:
                    count = row["count"] + 1
                    tools = set(filter(None, row["tools"].split(","))) | {tool_name}
                    kind_now = "blocked" if row["kind"] == "blocked" or kind == "blocked" else "error"
                    conn.execute(
                        "UPDATE block_streaks SET count = ?, last_ts = ?, tools = ?, kind = ? "
                        "WHERE fingerprint = ?",
                        (count, now, ",".join(sorted(tools)), kind_now, key))
                conn.commit()
            finally:
                conn.close()
    except Exception as err:
        logger.error("block_escalation: 持久化失败（不中断拦截链）: %s", err)
        return None
    return count, tools, kind_now


def _streak_threshold_action(count: int, tools: set, kind: str, session_id: str,
                             fp: tuple) -> None:
    """Contract: Preconditions: count/kind 为刚落库的 streak 计数与语义；
    Postconditions: kind=blocked 达阈值时置进程内升级标记并 WARNING（升级
    终止）；kind=error 达阈值时仅 WARNING 观测（执行失败循环——周期任务
    跨班合法重复与同会话死循环不可零成本区分，终止权只保留给护栏拦截，
    10-31 deputy-patrol 三次误杀链）。已知语义取舍：同指纹先累积 error 计
    数后首次 blocked 即达阈值升级（count 不分车道）——保守方向可接受，
    误升级代价仅一次终止提示文本且随后 ok 清账，反向（blocked 后 error
    保留升级权）为设计意图。"""
    if count < _STREAK_LIMIT:
        return
    if kind == "blocked":
        _escalated.setdefault(session_id, set()).add(fp)
        logger.warning(
            "block_escalation: 同一意图第 %d 次被拦（通道=%s，跨会话持久计数），升级终止",
            count, sorted(tools))
    else:
        logger.warning(
            "block_escalation: 同一意图第 %d 次执行失败（通道=%s，跨会话持久计数），"
            "观测告警不终止（执行失败≠护栏拦截，杀班权仅限拦截循环）",
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
    for sid in list(_escalated):
        _escalated[sid] = {f for f in _escalated[sid] if not any(
            e and e in f for e in fp)}
        if not _escalated[sid]:
            del _escalated[sid]


def _is_semantic_block(status: str, result: "str | dict | None") -> bool:
    """Contract: Preconditions: status 为 hook 复播状态，result 为工具原始
    结果（str/None/dict）；Postconditions: 返回该调用是否属于护栏拦截语义
    ——status=blocked（guardrail/pre_tool_call 插件块显式复播，
    tool_executor.py emit_post），或 result 包体 status=blocked（terminal
    工具体护栏拒绝：_blocked_json 含 error 键，被 model_tools
    _tool_result_observer_fields 复播为 error——按 status 分流会漏判真拦截）。
    result 非 JSON/非 dict 一律 False（执行失败归 error 车道）。"""
    if status == "blocked":
        return True
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
        return isinstance(parsed, dict) and parsed.get("status") == "blocked"
    except Exception as err:
        logger.warning("block_escalation: result 包体解析失败，按执行失败车道处理: %s", err)
        return False


def _on_post_tool_call(tool_name: str = "", status: str = "", error_type: str = "",
                       args: dict | None = None, session_id: str = "", **kwargs) -> None:
    """Contract: Preconditions: 核心以 status/error_type 复播工具结果；
    Postconditions: 护栏拦截语义（status=blocked 或包体 status=blocked）按
    blocked 车道计数（达阈值升级终止）；纯执行失败按 error 车道计数（达
    阈值仅观测告警不终止——周期任务跨班合法重复与死循环不可零成本区分，
    终止权只保留给拦截循环，10-31 deputy-patrol 三次误杀链）；status=ok
    时同指纹清账（成功落地=streak 作废，防误判残留）；纯观察无返回值。"""
    if status in ("blocked", "error"):
        kind = "blocked" if _is_semantic_block(status, kwargs.get("result")) else "error"
        _note_blocked(tool_name, args or {}, session_id=session_id, kind=kind)
    elif status == "ok":
        _clear_streak(tool_name, args or {})


def register(ctx):
    """Contract: Preconditions: ctx 暴露 register_hook(hook_name, callback)；
    Postconditions: post_tool_call/transform_llm_output 两钩子注册成功并落日志。"""
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    logger.info("block-escalation plugin registered (post_tool_call + transform_llm_output, sqlite-persisted)")


def _on_transform_llm_output(response_text: str = "", **kwargs) -> "str | None":
    """Contract: Preconditions: response_text 为本回合 LLM 输出（核心键名）；
    Postconditions: 存在已升级指纹（说明被拦后仍在尝试或继续输出）时追加
    带用户可见终止提示（ESCALATION_MARKER 前缀，cron scheduler 据此按
    失败记账），否则返回 None（核心契约：first non-None wins——不改写
    必须返回 None，返回原文会抢占槽位短路其后全部 transform 插件）。"""
    output = response_text or ""
    sid = str(kwargs.get("session_id") or "")
    if not _escalated.get(sid):
        return None
    logger.warning("block_escalation: 升级指纹仍活跃，追加用户可见终止提示")
    _escalated.pop(sid, None)  # 单次提示，避免重复堆叠
    return output + (

        "\n\n" + ESCALATION_MARKER + " 同一写操作意图已连续 2 次被拦且发生通道切换（跨会话持久计数）。"
        "本轮必须立即停止第 3 次尝试：①逐字重读两次拦截信息原文②按拦截信息指明的合法通道"
        "补齐证据原路重试；若指引不可行，停下向用户呈报拦截原文与已尝试通道清单，等待人工"
        "裁决。禁止再换任何新通道、禁止换会话规避。")
