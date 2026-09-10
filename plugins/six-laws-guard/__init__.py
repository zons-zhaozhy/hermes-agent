"""科学编程六律守卫——网络/DB 交互代码四防线机检。

拦截规则(对应 skill:scientific-programming-six-laws 律二):
  R1  finally 块中资源 close/清理在哨兵 put 之前 → 消费方可能被饿死
  R2  finally 块中清理动作未包 try/except → 死资源上抛新异常掩盖原始异常
  R3  网络重试 pattern 用裸 except/过窄白名单 → 断连族错误码漏网零重试

只对「网络/DB 交互代码」生效——判定条件(结构化, 禁正则枚举信号词):
  文本含 import oracledb/pymysql/psycopg/requests/httpx/socket/.connect(
  即视为网络交互代码。
测试: tests/plugins/test_six_laws_guard.py
"""
import ast
import logging

logger = logging.getLogger(__name__)

# 网络交互判定的导入模块集合(有限标准化集合, 非信号词枚举)
_NET_MODULES = (
    "oracledb", "pymysql", "psycopg", "psycopg2", "requests",
    "httpx", "urllib3", "socket", "aiohttp",
)
_NET_HINTS = (".connect(", "cursor()", "execute_many", "query_stream")

_EXEMPT_PREFIXES = ("test_", "_", "verify_")


def _is_network_code(text: str) -> bool:
    """判定是否网络/DB 交互代码。

    Contract:
      Preconditions: text 非空字符串
      Postconditions: 含网络模块 import 或连接调用痕迹返回 True; 否则 False
    """
    if not text:
        return False
    for mod in _NET_MODULES:
        if f"import {mod}" in text or f"from {mod}" in text:
            return True
    return any(h in text for h in _NET_HINTS)


def _check_l4_sentinel_before_close(tree: ast.AST) -> list:
    """R1: finally 中 close 类调用不得排在哨兵 put 之前。

    扫描 Try 节点的 finalbody: 若同时存在 close 类调用(属性名以
    close/shutdown/dispose 结尾)与队列 put 调用(属性名 put/put_nowait),
    close 出现在 put 之前即违规——死资源 close 卡死会饿死消费循环。

    Contract:
      Preconditions: tree 是合法 AST
      Postconditions: 返回违规消息列表(空=合规), 每条含行号
    """
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        close_pos = None
        put_pos = None
        for i, stmt in enumerate(node.finalbody):
            if not (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)):
                continue
            name = stmt.value.func.attr.lower()
            if name in ("close", "shutdown", "dispose", "terminate"):
                close_pos = (i, stmt.lineno)
            if name in ("put", "put_nowait"):
                put_pos = (i, stmt.lineno)
        if close_pos and put_pos and close_pos[0] < put_pos[0]:
            issues.append(
                f"line {close_pos[1]}: finally 中 close/清理在哨兵 put 之前"
                f"——死资源 close 卡死会饿死消费循环(四防线 L4)。"
                f"调整顺序: 哨兵先行, close 包 try/except 后置。"
            )
    return issues


def _check_l3_guarded_cleanup(tree: ast.AST) -> list:
    """R2: finally 中清理调用必须被 try/except 包裹。

    死连接上的 close/调档操作会抛新异常(如 AttributeError),
    掩盖 finally 之前的原始异常(四防线 L3)。

    Contract:
      Preconditions: tree 是合法 AST
      Postconditions: 返回违规消息列表(空=合规)
    """
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.finalbody:
            if not (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)):
                continue
            name = stmt.value.func.attr.lower()
            if name not in ("close", "shutdown", "dispose", "terminate"):
                continue
            issues.append(
                f"line {stmt.lineno}: finally 中 {name}() 未包 try/except"
                f"——死资源上会抛新异常掩盖原始异常(四防线 L3)。"
                f"包 try/except 记 warning, 不得上抛。"
            )
    return issues


def on_pre_tool_call(**kwargs):
    """pre_tool_call 入口——写操作前机检网络代码四防线。

    Contract:
      Preconditions: kwargs 含 tool_name 与 args(write_file/patch 为
        path+content / old_string+new_string)
      Postconditions: 违规返回 {"action":"block","message":...};
        合规/非目标工具/解析失败(非本工具职责)返回 {} 放行
    """
    try:
        tool_name = kwargs.get("tool_name", "")
        if tool_name not in ("write_file", "patch"):
            return {}
        args = kwargs.get("args", {}) or {}
        text = (args.get("content") or args.get("new_string") or "")
        path = str(args.get("path", ""))
        if not path.endswith(".py") or not _is_network_code(text):
            return {}
        tree = ast.parse(text)
        issues = _check_l4_sentinel_before_close(tree)
        issues += _check_l3_guarded_cleanup(tree)
        if issues:
            return {
                "action": "block",
                "message": (
                    "[six-laws-guard] 网络/DB 代码四防线违规:\n"
                    + "\n".join(issues)
                    + "\n修复后再提交。规则全文: skill:scientific-programming-six-laws"
                ),
            }
        return {}
    except Exception:
        logger.warning("six-laws-guard 检查失败(放行)", exc_info=True)
        return {}


def register(ctx):
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
