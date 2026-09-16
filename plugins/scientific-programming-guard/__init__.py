"""科学编程守卫 v2——skill:scientific-programming 的机检子集。

v2 升级(对应四层十五律):
  防错层(所有 .py 生效):
    R3 圈复杂度预算——单函数 cc>10 或行数>50 → 拆分(律 3 复杂度控制)
    R4 新函数缺类型注解——def 参数/返回值无注解(律 4 静态分析前置)
  运行层(仅网络/DB 交互代码生效, 判定=结构化导入集合):
    R1 finally 中 close 在哨兵 put 之前 → 消费方饿死(律 11 并发正确性)
    R2 finally 清理未包 try/except → 掩盖原始异常(律 11)
  测试纪律层(仅 tests/ 下 .py 生效):
    R6 assert 字面量期望值缺同线「# 期望:」推导注释 → 疑似凑绿灯/瞎猜
       (根因: 本会话两起实测——true_key 漏传/endswith(python) 不匹配
        python3, 均为期望值或参数未从真实源码/文档推导)

判定全部 AST 级零正则。豁免: test_ 前缀/单行函数。
测试: tests/plugins/test_scientific_programming_guard.py
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

_MAX_CC = 10
_MAX_LINES = 50

# R6 期望值推导注释: 断言比较的字面量必须带同线 "# 期望:" 注释
_EXPECTED_MARK = "# 期望:"


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


def _is_exempt(func: ast.FunctionDef) -> bool:
    """豁免判定——测试函数/魔法方法不检查。

    Contract:
      Preconditions: func 是 FunctionDef 节点
      Postconditions: 名字以 test_/__ 开头返回 True
    """
    return func.name.startswith(("test_", "__"))


def _cyclomatic_complexity(func: ast.FunctionDef) -> int:
    """圈复杂度计算(简化 McCabe)——1 + 分支/异常/布尔算子数。

    Contract:
      Preconditions: func 是 FunctionDef
      Postconditions: 返回 ≥1 的整数复杂度值
    """
    cc = 1
    for node in ast.walk(func):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor,
                             ast.ExceptHandler, ast.With, ast.AsyncWith)):
            cc += 1
        elif isinstance(node, ast.BoolOp):
            cc += len(node.values) - 1
        elif isinstance(node, (ast.Assert, ast.comprehension)):
            cc += 1
        elif isinstance(node, ast.Match):
            cc += len(node.cases)
    return cc


def _check_complexity_budget(tree: ast.AST) -> list:
    """R3: 单函数圈复杂度>10 或行数>50 → 违规(律 3)。

    Contract:
      Preconditions: tree 是合法 AST
      Postconditions: 返回违规消息列表(空=合规)
    """
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if _is_exempt(node):
            continue
        cc = _cyclomatic_complexity(node)
        lines = (getattr(node, "end_lineno", 0) or node.lineno) - node.lineno + 1
        if cc > _MAX_CC or lines > _MAX_LINES:
            issues.append(
                f"line {node.lineno}: {node.name}() 复杂度超预算"
                f"(cc={cc}>{_MAX_CC} 或 {lines}行>{_MAX_LINES})"
                f"——拆分为更小的单一职责函数(律 3)。"
            )
    return issues


def _check_type_annotations(tree: ast.AST) -> list:
    """R4: def 参数与返回值缺类型注解 → 违规(律 4)。

    仅查顶层 def(patch 场景看不到嵌套函数全貌, 查可见的全部)。
    self/cls 与 *args/**kwargs 豁免。

    Contract:
      Preconditions: tree 是合法 AST
      Postconditions: 返回违规消息列表(空=合规)
    """
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if _is_exempt(node):
            continue
        missing = []
        args = node.args
        positional = list(args.posonlyargs) + list(args.args)
        if positional and positional[0].arg in ("self", "cls"):
            positional = positional[1:]
        for a in positional:
            if a.annotation is None:
                missing.append(a.arg)
        if args.vararg and args.vararg.annotation is None:
            missing.append("*" + args.vararg.arg)
        if node.returns is None:
            missing.append("-> 返回值")
        if missing:
            issues.append(
                f"line {node.lineno}: {node.name}() 缺类型注解: "
                f"{', '.join(missing)}(律 4 静态分析前置)。"
            )
    return issues


def _check_l4_sentinel_before_close(tree: ast.AST) -> list:
    """R1: finally 中 close 类调用不得排在哨兵 put 之前(律 11)。

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
                f"——死资源 close 卡死会饿死消费循环(律 11 并发正确性)。"
                f"调整顺序: 哨兵先行, close 包 try/except 后置。"
            )
    return issues


def _check_l3_guarded_cleanup(tree: ast.AST) -> list:
    """R2: finally 中清理调用必须被 try/except 包裹(律 11)。

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
                f"——死资源上会抛新异常掩盖原始异常(律 11)。"
                f"包 try/except 记 warning, 不得上抛。"
            )
    return issues


def _check_expected_annotation(tree: ast.AST, lines: list) -> list:
    """R6: tests/ 文件中 assert 比较/成员断言的字面量期望值缺同线推导注释 → 违规。

    只查 assert 语句：二元比较(==/!=/in/not in)或容器成员断言，且期望侧
    是字面量(数字/字符串/布尔/None/列表/字典)。断言所在物理行须含
    "# 期望:" 注释说明独立推导依据；无注释=疑似凑绿灯/拍脑袋。

    Contract:
      Preconditions: tree 是合法 AST; lines 是源码物理行列表
      Postconditions: 返回违规消息列表(空=合规), 每条含行号
    """
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        literal = False
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            comp = test.comparators[0]
            # 「is not None」健全性检查豁免——断言的是存在性而非期望值内容,
            # 强制注释只会制造噪音
            if (isinstance(test.ops[0], (ast.IsNot, ast.Is))
                    and isinstance(comp, ast.Constant)
                    and comp.value is None):
                literal = not isinstance(test.ops[0], ast.IsNot)
            else:
                literal = isinstance(comp, (
                    ast.Constant, ast.List, ast.Dict, ast.Tuple))
        elif isinstance(test, ast.Constant):
            literal = test.value in (True, False, None)
        if not literal:
            continue
        lineno = node.lineno
        if lineno - 1 < len(lines) and _EXPECTED_MARK in lines[lineno - 1]:
            continue
        issues.append(
            f"line {lineno}: assert 字面量期望值缺同线「{_EXPECTED_MARK}」推导注释"
            f"——期望值必须独立推导并注明依据，禁先写代码再凑绿灯(测试纪律)。"
        )
    return issues


def on_pre_tool_call(**kwargs):
    """pre_tool_call 入口——写 .py 前机检科学编程纪律。

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
        if not path.endswith(".py"):
            return {}
        tree = ast.parse(text)
        # 防错层规则——所有 .py 生效
        issues = _check_complexity_budget(tree)
        issues += _check_type_annotations(tree)
        # 运行层规则——仅网络/DB 交互代码生效
        if _is_network_code(text):
            issues += _check_l4_sentinel_before_close(tree)
            issues += _check_l3_guarded_cleanup(tree)
        # 测试纪律层——仅 tests/ 下文件生效
        if "/tests/" in path or path.startswith("tests/"):
            issues += _check_expected_annotation(tree, text.splitlines())
        if issues:
            return {
                "action": "block",
                "message": (
                    "[scientific-programming-guard] 科学编程纪律违规:\n"
                    + "\n".join(issues)
                    + "\n修复后再提交。规则全文: skill:scientific-programming"
                ),
            }
        return {}
    except Exception:
        logger.warning("scientific-programming-guard 检查失败(放行)", exc_info=True)
        return {}


def register(ctx):
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
