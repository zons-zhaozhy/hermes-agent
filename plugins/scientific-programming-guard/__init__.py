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
    R5a import 仓库内本地模块但本会话从未 read_file 过 → 参数瞎猜防线
    R5b 调用本地模块函数传了签名中不存在的关键字参数 → 瞎编参数防线
       (Case: llm_judge_bool(true_key="review")——签名无该键)

判定全部 AST 级零正则。豁免: test_ 前缀/单行函数。
测试: tests/plugins/test_scientific_programming_guard.py
"""
import ast
import logging
from pathlib import Path

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

# R5 参数瞎猜拦截: 仓库根(含 pyproject.toml/setup.py)
_REPO_MARKERS = ("pyproject.toml", "setup.py")


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


def _repo_root(start: Path) -> Path | None:
    """向上找仓库根(含 pyproject.toml/setup.py 的最近祖先目录)。

    Contract:
      Preconditions: start 是存在的路径
      Postconditions: 找到返回该目录 Path, 到根目录仍无标记返回 None
    """
    cur = start.resolve()
    for candidate in (cur, *cur.parents):
        if any((candidate / m).exists() for m in _REPO_MARKERS):
            return candidate
    return None


def _resolve_local_module(mod_name: str, test_path: Path) -> Path | None:
    """把 import 的模块名解析为仓库内源文件路径; 第三方/标准库返回 None。

    Contract:
      Preconditions: mod_name 是点分模块名; test_path 是测试文件绝对路径
      Postconditions: 仓库内存在对应 .py 返回其 Path, 否则 None
    """
    root = _repo_root(test_path)
    if root is None:
        return None
    rel = mod_name.replace(".", "/")
    for cand in (root / rel, root / (rel + ".py")):
        if cand.suffix == ".py" and cand.exists():
            return cand
        if (cand / "__init__.py").exists():
            return cand / "__init__.py"
    return None


def _top_level_defs(src: str) -> tuple[set, dict]:
    """模块顶层可见符号: 定义名集合 + {函数名: 形参名集合}。

    Contract:
      Preconditions: src 是 Python 源码文本(可含仅类型的定义)
      Postconditions: 返回 (名字集合, 函数形参表); 解析失败返回空集
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), {}
    names: set = set()
    params: dict = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                p = {x.arg for x in (list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs))}
                if a.vararg:
                    p.add(a.vararg.arg)
                if a.kwarg:
                    p.add("**" + a.kwarg.arg)
                params[node.name] = p
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names, params


def _check_unverified_local_imports(tree: ast.AST, path: str) -> list:
    """R5a: tests/ 文件 import 仓库内本地模块, 但该模块在 本会话从未 read_file 过 → 违规。

    只拦 import 语句(静态可判), 不拦调用点。import 目标解析不到仓库内
    .py 的(第三方/标准库)放行。

    Contract:
      Preconditions: tree 是 tests/ 文件的 AST; path 是其绝对路径
      Postconditions: 返回违规消息列表(空=合规); 模块文件读取失败跳过该项
    """
    issues = []
    test_path = Path(path).resolve()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            targets = [node.module]
        else:
            continue
        for mod in targets:
            src_file = _resolve_local_module(mod, test_path)
            if src_file is None:
                continue
            if not _session_has_read(src_file):
                issues.append(
                    f"line {node.lineno}: import {mod} 引用本地模块 {src_file.name}, "
                    f"但本会话从未 read_file 过它——先读源码再引用其符号, "
                    f"禁凭记忆猜参数/属性(参数瞎猜防线)。"
                )
    return issues


def _check_guessed_kwargs(tree: ast.AST, path: str) -> list:
    """R5b: 调用本地模块函数时传了签名中不存在的关键字参数 → 违规。

    Case: llm_judge_bool(true_key="review")——签名无该键, 静默 fail-open。
    跨模块调用按 import 源解析目标文件; 同文件调用按本文本解析。

    Contract:
      Preconditions: tree 是 tests/ 文件的 AST; path 是其绝对路径
      Postconditions: 返回违规消息列表(空=合规); 解析不出目标文件跳过
    """
    issues = []
    test_path = Path(path).resolve()
    src_text = ""
    try:
        src_text = Path(path).read_text(encoding="utf-8")
    except OSError:
        src_text = ""
    own_params = _top_level_defs(src_text)[1]

    mod_targets: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mod_targets[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            src_file = _resolve_local_module(node.module, test_path)
            if src_file is not None:
                try:
                    mod_targets_local = _top_level_defs(src_file.read_text(encoding="utf-8"))[1]
                except OSError:
                    continue
                for a in node.names:
                    mod_targets[a.asname or a.name] = mod_targets_local

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.keywords):
            continue
        func = node.func
        fname = None
        param_source = None
        if isinstance(func, ast.Name):
            fname = func.id
            # from X import f 后的 f(...): 目标签名来自被导入模块而非本文件
            bound = mod_targets.get(fname)
            param_source = bound if isinstance(bound, dict) else own_params
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            fname = func.attr
            bound = mod_targets.get(func.value.id)
            param_source = bound if isinstance(bound, dict) else None
        if fname is None or param_source is None:
            continue
        known = param_source.get(fname)
        if known is None:
            continue
        # **kwargs 可变参函数任何键名都合法, 豁免
        if any(a.startswith("**") for a in known):
            continue
        bad = [k.arg for k in node.keywords if k.arg not in known and k.arg != "task_id"]
        if bad:
            issues.append(
                f"line {node.lineno}: {fname}(...) 传了签名不存在的关键字参数 {bad} "
                f"——参数必须从目标模块源码推导, 禁猜(参数瞎猜防线)。"
            )
    return issues


def _session_has_read(src_file: Path) -> bool:
    """本会话(进程内所有 task)是否读过该文件。

    读信号来自 tools.file_tools_read_tracking 的进程级 _read_tracker
    (read_history 按任务记录 read_file 触过的路径)。

    Contract:
      Preconditions: src_file 是绝对路径
      Postconditions: 任一 task 的读记录命中该路径(原串或归一化) → True;
        否则 False; 记账模块不可导入 → True(探测失败不拦车, fail-open)
    """
    try:
        from tools.file_tools_read_tracking import _read_tracker
    except Exception:
        logger.warning("R5 read-ledger 不可达, 参数防线探测 fail-open", exc_info=True)
        return True
    target = str(src_file)
    target_norm = str(src_file.resolve())
    for task_data in _read_tracker.values():
        history = task_data.get("read_history")
        if not history:
            continue
        for entry in history:
            entry_str = str(entry)
            if entry_str == target or entry_str == target_norm:
                return True
            try:
                if str(Path(entry).resolve()) == target_norm:
                    return True
            except (OSError, ValueError):
                continue
    return False


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
            issues += _check_unverified_local_imports(tree, path)
            issues += _check_guessed_kwargs(tree, path)
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
