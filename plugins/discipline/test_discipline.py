"""test_discipline —— 测试执行纪律动作化（.hermes-rules.md 测试纪律第 1 条）。

「一律 scripts/run_tests.sh，禁裸 pytest」从文本规矩下沉为动作拦截：
裸 pytest/裸 python -m pytest 的 terminal 命令直接 block，正门=测试套件
包装器（scripts/run_tests.sh）——它强制 CI 环境对齐（清凭据/TZ=UTC/
每文件子进程隔离），裸 pytest 在 16 核开发机上跑出的结果与 CI 漂移
（历史多次"本地绿 CI 红"事故的根因）。

仓库感知（2026-09-19 误报根修）：包装器纪律只适用于真有该包装器的
仓库（hermes-agent）。无包装器仓库（如 ontox，CI 即裸 python -m pytest）
被拦后正门不存在=堵死合法通道。判定升级：从命令内 cd 目标/workdir 沿
目录树向上探测 scripts/run_tests.sh，存在才拦，不存在放行（该仓库的
规范执行方式就是裸 pytest，与 CI 一致）。

判定规则（shlex 令牌级，无正则）：
  - 令牌含 "pytest" 或 ("python" + "-m" + "pytest" 序列) 即视为 pytest 调用
  - 命令以 scripts/run_tests.sh（或 ./scripts/run_tests.sh）为前缀 → 放行
  - 纯探针（pytest --version / --help / --collect-only）→ 放行
  - run_tests_parallel.py 直接调用 → 放行（同一包装器体系的内层）
  - 目标仓库（cd/workdir 向上）无 scripts/run_tests.sh → 放行（仓库感知）

只拦 terminal 工具；不拦只读命令；fail-open（解析失败放行）。

Contract:
  Preconditions: plugin system 提供 pre_tool_call 钩子（tool_name/args.command）。
  Postconditions: 命中裸 pytest 且无豁免时返回 {"action":"block","message":...}；
                  其余返回 None。绝不 raise。
  Invariants: 永不拦 scripts/run_tests.sh 自身；永不拦 --version/--help/--collect-only。
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WRAPPER_PREFIXES = ("scripts/run_tests.sh", "./scripts/run_tests.sh")
_PARALLEL_RUNNER = "run_tests_parallel.py"
# 纯探针旗标：不执行测试，只查询，放行
_PROBE_FLAGS = {"--version", "-V", "--help", "-h", "--collect-only", "--co"}


def _repo_has_wrapper(start_dir: str) -> bool:
    """目录树向上探测 scripts/run_tests.sh 是否存在（仓库感知核心）。

    Contract:
      Preconditions: start_dir 为命令中 cd/workdir 指示的目录（可能不存在）
      Postconditions: 返回 True/False；任何 IO 异常返回 True（fail-closed，
        即探测失败时保守拦截——包装器仓库的纪律不因探测故障而松动）
    """
    try:
        p = Path(start_dir).expanduser().resolve()
        for parent in [p, *p.parents]:
            if (parent / "scripts" / "run_tests.sh").is_file():
                return True
        return False
    except OSError:
        return True


def _target_dir_from_command(tokens: List[str]) -> str:
    """从命令令牌提取目标目录：最后一个 cd 参数；无 cd 时返回进程 cwd。

    pytest 在 cd 之后执行，故 cd 目标才是判定基准。
    """
    import os
    cd_target = None
    for i, tok in enumerate(tokens):
        if tok == "cd" and i + 1 < len(tokens) and tokens[i + 1] not in ("&&", ";"):
            cd_target = tokens[i + 1]
    if cd_target:
        return cd_target
    return os.getcwd()


def _is_module_invocation(tokens: List[str]) -> bool:
    """命令令牌中是否含 python -m pytest 序列（模块方式调 pytest）。

    解释器令牌按前缀判（python/python3/python3.11 等都算），
    且与 -m pytest 相邻才算显式模块调用。
    """
    for i in range(len(tokens) - 2):
        if (tokens[i].startswith("python") and tokens[i + 1] == "-m"
                and tokens[i + 2] == "pytest"):
            return True
    return False


def _is_bare_pytest(command: str) -> bool:
    """令牌级判定：命令是否为裸 pytest 调用（非包装器、非纯探针）。

    Contract:
      Postconditions: 返回 True/False；shlex 解析失败返回 False（fail-open）。
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    # 包装器体系一律放行
    if tokens[0] in _WRAPPER_PREFIXES or _PARALLEL_RUNNER in tokens:
        return False
    # 纯探针放行
    if _PROBE_FLAGS & set(tokens):
        return False
    is_pytest = tokens[0].endswith("pytest") or _is_module_invocation(tokens)
    if not is_pytest:
        return False
    # 仓库感知：目标仓库无包装器 → 该仓库规范即裸 pytest，放行
    if not _repo_has_wrapper(_target_dir_from_command(tokens)):
        return False
    return True


def on_pre_tool_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """裸 pytest terminal 命令 → block，指明 run_tests.sh 正门。fail-open。"""
    try:
        if str(kwargs.get("tool_name", "")) != "terminal":
            return None
        command = str((kwargs.get("args") or {}).get("command") or "")
        if not command or not _is_bare_pytest(command):
            return None
        logger.warning("test_discipline: 拦截裸 pytest 命令: %s", command)
        return {
            "action": "block",
            "message": (
                "[test_discipline] 禁裸 pytest——一律走 scripts/run_tests.sh。"
                "包装器强制 CI 环境对齐（清凭据/TZ=UTC/每文件子进程隔离），"
                "裸跑与 CI 漂移是历史多次「本地绿 CI 红」的根因。\n"
                "正门：scripts/run_tests.sh tests/your_dir/  或单文件"
                " scripts/run_tests.sh tests/foo/test_x.py -k test_y\n"
                "确需临时禁用：TEST_DISCIPLINE_DISABLE=1（仅限排障，用完即关）。"
            ),
        }
    except Exception as e:
        logger.warning("test_discipline hook failed: %s", e, exc_info=True)
        return None


def _plugin_disabled() -> bool:
    import os
    return os.environ.get("TEST_DISCIPLINE_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def register(ctx: Any) -> None:
    wrapped = on_pre_tool_call

    def _gated(**kwargs: Any) -> Optional[Dict[str, Any]]:
        if _plugin_disabled():
            return None
        return wrapped(**kwargs)

    ctx.register_hook("pre_tool_call", _gated)
    logger.info("test_discipline registered（裸 pytest 拦截）")
