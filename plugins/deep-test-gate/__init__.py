"""deep-test-gate：深度测试声明门禁（pre_verify 层）。

病根（1018 用户定性）：审查/测试方法论停在 skill 文档层，靠会话自觉执行——
「md 里几句话糊弄」。本插件把「深度测试完成」强声明变成机器可验证事件：
改了代码 + 声明深度测试完成 + pytest 项目无战役账本/无三门禁证据 → 拦截。

三门禁（skill:comprehensive-deep-testing scripts/ 与 ontox tools/ci/ 同源）：
  check_deep_test_matrix.py 战役矩阵（battle_plan.yaml 账本）
  check_rtm_ledger.py RTM 双向对账
  check_coverage_ratchet.py 覆盖率棘轮
"""
from __future__ import annotations

import logging
import re  # re-ok: 中文声明词+豁免词模糊语义匹配, str 方法无等价表达
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.plugins import PluginContext

logger = logging.getLogger("hermes.plugins.deep_test_gate")

# pre_verify 单回合最多强制续跑次数（防拦截死循环）
_MAX_NUDGES = 2

# 强声明词：深度测试/测试战役 + 完成语义（同回复共现即视为强声明）
_DEEP_CLAIM_RE = re.compile(r"(深度测试|测试战役|全方位测试|深度审查)")
_DONE_RE = re.compile(r"(完成|全绿|全部通过|全过|收官|交付达标)")
# 证据标记：三门禁 PASS 原文输出（工具 stdout 前缀）
_EVIDENCE_RE = re.compile(
    r"\[(deep-test-matrix|rtm-check|coverage-ratchet)\][^\n]*(PASS|全层覆盖)")
# 豁免声明：非 pytest 项目/机制不适用（显式说明才放行）
_EXEMPT_RE = re.compile(
    r"(非 ?pytest|无测试套件|非 Python 项目|战役账本不适用|门禁不适用)")


def _is_pytest_project(changed_paths: list[str]) -> bool:
    """改动文件所属项目是否 pytest 项目（存在 tests/ 目录且含 test_*.py）。

    Contract: Postconditions: 只在祖先链内搜索；找不到返回 False（不抛错）。
    """
    roots: set[Path] = set()
    for p in changed_paths:
        cur = Path(p).resolve()
        for _ in range(6):
            if (cur / "tests").is_dir():
                roots.add(cur)
                break
            if cur.parent == cur:
                break
            cur = cur.parent
    return any((root / "tests").glob("test_*.py") for root in roots)


def _has_battle_ledger(changed_paths: list[str]) -> bool:
    """项目内是否已有战役账本（tests/battle_plan.yaml 祖先搜索）。

    Contract: Postconditions: 存在即 True；搜索失败按不存在处理（fail-closed
    语义由调用方拦截补账本，不静默放行）。
    """
    for p in changed_paths:
        cur = Path(p).resolve()
        for _ in range(6):
            if (cur / "tests" / "battle_plan.yaml").exists():
                return True
            if cur.parent == cur:
                break
            cur = cur.parent
    return False


def _should_block(response: str, changed_paths: list[str]) -> bool:
    """拦截判定链：强声明 + 无证据/豁免 + pytest 项目 + 无战役账本。

    Contract: Preconditions: response 非空、changed_paths 为代码文件列表；
    Postconditions: 四条件全真返回 True；任一为假返回 False。
    """
    if not (_DEEP_CLAIM_RE.search(response) and _DONE_RE.search(response)):
        return False
    if _EVIDENCE_RE.search(response) or _EXEMPT_RE.search(response):
        return False
    if not _is_pytest_project(changed_paths):
        return False
    return not _has_battle_ledger(changed_paths)


def _on_pre_verify(**kwargs) -> dict:
    """Contract: Preconditions: 本回合改过文件（changed_paths 非空才调）；
    Postconditions: _should_block 为真且 attempt 未超限时返回
    {"action":"continue","message":...} 强制续跑补机制；否则返回 {} 放行。
    """
    attempt = int(kwargs.get("attempt") or 0)
    if attempt >= _MAX_NUDGES:
        return {}
    response = str(kwargs.get("final_response") or "")
    changed_paths = [p for p in (kwargs.get("changed_paths") or [])
                     if not str(p).endswith((".md", ".txt", ".rst"))]
    if not changed_paths or not response:
        return {}
    if not _should_block(response, changed_paths):
        return {}
    logger.warning(
        "deep-test-gate: 深度测试强声明无战役账本/三门禁证据（attempt=%d），强制补机制", attempt
    )
    return {
        "action": "continue",
        "message": (
            "[deep-test-gate] 回复声明「深度测试完成/全绿」，但目标 pytest 项目"
            "无 tests/battle_plan.yaml 战役账本，回复也无三门禁 PASS 证据。"
            "补齐机制再收尾：①拷 skill:comprehensive-deep-testing 的 "
            "templates/battle_plan.yaml 建账本，逐层填 status+evidence（实测数字）；"
            "②跑 tools/ci/check_deep_test_matrix.py --plan ... --blocked-baseline ...；"
            "③跑 check_rtm_ledger.py --ledger tests/rtm_ledger.yaml --tests tests；"
            "④跑 check_coverage_ratchet.py --source <包> --tests tests --baseline "
            "<json>（无基线先 --set-baseline 实测值）。三门禁 PASS 原文贴进回复后"
            "再收尾；若本项目非 pytest/机制不适用，回复中显式写明豁免原因。"
        ),
    }


def register(ctx: PluginContext) -> None:
    """注册 pre_verify 钩子。

    Contract: Preconditions: ctx 提供 register_hook(name, fn)；
    Postconditions: 注册成功记录 info 日志；失败抛出（禁静默降级）。
    """
    ctx.register_hook("pre_verify", _on_pre_verify)
    logger.info("deep-test-gate 已注册（pre_verify 深度测试声明门禁）")
