#!/usr/bin/env python3
# dupcheck-ok: workspace 漂移对账工具——全库无等价实现（search_files drift/installs workspace 零命中）
"""installs workspace 与 fork 源码的漂移对账与同步。

【退役 2026-10-14】gateway 启动已切换为直读 fork 源码（单一真相源，见
docs/single_source_gateway_20261014.md），workspace 快照不再被生产路径
加载，本工具已无对账对象。保留作历史参照与 installs 形态排查兜底。

背景（docs/terminology_guard_failopen_20261012.md）：install 建环境时复制
hermes_cli/ 与 plugins/ 到 environments/*/workspace/，此后 fork 侧修复不会
自动回流——运行副本滞留旧版（已两次实战：terminology-guard 19 份、goal
双缺口 0/28 生效）。本工具把「检测-同步-验证」机器化。

用法：
  对账（零写盘）: python3 tools/workspace_drift_check.py [--paths hermes_cli,plugins/guards]
  同步（带备份）: python3 tools/workspace_drift_check.py --apply [--paths ...]

Contract:
  Preconditions: fork 仓库工作区干净（脏区会同步未提交内容）；
                 目标 workspace 存在且非符号链接。
  Postconditions: dry-run 只打印漂移清单 exit(1 if 漂移 else 0)；
                  --apply 逐份先备份 .driftbak-<date> 再覆盖，同步后
                  编译验证，失败即报错列出（不静默跳过）。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import py_compile
import shutil
import sys
from pathlib import Path

FORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ("hermes_cli", "plugins")
INSTALLS_ROOT = Path.home() / ".hermes" / "installs"
SKIP_DIRS = {"__pycache__", "node_modules", ".git", ".ruff_cache"}


def md5_of(path: Path) -> str:
    """文件 MD5（小文件全量读）。"""
    return hashlib.md5(path.read_bytes()).hexdigest()


def iter_pair_files(fork_sub: Path, ws_sub: Path) -> list[tuple[Path, Path]]:
    """fork 子树与 workspace 子树的配对文件清单（跳过缓存目录；含仅单侧存在的文件）。"""
    def files_under(root: Path) -> set[Path]:
        out: set[Path] = set()
        if not root.exists():
            return out
        for p in root.rglob("*"):
            if p.is_file() and not (set(p.parts) & SKIP_DIRS):
                out.add(p.relative_to(root))
        return out

    rels = files_under(fork_sub) | files_under(ws_sub)
    return [(fork_sub / r, ws_sub / r) for r in sorted(rels)]


def diff_one_pair(fork_sub: Path, ws_sub: Path) -> dict[str, list[str]]:
    """单个子树对账：返回 {仅fork有: [], 内容漂移: []}。"""
    only_fork: list[str] = []
    drifted: list[str] = []
    for f, w in iter_pair_files(fork_sub, ws_sub):
        rel = f.relative_to(FORK_ROOT)
        if not f.exists():
            continue  # 仅 workspace 有=运行态产物（.driftbak 等），不判漂移
        if not w.exists():
            only_fork.append(str(rel))
        elif md5_of(f) != md5_of(w):
            drifted.append(str(rel))
    return {"only_fork": only_fork, "drifted": drifted}


def compile_check(path: Path) -> bool:
    """编译验证（.py 专用，其余类型恒过）。"""
    if path.suffix != ".py":
        return True
    try:
        py_compile.compile(str(path), doraise=True)
        return True
    except py_compile.PyCompileError:
        return False


def sync_one_ws(ws: Path, rels: list[str], stamp: str) -> tuple[int, int]:
    """同步一个 workspace：逐份备份+覆盖+编译验证；返回 (成功数, 失败数)。"""
    copied = failed = 0
    for rel in rels:
        src = FORK_ROOT / rel
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.copy2(dst, dst.with_name(dst.name + f".driftbak-{stamp}"))
        shutil.copy2(src, dst)
        if compile_check(dst):
            copied += 1
        else:
            failed += 1
            print(f"    [COMPILE-FAIL] {dst}")
    return copied, failed


def report_ws(ws: Path, per_ws: dict[str, list[str]], apply: bool, stamp: str) -> int:
    """打印一个 workspace 的漂移报告（+apply 时执行同步）；返回漂移项数。"""
    n = len(per_ws["only_fork"]) + len(per_ws["drifted"])
    label = ws.parents[1].name + "/" + ws.parent.name
    print(f"[DRIFT] {label}: 仅fork有 {len(per_ws['only_fork'])} / 内容漂移 {len(per_ws['drifted'])}")
    for rel in per_ws["only_fork"][:5]:
        print(f"    + {rel}")
    for rel in per_ws["drifted"][:5]:
        print(f"    ~ {rel}")
    if n > 10:
        print(f"    … 共 {n} 项")
    if apply:
        copied, failed = sync_one_ws(ws, per_ws["only_fork"] + per_ws["drifted"], stamp)
        print(f"    → 同步 {copied} 文件（备份后缀 .driftbak-{stamp}），编译失败 {failed}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", default=",".join(DEFAULT_PATHS),
                    help="对账子路径（逗号分隔，相对 fork 根）")
    ap.add_argument("--apply", action="store_true",
                    help="执行同步（默认 dry-run 只对账）")
    args = ap.parse_args()

    subs = [s.strip() for s in args.paths.split(",") if s.strip()]
    workspaces = sorted(INSTALLS_ROOT.glob("*/environments/*/workspace"))
    if not workspaces:
        print("[OK] 无 workspace 可对账")
        sys.exit(0)

    stamp = datetime.date.today().strftime("%y%m%d")
    total_drift = 0
    for ws in workspaces:
        per_ws: dict[str, list[str]] = {"only_fork": [], "drifted": []}
        for sub in subs:
            r = diff_one_pair(FORK_ROOT / sub, ws / sub)
            per_ws["only_fork"] += r["only_fork"]
            per_ws["drifted"] += r["drifted"]
        if per_ws["only_fork"] or per_ws["drifted"]:
            total_drift += report_ws(ws, per_ws, args.apply, stamp)

    if total_drift == 0:
        print("[OK] 全部 workspace 与 fork 一致")
        sys.exit(0)
    mode = "已同步" if args.apply else "dry-run（--apply 执行同步）"
    print(f"[SUMMARY] 漂移 {total_drift} 项，{mode}")
    sys.exit(0 if args.apply else 1)


if __name__ == "__main__":
    main()
