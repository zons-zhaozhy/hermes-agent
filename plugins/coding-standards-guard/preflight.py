"""coding-standards-guard 写前预检 CLI。

用法:
    python3 plugins/coding-standards-guard/preflight.py <file.py> [file2.py ...]

在 write_file/patch/execute_code 提交**之前**对目标文件(或临时脚本)
全量跑一遍与拦截器完全相同的规则集,一次性列出全部 error/warning,
替代「提交→被拦一处→修一处→再被拦」的打地鼠循环。

Contract:
  Preconditions: 至少一个文件路径参数;文件存在且可读。
  Postconditions: 退出码 0=无 error 级违规; 1=存在 error 级违规,
    全部违规按文件逐条打印到 stdout(不截断)。
  Invariants: 只读检查,不修改任何文件。

已查重(search_files):guard 包内无既有 CLI 入口。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

_spec = importlib.util.spec_from_file_location(
    "csguard", Path(__file__).resolve().parent / "__init__.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_run_all_checks = _mod._run_all_checks


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    has_error = False
    for path in argv[1:]:
        p = Path(path)
        if not p.is_file():
            print(f"[preflight] 跳过(不存在): {path}")
            continue
        source = p.read_text(encoding="utf-8", errors="replace")
        skip_tests = "/tests/" in p.as_posix() or p.as_posix().startswith("tests/")
        violations = _run_all_checks(source, skip_tests=skip_tests)
        if not violations:
            print(f"[preflight] {path}: 干净")
            continue
        for v in violations:
            print(f"{path}: L{v.line} [{v.rule_id}] {v.severity} {v.message}")
            if v.snippet:
                print(f"    {v.snippet}")
        if any(v.severity == "error" for v in violations):
            has_error = True
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
