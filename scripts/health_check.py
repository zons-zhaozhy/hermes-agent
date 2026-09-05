#!/usr/bin/env python3
"""Hermes 自身健康检查——程序化审计，输出结构化结果。

用法:
  python3 scripts/health_check.py           # 完整审计
  python3 scripts/health_check.py --quick   # 快速（跳过 git fetch）
  python3 scripts/health_check.py --json    # JSON 输出

写入 ~/.hermes/cache/health-check.json 作为可对比快照。
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
CACHE_FILE = HERMES_HOME / "cache" / "health-check.json"


def run(cmd: list[str], cwd=None, timeout=30) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd or REPO_ROOT, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"timeout after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"command not found: {cmd[0]}"


def check_git() -> dict:
    """Git 仓库健康度。"""
    result = {}
    code, out, _ = run(["git", "rev-list", "HEAD..upstream/main", "--count"])
    result["upstream_behind"] = int(out) if code == 0 else -1
    code, out, _ = run(["git", "rev-list", "upstream/main..HEAD", "--count"])
    result["local_ahead"] = int(out) if code == 0 else -1
    code, out, _ = run(["git", "status", "--short"])
    result["dirty_files"] = len(out.split("\n")) if out else 0
    code, out, _ = run(["git", "ls-files", "--others", "--exclude-standard"])
    result["untracked_files"] = len(out.split("\n")) if out else 0
    code, out, _ = run(["git", "log", "--oneline", "--grep=sync upstream", "-1"])
    result["last_sync"] = out if code == 0 else "unknown"
    return result


def check_god_files() -> dict:
    """大文件（>5000 行）列表。"""
    result = {"files": []}
    for pyfile in sorted(REPO_ROOT.rglob("*.py")):
        if ".git/" in str(pyfile) or ".venv/" in str(pyfile) or "venv/" in str(pyfile):
            continue
        try:
            lines = len(pyfile.read_text(encoding="utf-8").splitlines())
        except Exception:
            continue
        if lines >= 5000:
            result["files"].append({"path": str(pyfile.relative_to(REPO_ROOT)), "lines": lines})
    result["files"].sort(key=lambda x: -x["lines"])
    result["count"] = len(result["files"])
    return result


def check_config() -> dict:
    """配置健康度。"""
    result = {}
    config_file = HERMES_HOME / "config.yaml"
    if not config_file.exists():
        result["error"] = "config.yaml not found"
        return result
    try:
        content = config_file.read_text(encoding="utf-8")
        # parse simple YAML values with str methods
        version_line = None
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("_config_version:"):
                version_line = stripped.split(":", 1)[1].strip()
                break
        if version_line:
            result["config_version"] = int(version_line)
        # find read_think_gate section and its enabled value
        in_rtg = False
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("read_think_gate:"):
                in_rtg = True
                continue
            if in_rtg and stripped.startswith("enabled:"):
                result["read_think_gate"] = "enabled=" + stripped.split(":", 1)[1].strip()
                break
            if in_rtg and stripped and not line.startswith(" "):
                in_rtg = False
    except Exception as e:
        result["error"] = str(e)
    return result


def check_precommit_debt() -> dict:
    """Pre-commit 引擎：扫描 bare except pass。"""
    result = {"warnings": 0, "details": []}
    for pyfile in REPO_ROOT.rglob("*.py"):
        if ".git/" in str(pyfile) or ".venv/" in str(pyfile) or "venv/" in str(pyfile):
            continue
        try:
            lines = pyfile.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("except") and ":" in stripped:
                next_block = lines[i:min(i+5, len(lines))]
                has_pass_only = any(
                    "pass" in nl and "logger." not in nl and "raise" not in nl
                    for nl in next_block
                )
                if has_pass_only and not any(
                    "logger.exception" in nl or "logger.error" in nl or "raise" in nl
                    for nl in next_block
                ):
                    result["warnings"] += 1
                    result["details"].append(
                        f"{pyfile.relative_to(REPO_ROOT)}:{i}: bare except pass"
                    )
    return result


def check_plugins_skills() -> dict:
    """插件和 skill 统计。"""
    result = {}
    for key, path in [
        ("builtin_plugins", REPO_ROOT / "plugins"),
        ("builtin_skills", REPO_ROOT / "skills"),
        ("optional_skills", REPO_ROOT / "optional-skills"),
    ]:
        result[key] = len([d for d in path.iterdir() if d.is_dir()]) if path.exists() else 0
    user_plugins = HERMES_HOME / "plugins"
    user_skills = HERMES_HOME / "skills"
    result["user_plugins"] = len([d for d in user_plugins.iterdir() if d.is_dir()]) if user_plugins.exists() else 0
    result["user_skills"] = len([d for d in user_skills.iterdir() if d.is_dir()]) if user_skills.exists() else 0
    return result


def check_cron() -> dict:
    """Cron 作业状态。"""
    result = {}
    try:
        code, out, _ = run(["hermes", "cron", "list"], timeout=10)
        if code == 0:
            result["active_jobs"] = sum(1 for l in out.split("\n") if "[active]" in l)
    except Exception as e:
        result["error"] = str(e)
    return result


def load_previous() -> dict | None:
    """加载上一次检查快照。"""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def compare(current: dict, previous: dict | None) -> list[str]:
    """对比变化。"""
    if not previous:
        return ["首次审计"]
    changes = []
    git_cur = current.get("git", {})
    git_prev = previous.get("git", {})
    for key, label in [
        ("upstream_behind", "上游落后"), ("local_ahead", "本地领先"),
        ("dirty_files", "未提交修改"), ("untracked_files", "未跟踪文件"),
    ]:
        old_v = git_prev.get(key, -1)
        new_v = git_cur.get(key, -1)
        if old_v != new_v:
            changes.append(f"{label}: {old_v} → {new_v}")
    for section_key, section_label, metric_key in [
        ("god_files", "超5000行文件", "count"),
        ("precommit_debt", "吞异常数量", "warnings"),
    ]:
        old_v = previous.get(section_key, {}).get(metric_key, 0)
        new_v = current.get(section_key, {}).get(metric_key, 0)
        if old_v != new_v:
            changes.append(f"{section_label}: {old_v} → {new_v}")
    return changes if changes else ["无变化"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hermes 自身健康检查")
    parser.add_argument("--quick", action="store_true", help="跳过 git fetch")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    if not args.quick:
        run(["git", "fetch", "upstream", "--quiet"], timeout=30)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "hermes_home": str(HERMES_HOME),
        "git": check_git(),
        "god_files": check_god_files(),
        "config": check_config(),
        "precommit_debt": check_precommit_debt(),
        "plugins_skills": check_plugins_skills(),
        "cron": check_cron(),
    }

    previous = load_previous()
    result["changes"] = compare(result, previous)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"Hermes 健康检查 — {now}")
        print(f"  REPO: {result['repo_root']}")
        print(f"  Cache: {CACHE_FILE}")
        print()
        g = result["git"]
        print(f"Git: behind={g['upstream_behind']} ahead={g['local_ahead']} "
              f"dirty={g['dirty_files']} untracked={g['untracked_files']}")
        print(f"  last sync: {g['last_sync'][:80]}")
        print()
        print("God Files (>5000行):")
        for f in result["god_files"]["files"]:
            print(f"  {f['path']}: {f['lines']:,}")
        print()
        print("Pre-commit Debt:", result["precommit_debt"]["warnings"])
        for d in result["precommit_debt"]["details"]:
            print(f"  {d}")
        print()
        ps = result["plugins_skills"]
        print(f"Plugins: {ps['builtin_plugins']} builtin + {ps['user_plugins']} user")
        print(f"Skills: {ps['builtin_skills']} builtin + {ps['optional_skills']} optional "
              f"+ {ps['user_skills']} user")
        print(f"Cron: {result['cron'].get('active_jobs', '?')} active")
        print()
        print("Changes vs previous run:")
        for c in result["changes"]:
            print(f"  {c}")


if __name__ == "__main__":
    main()
