"""finish_guard L4-ui（ui_evidence）门禁测试——UI 交付走查证据判定。

负例直调纯函数规范：不依赖真实插件注册，直接调用判定函数断言行为。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from plugins.finish_guard import ui_evidence

_DEFAULT_CLAIM = "顶栏显示 身份 · admin；全部导航目标可达非404"


def _mk_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    (repo / ".hermes").mkdir()
    src = repo / "src" / "App.vue"
    src.parent.mkdir(parents=True)
    src.write_text("<template/>", encoding="utf-8")
    dist = repo / "dist"
    dist.mkdir()
    # 构建产物在源码之后落盘(新鲜), 引用带 hash 的 bundle
    (dist / "index.html").write_text(
        '<script type="module" src="/assets/index-test123.js"></script>',
        encoding="utf-8")
    return repo, src


def _mk_shot(tmp_path: Path, mtime: float | None = None) -> Path:
    shot = tmp_path / f"walk-{time.time_ns()}.png"
    shot.write_bytes(b"\x89PNG")
    if mtime is not None:
        os.utime(shot, (mtime, mtime))
    return shot


def _write_evidence(repo: Path, checked_at: int, shot: Path | None = None,
                    assets: list[str] | None = None,
                    claim: str = _DEFAULT_CLAIM) -> None:
    w: dict = {"url": "http://127.0.0.1:8130/", "checked_at": checked_at,
               "assertions": [{"claim": claim, "result": "pass"}]}
    if shot is not None:
        w["screenshots"] = [str(shot)]
    if assets is not None:
        w["dom_assets"] = assets
    payload = {"walkthroughs": [w]}
    (repo / ".hermes" / "ui_walkthrough.json").write_text(
        json.dumps(payload), encoding="utf-8")


class TestIsUiSource:
    def test_vue_is_ui(self):
        assert ui_evidence._is_ui_source("apps/x/frontend/src/App.vue") is True  # 期望: True（.vue 属界面源文件）

    def test_dist_excluded(self):
        assert ui_evidence._is_ui_source("apps/x/dist/ui/App.vue") is False  # 期望: False（dist 构建产物豁免）

    def test_python_not_ui(self):
        assert ui_evidence._is_ui_source("apps/x/webapp.py") is False  # 期望: False（.py 非界面文件）

    def test_spec_ts_not_ui(self):
        # 期望: False——*.spec.ts 是测试文件，无运行时界面语义；
        # 计入会把 min_checked_at 抬到测试文件 mtime，既有走查证据整体误判过期
        assert ui_evidence._is_ui_source("packages/shared_ui/src/__tests__/TripleState.spec.ts") is False  # 期望: False（测试文件豁免）

    def test_test_ts_not_ui(self):
        assert ui_evidence._is_ui_source("oms/frontend/src/__tests__/api/dataSource.test.ts") is False  # 期望: False（.test.ts 测试文件豁免）

    def test_tests_dir_vue_excluded(self):
        # __tests__/ 目录内即使是 .vue 夹具也不属用户可感知界面源
        assert ui_evidence._is_ui_source("apps/x/frontend/src/__tests__/Fixture.vue") is False  # 期望: False（__tests__ 目录豁免）

    def test_router_ts_still_ui(self):
        assert ui_evidence._is_ui_source("apps/x/frontend/src/router/index.ts") is True  # 期望: True（回归保护：正常前端 .ts 仍算 UI 源）


class TestCheck:
    def test_no_ui_files_pass(self):
        r = ui_evidence.check(["apps/x/webapp.py", "README.md"], 0, 3)
        assert r == {}  # 期望: {}（纯后端/文档改动不触发 UI 走查判定）

    def test_ui_without_evidence_nudged(self, tmp_path):
        _repo, src = _mk_repo(tmp_path)
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（无证据不允许收口）

    def test_ui_without_evidence_message_names_evidence_file(self, tmp_path):
        _repo, src = _mk_repo(tmp_path)
        r = ui_evidence.check([str(src)], 0, 3)
        assert "ui_walkthrough.json" in r["message"]  # 期望: 提示含证据文件名（拦截即教学）

    def test_attempt_cap_passes(self, tmp_path):
        _repo, src = _mk_repo(tmp_path)
        r = ui_evidence.check([str(src)], 3, 3)
        assert r == {}  # 期望: {}（attempt 达上限放行, 与 finish_guard 防死循环契约一致）

    def test_valid_evidence_passes(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()), shot=_mk_shot(tmp_path),
                        assets=["index-test123.js"])
        r = ui_evidence.check([str(src)], 0, 3)
        assert r == {}  # 期望: {}（三锚齐全: 时间在界内+截图同时刻+dom_assets 与产物交集）

    def test_stale_evidence_nudged(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()) - 3600,
                        shot=_mk_shot(tmp_path, time.time() - 3600))
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（陈旧证据=改动后未复走查）

    def test_stale_evidence_message_explains_timestamp(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()) - 3600,
                        shot=_mk_shot(tmp_path, time.time() - 3600))
        r = ui_evidence.check([str(src)], 0, 3)
        assert "早于" in r["message"]  # 期望: 提示指明时间戳缺陷位置


class TestAntiForgeryAnchors:
    def test_future_timestamp_rejected(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()) + 3600,
                        shot=_mk_shot(tmp_path, time.time() + 3600))
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（未来时间戳=永真伪证）
        assert "未来时间" in r["message"]  # 期望: 提示指明上界缺陷

    def test_small_clock_skew_still_passes(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        now = time.time()
        _write_evidence(repo, int(now + 10), shot=_mk_shot(tmp_path, now),
                        assets=["index-test123.js"])
        r = ui_evidence.check([str(src)], 0, 3)
        assert r == {}  # 期望: {}（±120s 内时钟偏差容忍, 不误拦真实走查）

    def test_screenshot_missing_rejected(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()), shot=None,
                        assets=["index-test123.js"])
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（无截图=纯文字自报）
        assert "screenshots" in r["message"]  # 期望: 提示指明截图锚缺失

    def test_stale_screenshot_rejected(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()),
                        shot=_mk_shot(tmp_path, time.time() - 3600),
                        assets=["index-test123.js"])
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（挪用旧截图）
        assert "偏差超" in r["message"]  # 期望: 提示指明截图时间窗缺陷

    def test_dom_assets_missing_rejected(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()), shot=_mk_shot(tmp_path),
                        assets=None)
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（缺页面版本锚）
        assert "dom_assets" in r["message"]  # 期望: 提示指明 dom 锚缺失

    def test_dom_assets_zero_intersection_rejected(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()), shot=_mk_shot(tmp_path),
                        assets=["index-oldhash.js"])
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（抄旧 hash 与当前产物零交集）
        assert "零交集" in r["message"]  # 期望: 提示指明版本不匹配

    def test_stale_bundle_rejected(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        old = time.time() - 7200
        os.utime(repo / "dist" / "index.html", (old, old))
        _write_evidence(repo, int(time.time()), shot=_mk_shot(tmp_path),
                        assets=["index-test123.js"])
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（改码未重建, 页面跑旧产物）
        assert "未重建" in r["message"]  # 期望: 提示指明产物陈旧


class TestValidateEvidence:
    def test_assertion_missing_claim(self, tmp_path):
        shot = _mk_shot(tmp_path)
        data = {"walkthroughs": [{"url": "u", "checked_at": int(time.time()),
                                  "screenshots": [str(shot)],
                                  "assertions": [{"result": "pass"}]}]}
        defect = ui_evidence._validate_evidence(data, 0)
        assert defect is not None and "claim" in defect  # 期望: 断言缺 claim 被拒（截图锚已配齐, 定位到 claim 缺陷）

    def test_top_level_not_object(self):
        defect = ui_evidence._validate_evidence([1, 2], 0)
        assert defect is not None  # 期望: 非对象结构被拒


class TestRepoRoot:
    def test_finds_git_ancestor(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        assert ui_evidence._repo_root(str(src)) == repo  # 期望: 从源文件祖先链定位 .git 根
