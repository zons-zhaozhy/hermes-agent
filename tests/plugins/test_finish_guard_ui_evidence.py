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
_DEFAULT_EVIDENCE = "CDP 实访 7 个导航目标, 全部 HTTP 200 + DOM 落地渲染"


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
                    claim: str = _DEFAULT_CLAIM,
                    evidence: str | None = _DEFAULT_EVIDENCE) -> None:
    a: dict = {"claim": claim, "result": "pass"}
    if evidence is not None:
        a["evidence"] = evidence
    w: dict = {"url": "http://127.0.0.1:8130/", "checked_at": checked_at,
               "assertions": [a]}
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


class TestHardAssertionEvidence:
    """硬断言与可达性断言必须带 evidence——关键词命中只证明措辞, 不证明实测。"""

    def _mk(self, tmp_path: Path, claim: str, evidence: str | None) -> tuple[Path, dict]:
        repo = tmp_path / "repo"
        (repo / ".hermes").mkdir(parents=True, exist_ok=True)
        (repo / ".hermes" / "ui_hard_assertions.json").write_text(
            json.dumps({"staff": ["键盘"]}), encoding="utf-8")
        a: dict = {"claim": claim, "result": "pass"}
        if evidence is not None:
            a["evidence"] = evidence
        w = {"role": "staff", "url": "http://127.0.0.1:8130/",
             "checked_at": int(time.time()), "assertions": [a]}
        return repo, w

    def test_keyword_hit_without_evidence_rejected(self, tmp_path):
        repo, w = self._mk(tmp_path, "键盘可达", None)
        defect = ui_evidence._hard_assertion_defect(0, w, repo)
        # 期望: 关键词命中但无 evidence 被拒（措辞迎合通道封堵）
        assert defect is not None and "evidence" in defect

    def test_keyword_hit_with_evidence_passes(self, tmp_path):
        repo, w = self._mk(tmp_path, "键盘可达", "CDP Input.dispatchKeyEvent Tab×8 焦点链实测")
        defect = ui_evidence._hard_assertion_defect(0, w, repo)
        assert defect is None  # 期望: 同条带实测证据则放行

    def test_reachability_claim_without_evidence_rejected(self, tmp_path):
        _repo, w = self._mk(tmp_path, "全部导航链接可达", None)
        defect = ui_evidence._validate_assertions(0, w)
        # 期望: 可达性断言缺 evidence 被拒（实访记录是唯一凭证）
        assert defect is not None and "evidence" in defect

    def test_no_declaration_file_no_enforcement(self, tmp_path):
        repo = tmp_path / "bare"
        (repo / ".hermes").mkdir(parents=True, exist_ok=True)
        w = {"role": "staff", "assertions": [{"claim": "键盘可达", "result": "pass"}]}
        assert ui_evidence._hard_assertion_defect(0, w, repo) is None  # 期望: 未声明项目零影响（向后兼容）

    def _mk_all(self, tmp_path: Path, cfg: dict, claim: str, role: str) -> tuple[Path, dict]:
        repo = tmp_path / "repo"
        (repo / ".hermes").mkdir(parents=True, exist_ok=True)
        (repo / ".hermes" / "ui_hard_assertions.json").write_text(json.dumps(cfg), encoding="utf-8")
        w = {"role": role, "url": "http://127.0.0.1:8130/",
             "checked_at": int(time.time()),
             "assertions": [{"claim": claim, "result": "pass",
                             "evidence": "CDP 实测记录"}]}
        return repo, w

    def test_all_wildcard_enforces_undeclared_role(self, tmp_path):
        repo, w = self._mk_all(tmp_path, {"all": ["步数"]}, "页面渲染正常无报错", "first_user")
        defect = ui_evidence._hard_assertion_defect(0, w, repo)
        assert defect is not None and "硬断言" in defect  # 期望: all 键对未声明 role 生效——纯软断言被拦（通用词应约束所有角色）

    def test_all_wildcard_hit_with_evidence_passes(self, tmp_path):
        repo, w = self._mk_all(tmp_path, {"all": ["步数"]}, "任务步数实测 3 步完成", "first_user")
        assert ui_evidence._hard_assertion_defect(0, w, repo) is None  # 期望: all 键命中且带 evidence → 放行（命中词+实测证据=合格断言）

    def test_all_and_role_keywords_merge(self, tmp_path):
        cfg = {"all": ["步数"], "staff": ["键盘"]}
        repo, w = self._mk_all(tmp_path, cfg, "键盘可达全链", "staff")
        assert ui_evidence._hard_assertion_defect(0, w, repo) is None  # 期望: role 命中自身声明词即放行（all 合并不阻断 role 原语义）
        repo2, w2 = self._mk_all(tmp_path, cfg, "任务步数实测 3 步", "staff")
        assert ui_evidence._hard_assertion_defect(0, w2, repo2) is None  # 期望: role 条目也可命中 all 通用词（合并=任一命中即算）


class TestDesignCriteria:
    """设计判据锚——参考图（对照对象）+ 数值判据（可对账指标），opt-in 声明即执法。"""

    _CFG = {"reference_images": {"required": True},
            "metrics": {"primary_cta_font_px": {"min": 13},
                        "body_line_height_ratio": {"min": 1.4, "max": 1.9}}}

    def _mk(self, tmp_path: Path, w: dict) -> tuple[Path, dict]:
        repo = tmp_path / f"repo-{time.time_ns()}"
        (repo / ".hermes").mkdir(parents=True)
        (repo / ".hermes" / "ui_design_criteria.json").write_text(
            json.dumps(self._CFG), encoding="utf-8")
        return repo, w

    def _ok_walkthrough(self, tmp_path: Path) -> dict:
        ref = tmp_path / f"ref-{time.time_ns()}.png"
        ref.write_bytes(b"\x89PNG")
        return {"url": "http://127.0.0.1:8130/", "checked_at": int(time.time()),
                "reference_images": [str(ref)],
                "assertions": [
                    {"claim": "主操作按钮字号达标", "result": "pass",
                     "metric": "primary_cta_font_px", "measured": 14,
                     "evidence": "CDP getComputedStyle 14px"},
                    {"claim": "正文行高比值达标", "result": "pass",
                     "metric": "body_line_height_ratio", "measured": 1.6,
                     "evidence": "line-height 25.6px / font-size 16px"},
                ]}

    def test_no_declaration_file_no_enforcement(self, tmp_path):
        repo = tmp_path / "bare"
        (repo / ".hermes").mkdir(parents=True)
        w = {"url": "u", "assertions": [{"claim": "c", "result": "pass"}]}
        assert ui_evidence._design_criteria_defect(0, w, repo) is None  # 期望: 未声明项目零影响（向后兼容）

    def test_ok_walkthrough_passes(self, tmp_path):
        repo, w = self._mk(tmp_path, self._ok_walkthrough(tmp_path))
        assert ui_evidence._design_criteria_defect(0, w, repo) is None  # 期望: 参考图存在+全指标落界+evidence 齐 → 放行

    def test_missing_reference_images_rejected(self, tmp_path):
        w = self._ok_walkthrough(tmp_path)
        w.pop("reference_images")
        repo, w = self._mk(tmp_path, w)
        defect = ui_evidence._design_criteria_defect(0, w, repo)
        # 期望: 参考图必填声明下缺 reference_images 被拦（凭空生成无对照）
        assert defect is not None and "reference_images" in defect

    def test_relative_reference_path_rejected(self, tmp_path):
        w = self._ok_walkthrough(tmp_path)
        w["reference_images"] = ["relative/ref.png"]
        repo, w = self._mk(tmp_path, w)
        defect = ui_evidence._design_criteria_defect(0, w, repo)
        assert defect is not None and "绝对路径" in defect  # 期望: 相对路径不可定位, 被拒

    def test_missing_metric_rejected(self, tmp_path):
        w = self._ok_walkthrough(tmp_path)
        w["assertions"] = [a for a in w["assertions"]
                           if a.get("metric") != "body_line_height_ratio"]
        repo, w = self._mk(tmp_path, w)
        defect = ui_evidence._design_criteria_defect(0, w, repo)
        # 期望: 声明指标缺数值断言被拦（软断言替代不了对账）
        assert defect is not None and "body_line_height_ratio" in defect

    def test_metric_out_of_range_rejected(self, tmp_path):
        w = self._ok_walkthrough(tmp_path)
        for a in w["assertions"]:
            if a.get("metric") == "primary_cta_font_px":
                a["measured"] = 12
        repo, w = self._mk(tmp_path, w)
        defect = ui_evidence._design_criteria_defect(0, w, repo)
        # 期望: 实测值低于 min=13 被拦（不达标即缺陷, 禁措辞修饰）
        assert defect is not None and "越界" in defect

    def test_metric_without_evidence_rejected(self, tmp_path):
        w = self._ok_walkthrough(tmp_path)
        for a in w["assertions"]:
            if a.get("metric") == "primary_cta_font_px":
                a.pop("evidence")
        repo, w = self._mk(tmp_path, w)
        defect = ui_evidence._design_criteria_defect(0, w, repo)
        # 期望: measured 命中但无取得方式被拦（数值可伪造, 取值路径须可复核）
        assert defect is not None and "evidence" in defect

    def test_bool_measured_not_counted_as_numeric(self, tmp_path):
        w = self._ok_walkthrough(tmp_path)
        for a in w["assertions"]:
            if a.get("metric") == "primary_cta_font_px":
                a["measured"] = True  # bool 是 int 子类, 裸 isinstance 会误判
        repo, w = self._mk(tmp_path, w)
        defect = ui_evidence._design_criteria_defect(0, w, repo)
        # 期望: True(=1) 不算数值实测, 按「缺数值判据」拦
        assert defect is not None and "缺数值判据" in defect

    def test_check_level_nudged_when_declared_project_lacks_reference(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        (repo / ".hermes" / "ui_design_criteria.json").write_text(
            json.dumps({"reference_images": {"required": True}}), encoding="utf-8")
        _write_evidence(repo, int(time.time()), shot=_mk_shot(tmp_path),
                        assets=["index-test123.js"])
        r = ui_evidence.check([str(src)], 0, 3)
        # 期望: 声明项目走查证据缺参考图锚 → 整体被拦强制续跑（新锚纳入 check 链）
        assert r.get("action") == "continue" and "reference_images" in r["message"]
