"""finish_guard L4-ui（ui_evidence）门禁测试——UI 交付走查证据判定。

负例直调纯函数规范：不依赖真实插件注册，直接调用判定函数断言行为。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from plugins.finish_guard import ui_evidence


def _mk_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    (repo / ".hermes").mkdir()
    src = repo / "src" / "App.vue"
    src.parent.mkdir(parents=True)
    src.write_text("<template/>", encoding="utf-8")
    return repo, src


def _write_evidence(repo: Path, checked_at: int, claim: str = "顶栏显示 身份 · admin；全部导航目标可达非404") -> None:
    payload = {
        "walkthroughs": [
            {"url": "http://127.0.0.1:8130/", "checked_at": checked_at,
             "assertions": [{"claim": claim, "result": "pass"}]},
        ]
    }
    (repo / ".hermes" / "ui_walkthrough.json").write_text(
        json.dumps(payload), encoding="utf-8")


class TestIsUiSource:
    def test_vue_is_ui(self):
        assert ui_evidence._is_ui_source("apps/x/frontend/src/App.vue") is True  # 期望: True（.vue 属界面源文件）

    def test_dist_excluded(self):
        assert ui_evidence._is_ui_source("apps/x/dist/ui/App.vue") is False  # 期望: False（dist 构建产物豁免）

    def test_python_not_ui(self):
        assert ui_evidence._is_ui_source("apps/x/webapp.py") is False  # 期望: False（.py 非界面文件）


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
        _write_evidence(repo, int(time.time()) + 10)
        r = ui_evidence.check([str(src)], 0, 3)
        assert r == {}  # 期望: {}（证据齐全且 checked_at 晚于改动 mtime）

    def test_stale_evidence_nudged(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()) - 3600)
        r = ui_evidence.check([str(src)], 0, 3)
        assert r.get("action") == "continue"  # 期望: 强制续跑（陈旧证据=改动后未复走查）

    def test_stale_evidence_message_explains_timestamp(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        _write_evidence(repo, int(time.time()) - 3600)
        r = ui_evidence.check([str(src)], 0, 3)
        assert "早于" in r["message"]  # 期望: 提示指明时间戳缺陷位置


class TestValidateEvidence:
    def test_assertion_missing_claim(self):
        data = {"walkthroughs": [{"url": "u", "checked_at": int(time.time()),
                                  "assertions": [{"result": "pass"}]}]}
        defect = ui_evidence._validate_evidence(data, 0)
        assert defect is not None and "claim" in defect  # 期望: 断言缺 claim 被拒

    def test_top_level_not_object(self):
        defect = ui_evidence._validate_evidence([1, 2], 0)
        assert defect is not None  # 期望: 非对象结构被拒


class TestRepoRoot:
    def test_finds_git_ancestor(self, tmp_path):
        repo, src = _mk_repo(tmp_path)
        assert ui_evidence._repo_root(str(src)) == repo  # 期望: 从源文件祖先链定位 .git 根
