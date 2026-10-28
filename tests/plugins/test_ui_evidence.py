"""ui_evidence 行为契约测试——UI 源识别/可达性断言/多角色覆盖/时间戳门。

Regression for skb 2026-10-26: 用户指令「全站所有功能可达性多角色走查纳入
每次改动的验收」——/compare 断链(router/index.ts 修改)与单角色放行是两类
漏拦, 本测试锁死两类判定。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from plugins.finish_guard import ui_evidence as ue


def test_vue_is_ui_source() -> None:
    assert ue._is_ui_source("apps/x/frontend/src/views/Home.vue") is True  # 期望: 界面源识别


def test_dist_not_ui_source() -> None:
    assert ue._is_ui_source("apps/x/dist/skb-ui/assets/index.js") is False  # 期望: 构建产物豁免


def test_frontend_ts_is_ui_source() -> None:
    assert ue._is_ui_source("apps/x/frontend/src/router/index.ts") is True  # 期望: 前端路由源识别


def test_plain_ts_not_ui_source() -> None:
    assert ue._is_ui_source("tools/ci/check_something.ts") is False  # 期望: 非前端 ts 豁免


def test_missing_reachability_assertion_rejected() -> None:
    w = {"url": "http://x/", "checked_at": 9e18, "assertions": [{"claim": "首屏正常", "result": "pass"}]}
    assert "可达" in str(ue._validate_assertions(0, w))  # 期望: 缺可达性断言被拦


def test_reachability_assertion_accepted() -> None:
    w = {"url": "http://x/", "checked_at": 9e18,
         "assertions": [{"claim": "全部导航目标可达非404", "result": "pass"}]}
    assert ue._validate_assertions(0, w) is None  # 期望: 含可达锚放行


def test_role_coverage_missing_role_rejected() -> None:
    data = {"roles": ["public", "staff"],
            "walkthroughs": [{"url": "http://x/", "role": "public", "checked_at": 9e18,
                              "assertions": [{"claim": "导航可达", "result": "pass"}]}]}
    assert "staff" in str(ue._validate_role_coverage(data, data["walkthroughs"]))  # 期望: 缺角色被拦


def test_role_coverage_full_pass() -> None:
    data = {"roles": ["public", "staff"],
            "walkthroughs": [
                {"url": "http://x/", "role": "public", "checked_at": 9e18,
                 "assertions": [{"claim": "导航可达", "result": "pass"}]},
                {"url": "http://x/admin/", "role": "staff", "checked_at": 9e18,
                 "assertions": [{"claim": "后台可达", "result": "pass"}]},
            ]}
    assert ue._validate_role_coverage(data, data["walkthroughs"]) is None  # 期望: 全角色覆盖放行


def test_no_roles_declared_passes() -> None:
    data = {"walkthroughs": [{"url": "http://x/", "checked_at": 9e18,
                              "assertions": [{"claim": "导航可达", "result": "pass"}]}]}
    assert ue._validate_role_coverage(data, data["walkthroughs"]) is None  # 期望: 单角色项目不误拦


def _make_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_check_full_evidence_passes(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    ui = root / "src" / "App.vue"
    ui.parent.mkdir()
    ui.write_text("<template/>", encoding="utf-8")
    shot = tmp_path / "walk.png"
    shot.write_bytes(b"\x89PNG")  # 与 checked_at 同时刻的新截图(截图锚)
    ev = root / ".hermes" / "ui_walkthrough.json"
    ev.parent.mkdir()
    ev.write_text(json.dumps({
        "roles": ["public"],
        "walkthroughs": [{"url": "http://x/", "role": "public",
                          "checked_at": int(time.time()),
                          "screenshots": [str(shot)],
                          "assertions": [{"claim": "全部导航可达", "result": "pass"}]}],
    }), encoding="utf-8")
    assert ue._check_evidence([str(ui)]) is None  # 期望: 合法证据放行（tmp 仓无 dist, dom 锚降级）


def test_check_stale_evidence_blocked(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    ui = root / "src" / "App.vue"
    ui.parent.mkdir()
    ui.write_text("<template/>", encoding="utf-8")
    ev = root / ".hermes" / "ui_walkthrough.json"
    ev.parent.mkdir()
    ev.write_text(json.dumps({
        "walkthroughs": [{"url": "http://x/", "checked_at": 1,
                          "assertions": [{"claim": "全部导航可达", "result": "pass"}]}],
    }), encoding="utf-8")
    assert ue._check_evidence([str(ui)]) is not None  # 期望: 早于改动 mtime 的证据被拦


def test_check_attempt_cap_releases(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    ui = root / "src" / "App.vue"
    ui.parent.mkdir()
    ui.write_text("<template/>", encoding="utf-8")
    verdict = ue.check([str(ui)], attempt=3, max_nudges=3)
    assert verdict == {}  # 期望: attempt 达上限放行防死锁


# ── opt-in 硬断言检测（2026-10-28 新增: 断言质量执法, 声明即执法/未声明零影响） ──

def test_hard_assertion_soft_claims_rejected(tmp_path: Path) -> None:
    d = tmp_path / ".hermes"
    d.mkdir()
    (d / "ui_hard_assertions.json").write_text(
        json.dumps({"staff": ["键盘", "一屏", "理解"]}), encoding="utf-8")
    w = {"role": "staff", "assertions": [{"claim": "渲染正常 13 行", "result": "pass"}]}
    out = ue._hard_assertion_defect(0, w, tmp_path)
    assert out is not None and "全软" in out  # 期望: 声明 role 软断言被拦


def test_hard_assertion_hard_claim_accepted(tmp_path: Path) -> None:
    d = tmp_path / ".hermes"
    d.mkdir()
    (d / "ui_hard_assertions.json").write_text(
        json.dumps({"staff": ["键盘", "一屏", "理解"]}), encoding="utf-8")
    w = {"role": "staff", "assertions": [
        {"claim": "全程键盘零鼠标完成一次任务查看", "result": "pass"}]}
    assert ue._hard_assertion_defect(0, w, tmp_path) is None  # 期望: 硬断言命中关键词放行


def test_hard_assertion_no_config_pass_through(tmp_path: Path) -> None:
    (tmp_path / ".hermes").mkdir()
    w = {"role": "staff", "assertions": [{"claim": "渲染正常", "result": "pass"}]}
    assert ue._hard_assertion_defect(0, w, tmp_path) is None  # 期望: 无声明文件不执法


def test_hard_assertion_undeclared_role_pass_through(tmp_path: Path) -> None:
    d = tmp_path / ".hermes"
    d.mkdir()
    (d / "ui_hard_assertions.json").write_text(
        json.dumps({"staff": ["键盘"]}), encoding="utf-8")
    w = {"role": "public", "assertions": [{"claim": "渲染正常", "result": "pass"}]}
    assert ue._hard_assertion_defect(0, w, tmp_path) is None  # 期望: 未声明 role 不执法
