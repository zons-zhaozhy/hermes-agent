"""roadmap 既定工程连做执法回归测试（行为契约）.

背景：2026-10-28 用户点名「谁让你总是停摆的」——首技能收官轮以
「其余 6 技能待铺开」待办清单收尾未被任何门禁拦截。本测试锁定
opt-in 声明即执法的行为契约：声明存在+pending+停轮措辞→拦；
无声明/工程收官/无停轮信号/达上限→放行。
"""

import json
from pathlib import Path

from plugins.finish_guard import roadmap as rm


def _make_repo(tmp_path: Path, *, with_roadmap: bool = True,
               statuses: list[str] | None = None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".hermes").mkdir()
    if with_roadmap:
        items = [
            {"name": n, "status": s}
            for n, s in zip(["item-a", "item-b", "item-c"], statuses or ["done", "pending", "pending"])
        ]
        data = {"projects": [{"name": "proj", "granted_by": "test", "items": items}]}
        (repo / ".hermes" / "engineering_roadmap.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return repo


def test_stall_tail_with_pending_is_blocked(tmp_path):
    repo = _make_repo(tmp_path)
    resp = "首技能全链收官，commit 已 push。其余 6 技能出题待按此工序铺开。"
    out = rm.check(resp, [str(repo / "docs" / "a.md")], attempt=0, max_nudges=3)
    assert out.get("action") == "continue"  # 期望: 停轮措辞+pending 项→强制续跑
    assert "既定工程" in out["message"] and "item-b" in out["message"]


def test_no_declaration_no_enforcement(tmp_path):
    repo = _make_repo(tmp_path, with_roadmap=False)
    resp = "其余 6 技能出题待按此工序铺开。"
    out = rm.check(resp, [str(repo / "docs" / "a.md")], attempt=0, max_nudges=3)
    assert out == {}  # 期望: 无声明文件→零执法（opt-in 向后兼容）


def test_all_done_project_passes(tmp_path):
    repo = _make_repo(tmp_path, statuses=["done", "done", "done"])
    resp = "首技能全链收官，其余 6 技能出题待按此工序铺开。"
    out = rm.check(resp, [str(repo / "docs" / "a.md")], attempt=0, max_nudges=3)
    assert out == {}  # 期望: 工程全部 done→不拦（收官收尾合法）


def test_done_statement_negates_stall(tmp_path):
    repo = _make_repo(tmp_path)
    resp = "最后一项出题完成，7 技能全部完成。"
    out = rm.check(resp, [str(repo / "docs" / "a.md")], attempt=0, max_nudges=3)
    assert out == {}  # 期望: 完成态收尾即使残留 pending 声明也不误拦


def test_no_stall_wording_passes(tmp_path):
    repo = _make_repo(tmp_path)
    resp = "本回合完成 item-b 的矩阵分解与出题，commit abc123。"
    out = rm.check(resp, [str(repo / "docs" / "a.md")], attempt=0, max_nudges=3)
    assert out == {}  # 期望: 正常阶段性汇报（无停轮措辞）不拦


def test_attempt_cap_passes(tmp_path):
    repo = _make_repo(tmp_path)
    resp = "其余 6 技能出题待按此工序铺开。"
    out = rm.check(resp, [str(repo / "docs" / "a.md")], attempt=3, max_nudges=3)
    assert out == {}  # 期望: 达续跑上限→放行（防死循环兜底）


def test_broken_roadmap_json_disables(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".hermes" / "engineering_roadmap.json").write_text("{bad json", encoding="utf-8")
    resp = "其余 6 技能出题待按此工序铺开。"
    out = rm.check(resp, [str(repo / "docs" / "a.md")], attempt=0, max_nudges=3)
    assert out == {}  # 期望: 声明解析失败→不执法（fail-open 同 ui_hard_assertions）


def test_repo_root_discovery_via_changed_path(tmp_path):
    repo = _make_repo(tmp_path)
    deep = repo / "apps" / "x" / "docs"
    deep.mkdir(parents=True)
    resp = "剩余 3 项待完成。"
    out = rm.check(resp, [str(deep / "matrix.md")], attempt=0, max_nudges=3)
    assert out.get("action") == "continue"  # 期望: 深层改动路径也能沿祖先链找到仓库根声明
