"""modify-guard 契约测试：仓级写互斥+git 夹带禁令。"""

import json
import time
from pathlib import Path

import pytest

from plugins.modify_guard import _acquire, _lock_path, _on_pre_tool_call, _repo_root_for


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """临时 git 仓 + 隔离 HOME（锁文件落在临时家目录）。"""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    root = tmp_path / "proj"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    return root


def _lock_record(root: Path, session_id: str, acquired_at: float) -> None:
    lock = _lock_path(root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"session_id": session_id, "acquired_at": acquired_at, "repo": str(root)}), encoding="utf-8")


class TestWriteMutex:
    def test_no_lock_allows_and_records(self, repo: Path) -> None:
        ok, why = _acquire(repo, "sess-a")
        assert ok is True and why == ""  # 期望: 无锁→首次获得,无拦截说明(锁协议初态)

    def test_same_session_reentry_allows(self, repo: Path) -> None:
        _lock_record(repo, "sess-a", time.time())
        ok, _ = _acquire(repo, "sess-a")
        assert ok is True  # 期望: 同会话二次写→放行(一个修改流程多次落盘)

    def test_other_session_within_ttl_blocks(self, repo: Path) -> None:
        _lock_record(repo, "sess-a", time.time() - 10)
        ok, why = _acquire(repo, "sess-b")
        assert ok is False  # 期望: 他会话持锁未过期(TTL 600s>10s)→拦截
        assert "sess-a" in why and "另一会话" in why  # 期望: 拦截说明含持有者标识

    def test_other_session_expired_takes_over(self, repo: Path) -> None:
        _lock_record(repo, "sess-a", time.time() - 700)
        ok, _ = _acquire(repo, "sess-b")
        assert ok is True  # 期望: 持锁 700s>TTL 600s→视为已释放,接管

    def test_corrupt_lock_self_heals(self, repo: Path) -> None:
        lock = _lock_path(repo)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("not-json", encoding="utf-8")
        ok, _ = _acquire(repo, "sess-a")
        assert ok is True  # 期望: 锁文件损坏→当无锁自愈(损坏自愈条款)

    def test_on_pre_tool_call_write_file_blocks_for_other_session(self, repo: Path) -> None:
        _lock_record(repo, "sess-a", time.time() - 5)
        result = _on_pre_tool_call("write_file", {"path": str(repo / "src" / "a.py")}, "sess-b")
        assert result is not None and result["action"] == "block"  # 期望: 他会话持锁中,写其仓内文件被拦

    def test_on_pre_tool_call_outside_repo_allows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)
        result = _on_pre_tool_call("write_file", {"path": str(tmp_path / "scratch.txt")}, "sess-x")
        assert result is None  # 期望: 无 .git 祖先的仓外路径→不归本门禁管


class TestGitSwallowBan:
    @pytest.mark.parametrize("command", ["git add -A", "git add .", "git commit -a", "git add --all && git commit -m x"])
    def test_swallow_forms_block(self, command: str) -> None:
        result = _on_pre_tool_call("terminal", {"command": command}, "sess-a")
        assert result is not None and result["action"] == "block"  # 期望: 四种全量暂存/提交形态一律拦

    def test_explicit_paths_allow(self) -> None:
        result = _on_pre_tool_call("terminal", {"command": "git add src/a.py && git commit -m fix"}, "sess-a")
        assert result is None  # 期望: 显式文件列表提交→放行

    def test_waiver_allows(self) -> None:
        result = _on_pre_tool_call("terminal", {"command": "git add -A  # modify-ok: 初始化导入"}, "sess-a")
        assert result is None  # 期望: 行内豁免注释→放行


def test_repo_root_anchoring(repo: Path) -> None:
    assert _repo_root_for(repo / "src") == repo  # 期望: 仓内子目录锚定到仓根
    assert _repo_root_for(repo / "no" / "such.py") == repo  # 期望: 仓内不存在的深层路径仍锚定仓根
