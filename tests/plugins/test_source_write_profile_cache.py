"""source_write cache 前缀 profile-aware 回归（反方审查 D4）。

缺陷: _SAFE_PREFIXES 的 hermes cache 前缀在 import 期 expanduser("~/.hermes/")
冻结默认 home——非默认 profile 会话的 TMPDIR（<profile home>/cache/scratch）
不在冻结前缀内，临时区写误拦原样存在（PR #3575 同类路径漂移）。

期望值独立推导（源自「插件路径须与 get_hermes_home() 同源解析」）：
- monkeypatch get_hermes_home 指向任意目录 → <该目录>/cache/** 视为安全路径
- 常规源码路径不受影响（不因豁免扩大而放行）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hermes_constants
from plugins.guards import source_write


def test_cache_prefix_follows_active_profile_home(monkeypatch, tmp_path) -> None:
    # 期望: profile home 变化 → cache 豁免前缀随之变化（运行时解析）
    fake_home = tmp_path / "lab-home"
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: fake_home)
    prefix = source_write._hermes_cache_prefix()
    assert prefix == (str(fake_home).lower() + "/cache/")  # 期望: 前缀取当前 home 而非冻结值
    assert source_write._is_safe_path(str(fake_home / "cache" / "scratch" / "payload.json"))


def test_regular_source_path_still_protected(monkeypatch, tmp_path) -> None:
    # 期望: 豁免不扩大——仓库源码路径仍非安全
    fake_home = tmp_path / "lab-home"
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: fake_home)
    assert not source_write._is_safe_path(str(tmp_path / "repo" / "app" / "core.py"))
