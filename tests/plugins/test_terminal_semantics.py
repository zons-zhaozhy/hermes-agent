"""terminal_semantics 唯一事实源回归测试。

期望值来源: POSIX shell 重定向语义独立推导——`>`/`>>` 均写文件、fd 前缀
重定向到 sink 不写、`>&n` 描述符复制不写、`>(cmd)` 进程替换不写、SQL
比较符后的数字字面量不是重定向目标。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.guards import terminal_semantics as ts


def test_plain_redirect_writes() -> None:
    assert ts.redirect_writes_file("echo hi > out.txt") is True  # 期望: > 文件=写
    assert ts.redirect_writes_file("echo hi >> log.txt") is True  # 期望: append 同为写


def test_no_extension_target_still_writes() -> None:
    """宽口径: `> out` 无扩展名同样是文件写（gate 旧窄口径漏判修正）。"""
    assert ts.redirect_writes_file("echo hi > out") is True  # 期望: 非 sink 目标即写


def test_sink_and_fd_dup_negative() -> None:
    assert ts.redirect_writes_file("cmd 2>/dev/null") is False  # 期望: sink 不写
    assert ts.redirect_writes_file("cmd >/dev/stderr") is False  # 期望: sink 不写
    assert ts.redirect_writes_file("cmd >&2") is False  # 期望: fd 复制不写


def test_procsub_not_write() -> None:
    assert ts.redirect_writes_file("diff >(ls a) >(ls b)") is False  # 期望: 进程替换不写


def test_sql_numeric_literal_not_target() -> None:
    """SQL 比较符 `price > 9.99` 的数字字面量不是重定向目标（D2 下沉）。"""
    assert ts.redirect_writes_file("sqlite3 db 'SELECT 1 FROM t WHERE price > 9.99'") is False  # 期望: 数字字面量排除
    assert ts.redirect_writes_file("psql -c 'SELECT count(*) FROM t WHERE amt > 100.5'") is False  # 期望: 小数同排除


def test_targets_extraction() -> None:
    assert ts.redirect_file_targets("a > x.txt && b >> y.txt") == ["x.txt", "y.txt"]  # 期望: 按序提取写目标
    assert ts.redirect_file_targets("a 2>/dev/null; b > z.sql") == ["z.sql"]  # 期望: sink 不进目标列


def test_empty_and_bare_gt() -> None:
    assert ts.parse_redirects("") == []  # 期望: 空段零条目
    r = ts.parse_redirects("echo >")
    assert len(r) == 1 and r[0].writes_file is False  # 期望: 行尾裸 > 空目标不构成写且不抛异常


def test_looks_like_path() -> None:
    assert ts.looks_like_path("file.py") is True  # 期望: 扩展名形态
    assert ts.looks_like_path("/abs/x.yaml") is True  # 期望: 绝对路径
    assert ts.looks_like_path("./rel") is True  # 期望: 相对路径
    assert ts.looks_like_path("9.99") is False  # 期望: 数字字面量
    assert ts.looks_like_path("out") is False  # 期望: 裸词非路径形态
    assert ts.looks_like_path("-n") is False  # 期望: 选项非路径
    assert ts.looks_like_path("'quoted.json'") is True  # 期望: 剥引号后判定


def test_never_raises_on_malformed() -> None:
    for seg in ("'", 'echo "unclosed > x', "a > ", ">" * 10, "a>&", "a >("):
        ts.parse_redirects(seg)  # 期望: 残缺形态不抛异常
        ts.redirect_writes_file(seg)
