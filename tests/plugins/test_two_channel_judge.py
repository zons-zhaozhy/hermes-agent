"""验证分通道（build_two_channel_text）行为测试。

依据：arXiv 2609.15364 失败模式一——verifier 必须只认环境反馈（工具输出），
禁采信执行方自述。分通道函数的契约：采信序声明起首、工具输出在前、自述
可有可无、空输入安全。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins._llm_judge import build_two_channel_text


def test_two_channel_order_and_declaration() -> None:
    out = build_two_channel_text("exit 1\nAssertionError: boom", "我已经全部验证通过了")
    # 期望: 采信序声明在最前（覆盖性指令必须先于任何证据出现）
    assert out.startswith("证据采信序")
    # 期望: 工具输出段在自述段之前（环境反馈优先级高的物理体现）
    assert out.index("【工具输出】") < out.index("【执行方自述】")
    assert "AssertionError: boom" in out and "我已经全部验证通过了" in out


def test_empty_actor_summary_omits_channel() -> None:
    out = build_two_channel_text("=== 29 passed ===", "")
    # 期望: 无自述时不产生自述通道段——声明内文会提及该词，故断言段头标记（前置换行）而非裸词
    assert "\n【执行方自述】" not in out
    assert "【工具输出】" in out and "29 passed" in out


def test_both_empty_still_declares() -> None:
    out = build_two_channel_text("", "")
    # 期望: 全空输入仍有声明+工具输出段（fail-loud 结构，不产出裸空串）
    assert out.startswith("证据采信序")
    assert "【工具输出】" in out
