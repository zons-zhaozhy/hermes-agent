"""test_discipline 豁免前缀形态覆盖（deputy-patrol 11-03 11:28 杀班案）。

现场：deputy 班按拦截消息 :136 自教通道重试 ``TEST_DISCIPLINE_DISABLE=1
python3 -m pytest ...``，前缀只设子进程 env，而 _plugin_disabled(:146) 只读
网关进程 os.environ——自教豁免通道被自己拦截，同指纹第 2 次 blocked
触发 block_escalation 升级终止，Job failed。期望（独立于实现推导）：
- 命令前缀显式声明 TEST_DISCIPLINE_DISABLE=1 的 pytest 形态=放行（显式
  豁免声明即合法，拦截消息本就教这个通道）；
- 无前缀裸 pytest 仍拦（正门纪律不变量）；
- env 前缀形态（env TEST_DISCIPLINE_DISABLE=1 pytest）同样放行。
"""

import importlib.util
import pathlib
from typing import Any

_PLUGIN = pathlib.Path(__file__).resolve().parents[2] / "plugins" / "discipline" / "test_discipline.py"


def _td() -> Any:
    spec = importlib.util.spec_from_file_location("td_prefix_under_test", _PLUGIN)
    assert spec is not None and spec.loader is not None  # 期望: 插件可加载
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


td = _td()


def test_inline_disable_prefix_is_exempt() -> None:
    cmd = ("cd ~/code/ai/github/fork/hermes-agent && TEST_DISCIPLINE_DISABLE=1 "
           "python3 -m pytest tests/gateway/test_dingtalk.py -q 2>&1 | tail -4")
    assert not td._is_bare_pytest(cmd)  # 期望: False — 拦截消息:136 自教的豁免前缀必须真正生效


def test_env_disable_prefix_is_exempt() -> None:
    cmd = ("cd ~/code/ai/github/fork/hermes-agent && env TEST_DISCIPLINE_DISABLE=1 "
           "python3 -m pytest tests/gateway/test_dingtalk.py -q")
    assert not td._is_bare_pytest(cmd)  # 期望: False — env 前缀形态同义


def test_bare_pytest_still_blocked() -> None:
    cmd = "cd ~/code/ai/github/fork/hermes-agent && python3 -m pytest tests/gateway/test_dingtalk.py -q"
    assert td._is_bare_pytest(cmd)  # 期望: True — 无豁免声明的裸 pytest 维持拦截(正门纪律不变量)
