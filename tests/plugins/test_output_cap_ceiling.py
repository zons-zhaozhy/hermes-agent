"""output_cap_ceiling_middleware invariants (adaptive-reasoning plugin).

The bigmodel/zai coding channel answers any max_tokens above 131072 with
HTTP 400 code 1210. Recovery budgets computed from the context window
(window-minus-input ≈ 250k on a 400k-window model) exceed that band, so the
final request cap must be clamped to the per-model ceiling declared under
``providers.<p>.models.<m>.max_output_tokens`` — whatever the cap's source.
"""

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "adaptive-reasoning" / "__init__.py"


def _load_plugin() -> Any:
    spec = importlib.util.spec_from_file_location("adaptive_reasoning_ceiling_under_test", _PLUGIN)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件存在且可加载
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    module = _load_plugin()
    module._CEILING_CACHE.update(fingerprint=None, value=None)
    return module


_CEILING_CFG = (
    "providers:\n"
    "  zai:\n"
    "    models:\n"
    "      glm-5.3:\n"
    "        max_output_tokens: 131072\n"
)


def _request(cap: int) -> dict:
    return {"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": cap}


def test_clamps_oversized_recovery_budget(plugin: Any) -> None:
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(_CEILING_CFG)
    result = plugin.output_cap_ceiling_middleware(
        request=_request(250000), api_mode="chat_completions", provider="zai")
    assert result is not None and result["request"]["max_tokens"] == 131072  # 期望: 窗口推导预算被截到服务端上限
    assert result["source"] == "output-cap-ceiling"  # 期望: trace 来源可辨识


def test_cap_at_ceiling_passes_through(plugin: Any) -> None:
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(_CEILING_CFG)
    req = _request(131072)
    result = plugin.output_cap_ceiling_middleware(request=req, provider="zai")
    assert result is None  # 期望: 恰在上限(合法域边界)不动
    assert req["max_tokens"] == 131072  # 期望: 原请求未被原地改动


def test_no_ceiling_configured_is_noop(plugin: Any) -> None:
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text("model:\n  default: glm-5.3\n")
    result = plugin.output_cap_ceiling_middleware(
        request=_request(250000), provider="zai")
    assert result is None  # 期望: 未登记上限的 provider/model 行为不变


def test_config_ceiling_refreshes_on_fingerprint_change(plugin: Any) -> None:
    cfg_path = Path(os.environ["HERMES_HOME"], "config.yaml")
    cfg_path.write_text(_CEILING_CFG)
    clamped = plugin.output_cap_ceiling_middleware(
        request=_request(250000), provider="zai")
    assert clamped is not None  # 期望: 首个指纹(含 max_output_tokens)生效
    cfg_path.write_text("providers:\n  zai:\n    models:\n      glm-5.3: {}\n")
    gone = plugin.output_cap_ceiling_middleware(
        request=_request(250000), provider="zai")
    assert gone is None  # 期望: 键移除后 no-op(缓存随 mtime/size 指纹刷新)
