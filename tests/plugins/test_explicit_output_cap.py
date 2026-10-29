"""explicit_output_cap_middleware invariants (adaptive-reasoning plugin).

The middleware fills ``max_tokens`` from config ``model.max_tokens`` into chat
requests that carry NO explicit output cap — closing the relay-side 8192 default
that truncated glm-5.3 forced-thinking responses (finish_reason=length).
"""

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "adaptive-reasoning" / "__init__.py"


def _load_plugin() -> Any:
    spec = importlib.util.spec_from_file_location("adaptive_reasoning_under_test", _PLUGIN)
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
    module._CAP_CACHE.update(fingerprint=None, value=None, warned_invalid=False)
    return module


def _write_config(body: str) -> None:
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(body)


def test_fills_cap_when_absent(plugin: Any) -> None:
    _write_config("model:\n  max_tokens: 32768\n")
    request = {"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}]}
    result = plugin.explicit_output_cap_middleware(
        request=request, api_mode="chat_completions", provider="zai")
    assert result is not None and result["request"]["max_tokens"] == 32768  # 期望: 补上配置的 32768
    assert "max_tokens" not in request  # 期望: 原请求不被原地改动(中间件契约为拷贝)
    assert result["source"] == "explicit-output-cap"  # 期望: trace 来源可辨识


def test_existing_cap_wins(plugin: Any) -> None:
    _write_config("model:\n  max_tokens: 32768\n")
    request = {"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000}
    result = plugin.explicit_output_cap_middleware(request=request, provider="zai")
    assert result is None  # 期望: 显式上限(子代理/临时助手设定)不被覆盖


def test_max_completion_tokens_counts_as_cap(plugin: Any) -> None:
    _write_config("model:\n  max_tokens: 32768\n")
    request = {"model": "x", "messages": [{"role": "user", "content": "hi"}],
               "max_completion_tokens": 512}
    result = plugin.explicit_output_cap_middleware(request=request, provider="custom")
    assert result is None  # 期望: 方言键 max_completion_tokens 同样算已有上限


def test_non_chat_api_mode_untouched(plugin: Any) -> None:
    _write_config("model:\n  max_tokens: 32768\n")
    request = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    result = plugin.explicit_output_cap_middleware(
        request=request, api_mode="anthropic_messages", provider="zai")
    assert result is None  # 期望: 非 chat_completions 协议不注入(anthropic 适配器自带上限)


def test_openai_dialect_provider_skipped(plugin: Any) -> None:
    _write_config("model:\n  max_tokens: 32768\n")
    request = {"model": "o-series", "messages": [{"role": "user", "content": "hi"}]}
    result = plugin.explicit_output_cap_middleware(request=request, provider="openai")
    assert result is None  # 期望: openai/azure 跳过(o 系列拒绝顶层 max_tokens,防 400)


def test_absent_config_noop(plugin: Any) -> None:
    _write_config("model:\n  default: glm-5.3\n")
    request = {"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}]}
    result = plugin.explicit_output_cap_middleware(request=request, provider="zai")
    assert result is None  # 期望: 未配置键时零行为变化(插件 no-op)


def test_invalid_config_value_noop(plugin: Any) -> None:
    _write_config("model:\n  max_tokens: big\n")
    request = {"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}]}
    result = plugin.explicit_output_cap_middleware(request=request, provider="zai")
    assert result is None  # 期望: 非正整数值告警并忽略,不注入


def test_request_without_messages_skipped(plugin: Any) -> None:
    _write_config("model:\n  max_tokens: 32768\n")
    result = plugin.explicit_output_cap_middleware(request={"model": "x"}, provider="zai")
    assert result is None  # 期望: 非对话载荷不注入
