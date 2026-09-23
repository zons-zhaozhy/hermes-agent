"""Stable import surface for the read-think-gate plugin's gate module.

The plugin directory uses a hyphen (``plugins/read-think-gate/``, the plugin
loader loads it by path), which is not importable as a package name. Tests and
the guards plugin's second line of defense import the gate through this module
so no consumer needs path-based loading.

This module is the ONE import surface (single source of truth) for the gate
implementation: ``from plugins.read_think_gate_host import ReadThinkGate``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_GATE_PATH = Path(__file__).resolve().parent / "read-think-gate" / "gate.py"


def _load_gate_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plugins._read_think_gate_impl", _GATE_PATH)
    assert spec is not None and spec.loader is not None  # 期望: gate.py 与本文件同仓同存
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module  # 期望: dataclass 等机制需要模块已注册
    spec.loader.exec_module(module)
    return module


_gate = _load_gate_module()

ReadThinkGate = _gate.ReadThinkGate
ReadThinkGateConfig = _gate.ReadThinkGateConfig
GATED_TOOL_NAMES = _gate.GATED_TOOL_NAMES
READ_ONLY_INVESTIGATION_TOOLS = _gate.READ_ONLY_INVESTIGATION_TOOLS
_four_axis_marker_path = _gate._four_axis_marker_path
_build_history_summary = _gate._build_history_summary
_judge_investigation = _gate._judge_investigation
_terminal_writes_file = _gate._terminal_writes_file
detect_complexity = _gate.detect_complexity
_complexity_cache = _gate._complexity_cache

# Module-level attribute proxy for the two classifier seams: ``patch(host,
# "_classify_via_llm")`` writes must reach the implementation module's global
# (gate-internal direct calls resolve there), and reads forward back. A plain
# __getattr__ only covers reads — patch() would park the mock in this module's
# dict where the gate never looks — so the module class is swapped to forward
# BOTH directions.
_FORWARDED_ATTRS = ("_classify_via_llm", "_fallback_detect")


class _HostModule(ModuleType):
    """Host module with two-way forwarding for the classifier seams."""

    def __getattr__(self, name: str) -> object:
        if name in _FORWARDED_ATTRS:
            return getattr(_gate, name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: object) -> None:
        if name in _FORWARDED_ATTRS:
            setattr(_gate, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in _FORWARDED_ATTRS:
            delattr(_gate, name)
        else:
            super().__delattr__(name)


sys.modules[__name__].__class__ = _HostModule  # type: ignore[assignment]
