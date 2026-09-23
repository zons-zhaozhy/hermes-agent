"""
Post-response hooks framework for Hermes Agent.

Provides the shared dataclass and loader used by:
  - plugins/post_response_hooks/  (official plugin lifecycle integration)
  - ~/.hermes/hooks/*.py          (individual hook implementations)

Hook contract (each hook module must expose a ``Hook`` class):

  class Hook:
      @property
      def system_prompt_addition(self) -> str:
          return ""

      def check(self, response: str, context: dict) -> HookResult:
          return HookResult(passed=True)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HookResult:
    """Result of a post-response hook check."""

    passed: bool = True
    action: str = ""
    message: str = ""

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# Hook loader
# ---------------------------------------------------------------------------

_HOOKS_DIR = Path.home() / ".hermes" / "hooks"


def load_hooks(hook_configs: list[dict[str, Any]]) -> list:
    """Load enabled hooks from *hook_configs*.

    Each config entry must have at least:
      - ``enabled`` (bool)
      - ``module``  (str) — filename without ``.py`` under ~/.hermes/hooks/

    Returns a list of instantiated Hook objects.
    """
    hooks: list = []
    for cfg in hook_configs:
        if not cfg.get("enabled", False):
            continue
        module_name = cfg.get("module", "").strip()
        if not module_name:
            logger.warning("Skipping hook with empty module name: %s", cfg)
            continue

        module_path = _HOOKS_DIR / f"{module_name}.py"
        if not module_path.is_file():
            logger.warning(
                "Hook module not found: %s (looked in %s)",
                module_name,
                _HOOKS_DIR,
            )
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                module_name, str(module_path)
            )
            if spec is None or spec.loader is None:
                logger.warning("Cannot create module spec for %s", module_path)
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)

            hook_cls = getattr(mod, "Hook", None)
            if hook_cls is None or not callable(hook_cls):
                logger.warning(
                    "Hook module %s has no callable Hook class", module_name
                )
                continue

            hook_instance = hook_cls()
            hooks.append(hook_instance)
            logger.debug("Loaded hook: %s", module_name)
        except Exception:
            logger.exception("Failed to load hook module: %s", module_path)

    return hooks


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def run_post_response_checks(
    hooks: list,
    response: str,
    context: dict,
) -> HookResult:
    """Run all loaded hooks against a response.

    Returns the first failing HookResult, or a passing HookResult if all
    hooks pass (or no hooks are loaded). Always returns a HookResult —
    callers never need a None check.
    """
    for hook in hooks:
        try:
            result = hook.check(response, context)
            # NOTE: HookResult.__bool__ returns self.passed, so a FAILING
            # result is falsy — `if result and ...` would short-circuit and
            # drop it. Check for None explicitly, then the passed flag.
            if result is not None and not result.passed:
                return result
        except Exception:
            logger.exception("Hook %s.check() failed", type(hook).__name__)

    return HookResult(passed=True)
