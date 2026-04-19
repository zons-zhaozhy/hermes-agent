"""Post-response hook framework for Hermes Agent.

Loads user-defined hooks from ~/.hermes/hooks/ that can:
  1. Inject additions into the system prompt (pre-response guidance)
  2. Inspect responses and block or request re-generation (post-response gate)

Hook Interface
--------------
Each hook is a .py file in ~/.hermes/hooks/ that exposes a Hook class:

    class Hook:
        @property
        def system_prompt_addition(self) -> str:
            '''Optional text appended to system prompt.'''

        def check(self, response: str, context: dict) -> HookResult:
            '''Return HookResult to pass, block, or nudge.'''

HookResult
----------
    @dataclass
    class HookResult:
        passed: bool          # True = let response through
        action: str = ""      # "block" | "nudge" | ""  (meaningful only when passed=False)
        message: str = ""     # block: refusal text | nudge: corrective hint

Actions:
  - passed=True: response is delivered normally
  - passed=False + action="block": response replaced with message, no re-generation
  - passed=False + action="nudge": response discarded, model re-generates with message as hint
  - passed=True + side effects: passive mode (logging, scoring, audit)

Configuration
-------------
config.yaml:

    agent:
      post_response_hooks:
        - module: pii_filter
          enabled: true
        - module: audit_log
          enabled: true
"""

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HOOKS_DIR = Path.home() / ".hermes" / "hooks"


# ── Core types ────────────────────────────────────────────────────────

@dataclass
class HookResult:
    """Result of a post-response hook check.

    Attributes:
        passed: True if the response is acceptable.
        action: What to do when passed=False. "block" replaces the response
                without re-generation. "nudge" triggers one re-generation.
        message: For block: the replacement text shown to user.
                 For nudge: the corrective hint injected into the conversation.
    """
    passed: bool
    action: str = ""     # "block" | "nudge" | ""
    message: str = ""


# ── HookSpec: wraps a loaded hook with config ─────────────────────────

class HookSpec:
    """Loaded and validated hook instance with its config."""

    def __init__(self, module_name: str, hook_instance: Any, enabled: bool = True):
        self.module_name = module_name
        self.hook = hook_instance
        self.enabled = enabled

    @property
    def system_prompt_addition(self) -> str:
        try:
            return getattr(self.hook, "system_prompt_addition", "")
        except Exception:
            return ""

    def check(self, response: str, context: dict) -> HookResult:
        """Run hook.check(), wrapping legacy bool returns in HookResult."""
        try:
            result = self.hook.check(response, context)
            # Backward compat: hooks returning bool are treated as nudge-style
            if isinstance(result, bool):
                return HookResult(passed=result, action="nudge" if not result else "")
            if isinstance(result, HookResult):
                return result
            # Unknown return type — treat as pass
            logger.warning(
                "Hook %s.check() returned unexpected type %s, treating as passed",
                self.module_name, type(result).__name__,
            )
            return HookResult(passed=True)
        except Exception as exc:
            logger.error("Hook %s.check() raised: %s", self.module_name, exc)
            return HookResult(passed=True)  # fail-open


# ── Module loading ────────────────────────────────────────────────────

def _load_hook_module(module_name: str) -> Optional[Any]:
    """Import a hook .py file from ~/.hermes/hooks/ and return its Hook class instance."""
    hook_path = HOOKS_DIR / f"{module_name}.py"
    if not hook_path.exists():
        logger.error("Hook module not found: %s", hook_path)
        return None

    try:
        spec = importlib.util.spec_from_file_location(
            f"hermes_hooks.{module_name}", str(hook_path)
        )
        if spec is None or spec.loader is None:
            logger.error("Cannot create module spec for %s", hook_path)
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(f"hermes_hooks.{module_name}", module)
        spec.loader.exec_module(module)

        # Look for a Hook class (preferred) or module-level functions
        if hasattr(module, "Hook"):
            return module.Hook()

        # Fallback: wrap module-level callables into a duck-typed object
        if hasattr(module, "check"):
            return _ModuleShim(module)

        logger.error("Hook %s has no Hook class or check() function", module_name)
        return None

    except Exception as exc:
        logger.error("Failed to load hook %s: %s", module_name, exc, exc_info=True)
        return None


class _ModuleShim:
    """Adapts module-level callables to the Hook interface."""

    def __init__(self, module):
        self._mod = module

    @property
    def system_prompt_addition(self) -> str:
        fn = getattr(self._mod, "system_prompt_addition", None)
        if callable(fn):
            return fn()
        return getattr(self._mod, "system_prompt_addition", "")

    def check(self, response: str, context: dict) -> HookResult:
        result = self._mod.check(response, context)
        if isinstance(result, bool):
            return HookResult(passed=result)
        return result


# ── Public API ────────────────────────────────────────────────────────

def load_hooks(hook_configs: List[Dict[str, Any]]) -> List[HookSpec]:
    """Load all configured hooks from ~/.hermes/hooks/.

    Args:
        hook_configs: List of dicts with 'module' and optional 'enabled' keys.

    Returns:
        List of HookSpec instances (only successfully loaded ones).
    """
    hooks: List[HookSpec] = []
    for cfg in hook_configs:
        module_name = cfg.get("module", "")
        if not module_name:
            continue
        enabled = cfg.get("enabled", True)
        instance = _load_hook_module(module_name)
        if instance is not None:
            hooks.append(HookSpec(module_name, instance, enabled))
            logger.info("Loaded post-response hook: %s (enabled=%s)", module_name, enabled)
    return hooks


def build_system_prompt_additions(hooks: List[HookSpec]) -> str:
    """Aggregate system prompt additions from all enabled hooks."""
    parts = []
    for h in hooks:
        if not h.enabled:
            continue
        addition = h.system_prompt_addition
        if addition:
            parts.append(addition)
    return "\n\n".join(parts)


def run_post_response_checks(
    hooks: List[HookSpec],
    response: str,
    context: dict,
) -> Optional[HookResult]:
    """Run all enabled hooks' check() in order.

    Returns:
        HookResult of the first hook that does not pass (action=block or nudge),
        or None if all hooks pass.
    """
    for h in hooks:
        if not h.enabled:
            continue
        result = h.check(response, context)
        if not result.passed:
            if result.action:
                # Has action (block/nudge) — trigger regardless of message content
                logger.info(
                    "Post-response hook '%s' triggered %s (len=%d)",
                    h.module_name, result.action, len(response),
                )
                return result
            if result.message:
                # Has message but no action — default to block
                logger.info(
                    "Post-response hook '%s' triggered with message only, defaulting to block (len=%d)",
                    h.module_name, len(response),
                )
                return HookResult(passed=False, action="block", message=result.message)
            # No action and no message — treat as pass (silent fail)
            logger.warning(
                "Hook '%s' returned passed=False with no action/message, ignoring",
                h.module_name,
            )
    return None
