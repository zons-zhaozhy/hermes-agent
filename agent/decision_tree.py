"""Decision Tree World Model — structured belief tracking for agent execution.

A persistent, JSON-structured decision tree that the agent maintains across
turns to track what it tried, why it worked/failed, and what it believes
about the problem space. Directly inspired by K-Search's co-evolving
intrinsic world model (arXiv 2602.19128).

Key concepts:
- Decision tree: prefix tree of decisions (root→leaf = one strategy)
- Each node carries: rating (0-10), confidence (0-1), impacts × risk, action
- Sibling nodes are mutually exclusive alternatives (SELF_CHECK enforced)
- Three anti-self-deception checks: FOLLOW_THROUGH, PERF_GAP, SELF_CHECK

Usage (standalone, no Hermes dependencies):
    from agent.decision_tree import WorldModel

    wm = WorldModel(task_name="fix-login-bug")
    wm.init_tree("Fix the login timeout issue on mobile")
    wm.add_action("n1", "Try increasing timeout to 30s",
                  difficulty=2, expected_vs_baseline=1.5)
    wm.attach_result("n1", status="PASSED", latency_ms=120, baseline_ms=200,
                     speedup=1.67)
    wm.update_belief("n1", rating=7, confidence=0.8)
    report = wm.check_self_deception("n1", "I claimed the fix works")
    if report.has_gaps:
        ...
    wm.save()  # persists to ~/.hermes/decision_tree/
"""

from __future__ import annotations

import json
import logging
import os
import re  # noqa: structured text parsing — regex is essential
import datetime
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Impact dimensions (same as K-Search's BASE_DIMENSIONS)
IMPACT_DIMENSIONS: Tuple[str, ...] = ("task_outcome",)

# Default stagnation window: rounds without improvement before forcing branch switch
DEFAULT_STAGNATION_WINDOW = 3

# Maximum new nodes per turn (prevents tree explosion)
MAX_NEW_NODES_PER_TURN = 3

# Node cap — prevent unbounded growth (safety valve)
MAX_TOTAL_NODES = 100


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Impact:
    """Per-dimension belief about how a decision affects a resource."""
    rating: float = 0.0  # 0..10
    risk: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {"rating_0_to_10": self.rating, "risk": self.risk, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "Impact":
        return cls(
            rating=_clamp(float(d.get("rating_0_to_10", 0)), 0, 10),
            risk=str(d.get("risk", "")),
            notes=str(d.get("notes", "")),
        )


@dataclass
class SolutionRef:
    """Link to a concrete code solution and its evaluation result."""
    solution_id: Optional[str] = None
    parent_solution_id: Optional[str] = None
    status: str = ""        # PASSED / FAILED / TIMEOUT / COMPILE_ERROR
    score: Optional[float] = None
    latency_ms: Optional[float] = None
    baseline_ms: Optional[float] = None
    speedup_factor: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "solution_id": self.solution_id,
            "parent_solution_id": self.parent_solution_id,
            "eval": {
                "status": self.status,
                "score": self.score,
                "latency_ms": self.latency_ms,
                "baseline_latency_ms": self.baseline_ms,
                "speedup_factor": self.speedup_factor,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SolutionRef":
        ev = d.get("eval", {}) if isinstance(d.get("eval"), dict) else {}
        return cls(
            solution_id=d.get("solution_id"),
            parent_solution_id=d.get("parent_solution_id"),
            status=str(ev.get("status", "")),
            score=_optional_float(ev.get("score")),
            latency_ms=_optional_float(ev.get("latency_ms")),
            baseline_ms=_optional_float(ev.get("baseline_latency_ms")),
            speedup_factor=_optional_float(ev.get("speedup_factor")),
        )


@dataclass
class Action:
    """A concrete next step the agent can take."""
    title: str = ""
    description: str = ""
    difficulty: int = 3          # 1..5 (1=easy, 5=hard)
    score: float = 0.0           # 0..1  expected reward
    expected_vs_baseline: Optional[float] = None  # >1 means expected improvement
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "difficulty_1_to_5": self.difficulty,
            "score_0_to_1": self.score,
            "expected_vs_baseline_factor": self.expected_vs_baseline,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        diff = int(d.get("difficulty_1_to_5", d.get("difficulty_0_to_3", 3)))
        # Legacy 0..3 → 1..4 mapping
        if "difficulty_0_to_3" in d and "difficulty_1_to_5" not in d:
            diff = int(d.get("difficulty_0_to_3", 2)) + 1
        diff = max(1, min(5, diff))
        return cls(
            title=str(d.get("title", "")).strip(),
            description=str(d.get("description", "")).strip(),
            difficulty=diff,
            score=_clamp(float(d.get("score_0_to_1", 0)), 0, 1),
            expected_vs_baseline=_optional_float(d.get("expected_vs_baseline_factor")),
            rationale=str(d.get("rationale", "")).strip(),
        )


@dataclass
class TreeNode:
    """One node in the decision prefix tree."""
    node_id: str
    parent_id: Optional[str] = None
    decision: Optional[str] = None          # What decision was made (null for root)
    choice: Optional[str] = None            # Which option was chosen
    rating: float = 0.0                     # 0..10
    confidence: float = 0.0                 # 0..1
    impacts: Dict[str, Impact] = field(default_factory=dict)
    solution: Optional[SolutionRef] = None
    action: Optional[Action] = None
    notes: str = ""
    last_updated_round: int = 0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_open(self) -> bool:
        """Open node: has an action title but no attached solution yet."""
        return (
            self.action is not None
            and bool(self.action.title)
            and (self.solution is None or not self.solution.solution_id)
        )

    @property
    def is_filled(self) -> bool:
        """Filled node: has an attached solution."""
        return self.solution is not None and bool(self.solution.solution_id)

    def to_dict(self) -> dict:
        impacts = {k: v.to_dict() for k, v in self.impacts.items()}
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "decision": self.decision,
            "choice": self.choice,
            "overall_rating_0_to_10": self.rating,
            "confidence_0_to_1": self.confidence,
            "impacts": impacts if impacts else {
                k: Impact().to_dict() for k in IMPACT_DIMENSIONS
            },
            "solution_ref": self.solution.to_dict() if self.solution else {
                "solution_id": None, "parent_solution_id": None,
                "eval": {"status": "", "score": None, "latency_ms": None,
                         "baseline_latency_ms": None, "speedup_factor": None},
            },
            "action": self.action.to_dict() if self.action else Action().to_dict(),
            "notes": self.notes,
            "last_updated_round": self.last_updated_round,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TreeNode":
        imp = {}
        for k in IMPACT_DIMENSIONS:
            raw = d.get("impacts", {}).get(k, {})
            if isinstance(raw, dict):
                imp[k] = Impact.from_dict(raw)
            else:
                imp[k] = Impact()
        sol = d.get("solution_ref")
        act = d.get("action")
        return cls(
            node_id=str(d.get("node_id", "")),
            parent_id=d.get("parent_id"),
            decision=d.get("decision"),
            choice=d.get("choice"),
            rating=_clamp(float(d.get("overall_rating_0_to_10", 0)), 0, 10),
            confidence=_clamp(float(d.get("confidence_0_to_1", 0)), 0, 1),
            impacts=imp,
            solution=SolutionRef.from_dict(sol) if isinstance(sol, dict) else None,
            action=Action.from_dict(act) if isinstance(act, dict) else None,
            notes=str(d.get("notes", "")).strip(),
            last_updated_round=int(d.get("last_updated_round", 0)),
        )


@dataclass
class SelfDeceptionReport:
    """Result of running anti-self-deception checks on a node."""
    has_gaps: bool = False
    follow_through_ok: bool = True
    follow_through_detail: str = ""
    performance_gap: Optional[float] = None  # expected vs observed delta
    performance_gap_detail: str = ""
    sibling_conflicts: List[str] = field(default_factory=list)
    zero_confidence_nodes: List[str] = field(default_factory=list)
    stagnation_alert: bool = False
    stagnation_detail: str = ""


# ---------------------------------------------------------------------------
# WorldModel — the main class
# ---------------------------------------------------------------------------

class WorldModel:
    """A persistent decision tree that tracks the agent's beliefs across turns.

    Lifecycle:
        wm = WorldModel(task_name="fix-login")
        wm.init_tree("Fix login timeout in production")   # optional
        wm.add_action(parent_id, title, ...)
        wm.attach_result(node_id, ...)
        wm.check_self_deception(node_id, claimed_intent)
        wm.save()

    The tree is persisted to ~/.hermes/decision_tree/<profile>/<session>.json
    The `task_name` is typically the session id; same task across turns.
    """

    def __init__(
        self,
        task_name: str,
        storage_dir: Optional[Path] = None,
        stagnation_window: int = DEFAULT_STAGNATION_WINDOW,
    ):
        self.task_name = task_name
        self.stagnation_window = stagnation_window
        self._storage_dir = storage_dir or _default_storage_dir()
        self._file_path = self._storage_dir / f"{_safe_filename(task_name)}.json"

        # Internal state
        self.kernel_summary: str = ""
        self.open_questions: List[str] = []
        self.nodes: Dict[str, TreeNode] = {}      # node_id → node
        self.root_id: str = "root"
        self.active_leaf_id: str = "root"
        self.round_index: int = 0
        self.round_results: List[str] = []        # last N results for stagnation detection
        self._next_node_counter: int = 1
        self._loaded: bool = False

    # ---- Persistence -------------------------------------------------------

    def load(self) -> bool:
        """Load existing tree from disk. Returns True if loaded, False if not found."""
        if not self._file_path.exists():
            return False
        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
            self._deserialize(data)
            self._loaded = True
            return True
        except Exception:
            logger.exception("Failed to load decision tree from %s", self._file_path)
            return False

    def save(self) -> None:
        """Persist current tree to disk."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._serialize()
        self._file_path.write_text(json.dumps(data, indent=2, sort_keys=True),
                                   encoding="utf-8")

    def exists(self) -> bool:
        return self._file_path.exists()

    def delete(self) -> None:
        if self._file_path.exists():
            self._file_path.unlink()

    # ---- Initialization ----------------------------------------------------

    def init_tree(self, kernel_summary: str = "", open_questions: List[str] | None = None) -> None:
        """Create a fresh decision tree with root node."""
        self.kernel_summary = kernel_summary
        self.open_questions = open_questions or []
        self.nodes.clear()
        self.round_index = 0
        self.round_results.clear()
        self._next_node_counter = 1

        root = TreeNode(
            node_id="root",
            parent_id=None,
            decision=None,
            choice=None,
            rating=0.0,
            confidence=0.0,
            notes=f"Task: {kernel_summary[:200]}" if kernel_summary else "",
        )
        for dim in IMPACT_DIMENSIONS:
            root.impacts[dim] = Impact()
        self.nodes["root"] = root
        self.root_id = "root"
        self.active_leaf_id = "root"
        self._loaded = True

    # ---- Tree Operations ---------------------------------------------------

    def add_node(
        self,
        parent_id: str,
        decision: str,
        choice: str = "",
        rating: float = 0.0,
        confidence: float = 0.0,
        notes: str = "",
        action: Optional[Action] = None,
    ) -> str:
        """Add a child node. Returns the new node_id."""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id!r} not found")
        if len(self.nodes) >= MAX_TOTAL_NODES:
            raise RuntimeError(f"Decision tree has reached max size ({MAX_TOTAL_NODES} nodes)")

        node_id = self._next_id()
        node = TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            decision=decision,
            choice=choice,
            rating=rating,
            confidence=confidence,
            notes=notes,
            action=action,
            last_updated_round=self.round_index,
        )
        for dim in IMPACT_DIMENSIONS:
            node.impacts[dim] = Impact()
        self.nodes[node_id] = node
        return node_id

    def add_action(
        self,
        parent_id: str,
        title: str,
        description: str = "",
        difficulty: int = 3,
        expected_vs_baseline: Optional[float] = None,
        rationale: str = "",
        confidence: float = 0.0,
    ) -> str:
        """Add an open action node (no solution attached yet)."""
        action = Action(
            title=title,
            description=description,
            difficulty=max(1, min(5, difficulty)),
            expected_vs_baseline=expected_vs_baseline,
            rationale=rationale,
            score=confidence,
        )
        return self.add_node(
            parent_id=parent_id,
            decision="Action planned",
            choice=title[:120],
            rating=confidence * 10 if confidence else 0.0,
            confidence=confidence,
            action=action,
        )

    def attach_result(
        self,
        node_id: str,
        status: str = "",
        score: Optional[float] = None,
        latency_ms: Optional[float] = None,
        baseline_ms: Optional[float] = None,
        speedup: Optional[float] = None,
        solution_id: Optional[str] = None,
    ) -> None:
        """Attach an evaluation result to a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id!r} not found")
        node = self.nodes[node_id]
        node.solution = SolutionRef(
            solution_id=solution_id or f"sol_{uuid.uuid4().hex[:12]}",
            parent_solution_id=_resolve_parent_solution_id(self.nodes, node.parent_id),
            status=status,
            score=score,
            latency_ms=latency_ms,
            baseline_ms=baseline_ms,
            speedup_factor=speedup,
        )
        node.last_updated_round = self.round_index
        self.active_leaf_id = node_id

        # Track result for stagnation detection
        self.round_results.append(status)
        if len(self.round_results) > self.stagnation_window * 3:
            self.round_results = self.round_results[-self.stagnation_window * 3:]

    def update_belief(
        self,
        node_id: str,
        rating: Optional[float] = None,
        confidence: Optional[float] = None,
        notes: str = "",
        action_score: Optional[float] = None,
    ) -> None:
        """Update belief ratings after seeing evidence."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id!r} not found")
        node = self.nodes[node_id]
        if rating is not None:
            node.rating = _clamp(rating, 0, 10)
        if confidence is not None:
            node.confidence = _clamp(confidence, 0, 1)
        if notes:
            existing = node.notes
            node.notes = f"{existing}\n[R{self.round_index}] {notes}" if existing else f"[R{self.round_index}] {notes}"
        if action_score is not None and node.action:
            node.action.score = _clamp(action_score, 0, 1)
        node.last_updated_round = self.round_index

    def set_active_leaf(self, node_id: str) -> None:
        """Switch the active exploration path to a different leaf."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id!r} not found")
        self.active_leaf_id = node_id

    def advance_round(self) -> None:
        """Mark start of a new exploration round."""
        self.round_index += 1

    # ---- Anti-Self-Deception Checks ----------------------------------------

    def check_self_deception(
        self,
        node_id: str,
        claimed_intent: str = "",
    ) -> SelfDeceptionReport:
        """Run all anti-self-deception checks on a node.

        Args:
            node_id: The node to check (typically the one that just got a result)
            claimed_intent: What the agent claims this action achieved,
                            for FOLLOW_THROUGH comparison

        Returns a SelfDeceptionReport with any gaps found.
        """
        report = SelfDeceptionReport()
        if node_id not in self.nodes:
            report.has_gaps = True
            report.follow_through_ok = False
            report.follow_through_detail = f"Node {node_id!r} does not exist"
            return report

        node = self.nodes[node_id]

        # 1. FOLLOW_THROUGH: did the implementation match the intent?
        if claimed_intent and node.action and node.action.title:
            # Extract meaningful words (filter out stop words and common action verbs)
            _STOP_WORDS = frozenset({
                "a", "an", "the", "is", "are", "was", "were", "be", "been",
                "in", "on", "at", "to", "for", "of", "by", "with", "from",
                "it", "its", "and", "or", "not", "no", "that", "this", "i", "we",
            })
            _ACTION_VERBS = frozenset({
                "fix", "try", "test", "check", "add", "remove", "change",
                "update", "set", "get", "make", "use", "run", "do", "implement",
                "increase", "decrease", "enable", "disable", "switch", "replace",
                "refactor", "tweak", "adjust", "optimize",
            })

            def _meaningful_words(text: str) -> set:
                return {w for w in re.findall(r"[a-zA-Z0-9]+", text.lower())
                        if w not in _STOP_WORDS and w not in _ACTION_VERBS}

            intent_words = _meaningful_words(claimed_intent)
            action_words = _meaningful_words(node.action.title)
            if intent_words and action_words:
                overlap = intent_words & action_words
                ratio = len(overlap) / min(len(intent_words), len(action_words))
                if ratio < 0.2:
                    report.follow_through_ok = False
                    report.follow_through_detail = (
                        f"Intent \"{claimed_intent[:100]}\" has low semantic overlap "
                        f"({ratio:.0%}) with action \"{node.action.title[:100]}\". "
                        f"Intent words: {sorted(intent_words)}, action words: {sorted(action_words)}"
                    )
                    report.has_gaps = True
            elif intent_words:
                # Intent has meaningful words but action doesn't → misalignment
                report.follow_through_ok = False
                report.follow_through_detail = (
                    f"Intent \"{claimed_intent[:100]}\" has no meaningful word "
                    f"overlap with action \"{node.action.title[:100]}\""
                )
                report.has_gaps = True
            elif node.solution and node.solution.status == "PASSED":
                # Even if passed, check if the result aligns with what was expected
                if node.action.expected_vs_baseline is not None and node.solution.speedup_factor is not None:
                    delta = node.solution.speedup_factor - node.action.expected_vs_baseline
                    if abs(delta) > 0.3 * abs(node.action.expected_vs_baseline):
                        report.follow_through_detail = (
                            f"Passed but result ({node.solution.speedup_factor:.2f}x) "
                            f"deviates from expected ({node.action.expected_vs_baseline:.2f}x)"
                        )

        # 2. PERF_GAP: expected vs observed
        if (
            node.action
            and node.action.expected_vs_baseline is not None
            and node.solution
            and node.solution.speedup_factor is not None
        ):
            delta = node.solution.speedup_factor - node.action.expected_vs_baseline
            report.performance_gap = delta
            if abs(delta) > 0.2 * abs(node.action.expected_vs_baseline) + 1e-9:
                report.has_gaps = True
                report.performance_gap_detail = (
                    f"PERF_GAP: expected vs_baseline={node.action.expected_vs_baseline:.2f}x, "
                    f"observed speedup={node.solution.speedup_factor:.2f}x, "
                    f"delta={delta:+.2f}x"
                )

        # 3. Sibling conflicts: do any siblings exist without SELF_CHECK?
        if node.parent_id:
            siblings = self.get_children(node.parent_id)
            for sib in siblings:
                if sib.node_id != node_id and "SELF_CHECK" not in sib.notes.upper():
                    report.sibling_conflicts.append(
                        f"Node {sib.node_id!r} lacks SELF_CHECK annotation "
                        f"explaining why it's mutually exclusive with {node_id!r}"
                    )
            if report.sibling_conflicts:
                report.has_gaps = True

        # 4. Zero-confidence nodes (unfilled beliefs)
        for nid, n in self.nodes.items():
            if not n.is_root and n.is_open and n.confidence == 0.0 and n.rating == 0.0:
                report.zero_confidence_nodes.append(nid)
        if report.zero_confidence_nodes:
            report.has_gaps = True

        # 5. Stagnation: N consecutive rounds without improvement
        if len(self.round_results) >= self.stagnation_window:
            recent = self.round_results[-self.stagnation_window:]
            if all(r != "PASSED" for r in recent):
                report.stagnation_alert = True
                report.stagnation_detail = (
                    f"Last {self.stagnation_window} rounds all non-PASSED: {recent}. "
                    f"Consider switching to a different branch."
                )
                report.has_gaps = True

        return report

    # ---- Query -------------------------------------------------------------

    def get_children(self, node_id: str) -> List[TreeNode]:
        """Get direct children of a node."""
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def get_active_path(self) -> List[TreeNode]:
        """Get nodes from root to active_leaf."""
        path = []
        node_id = self.active_leaf_id
        while node_id is not None:
            if node_id in self.nodes:
                path.append(self.nodes[node_id])
                node_id = self.nodes[node_id].parent_id
            else:
                break
        path.reverse()
        return path

    def get_open_actions(self, max_items: int = 10) -> List[TreeNode]:
        """Get open action nodes (action planned, no solution attached yet)."""
        candidates = []
        for n in self.nodes.values():
            if n.is_open:
                # Must have a parent with solution attached (or be root's child)
                if n.parent_id == "root":
                    candidates.append(n)
                elif n.parent_id and n.parent_id in self.nodes:
                    parent = self.nodes[n.parent_id]
                    if parent.is_filled:
                        candidates.append(n)
        # Sort by rating desc, then confidence desc
        candidates.sort(key=lambda n: (-n.rating, -n.confidence, n.node_id))
        return candidates[:max_items]

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self.nodes.get(node_id)

    def get_root(self) -> TreeNode:
        return self.nodes.get(self.root_id, TreeNode(node_id="root"))

    # ---- Serialization -----------------------------------------------------

    def _serialize(self) -> dict:
        nodes_list = []
        # Ensure root is first
        if self.root_id in self.nodes:
            nodes_list.append(self.nodes[self.root_id].to_dict())
        for nid, n in self.nodes.items():
            if nid != self.root_id:
                nodes_list.append(n.to_dict())

        return {
            "task_name": self.task_name,
            "kernel_summary": self.kernel_summary,
            "open_questions": self.open_questions[:8],
            "decision_tree": {
                "root_id": self.root_id,
                "active_leaf_id": self.active_leaf_id,
                "nodes": nodes_list,
            },
            "computed_signals": {
                "round_index": self.round_index,
                "total_nodes": len(self.nodes),
                "stagnation_window": self.stagnation_window,
            },
            "saved_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _deserialize(self, data: dict) -> None:
        self.task_name = data.get("task_name", self.task_name)
        self.kernel_summary = data.get("kernel_summary", "")
        self.open_questions = data.get("open_questions", [])
        dt = data.get("decision_tree", {})
        self.root_id = dt.get("root_id", "root")
        self.active_leaf_id = dt.get("active_leaf_id", "root")
        cs = data.get("computed_signals", {})
        self.round_index = cs.get("round_index", 0)

        self.nodes.clear()
        for raw in dt.get("nodes", []):
            if isinstance(raw, dict):
                node = TreeNode.from_dict(raw)
                self.nodes[node.node_id] = node

        # Recover _next_node_counter
        max_id = 0
        for nid in self.nodes:
            m = re.match(r"n(\d+)", nid)
            if m:
                max_id = max(max_id, int(m.group(1)))
        self._next_node_counter = max_id + 1
        self._loaded = True

    def _next_id(self) -> str:
        """Generate next unique node_id."""
        nid = f"n{self._next_node_counter}"
        self._next_node_counter += 1
        while nid in self.nodes:
            nid = f"n{self._next_node_counter}"
            self._next_node_counter += 1
        return nid

    # ---- Summary -----------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary for logging."""
        open_count = sum(1 for n in self.nodes.values() if n.is_open)
        filled_count = sum(1 for n in self.nodes.values() if n.is_filled)
        return (
            f"WorldModel(task={self.task_name!r}, "
            f"nodes={len(self.nodes)} open={open_count} filled={filled_count}, "
            f"round={self.round_index}, active={self.active_leaf_id})"
        )

    def compact_view(self, max_chars: int = 3000) -> str:
        """Compact JSON projection for prompt injection."""
        nodes_preview = []
        path = self.get_active_path()
        for n in path[:8]:
            nodes_preview.append({
                "node_id": n.node_id,
                "decision": n.decision,
                "choice": n.choice[:80] if n.choice else None,
                "rating": n.rating,
                "conf": f"{n.confidence:.2f}",
                "status": n.solution.status if n.solution else "",
            })
        open_actions = self.get_open_actions(5)
        open_preview = [
            {"node_id": a.node_id, "title": a.action.title[:80] if a.action else "",
             "difficulty": a.action.difficulty if a.action else "?"}
            for a in open_actions
        ]

        result = {
            "task": self.task_name,
            "kernel_summary": self.kernel_summary[:200],
            "round": self.round_index,
            "active_path": nodes_preview,
            "open_actions": open_preview,
            "open_questions": self.open_questions[:5],
        }
        raw = json.dumps(result, indent=2, ensure_ascii=False)
        return raw[:max_chars]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _optional_float(value: Any) -> Optional[float]:
    """Parse to float, return None if unparseable/no value."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_filename(name: str) -> str:
    """Sanitize a task name into a safe filename."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)[:120]


def _default_storage_dir() -> Path:
    """Get the default storage directory from HERMES_HOME env."""
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(home) / "decision_tree"


def _resolve_parent_solution_id(nodes: Dict[str, TreeNode], parent_id: Optional[str]) -> Optional[str]:
    """Safely get parent node's solution_id."""
    if not parent_id or parent_id not in nodes:
        return None
    parent = nodes[parent_id]
    if parent.solution and parent.solution.solution_id:
        return parent.solution.solution_id
    return None


# ---------------------------------------------------------------------------
# Module-level API — convenient for scripting
# ---------------------------------------------------------------------------

def load_or_create(
    task_name: str,
    storage_dir: Optional[Path] = None,
    stagnation_window: int = DEFAULT_STAGNATION_WINDOW,
) -> WorldModel:
    """Load existing tree or create a new one."""
    wm = WorldModel(
        task_name=task_name,
        storage_dir=storage_dir,
        stagnation_window=stagnation_window,
    )
    if not wm.load():
        wm.init_tree()
    return wm
