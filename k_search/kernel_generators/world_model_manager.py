"""Orchestrates world model lifecycle (init/refine/merge) for kernel generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from k_search.tasks.task_base import EvalResult
from .world_model import (
    ActionCandidate,
    ActionRanking,
    DecisionTreeEditOps,
    Prediction,
    build_decision_tree_edit_prompt,
    dump_world_model_obj,
    load_world_model_obj,
    merge_computed_signals,
    try_parse_decision_tree_edit_ops,
    build_world_model_prompts,
    try_parse_world_model_json,
)


LLMCall = Callable[[str], str]

def _extract_reference_from_definition_text(definition_text: str) -> str:
    """
    Best-effort extraction of the reference implementation from a rendered definition text.
    This preserves the old behavior of embedding reference excerpts into WM root notes
    without requiring access to a flashinfer-bench Definition object.
    """
    s = str(definition_text or "")
    marker = "Reference Implementation:"
    if marker not in s:
        return ""
    # Take everything after the first marker.
    parts = s.split(marker, 1)
    if len(parts) != 2:
        return ""
    return parts[1].lstrip("\n").rstrip()

@dataclass
class WorldModelSelectionPolicy:
    """
    Configurable, deterministic selection policy for choosing the next frontier action node.

    We aim to avoid hardcoded if/else ladders by using:
    - a small set of constraints (difficulty gating)
    - a single utility function with weights
    """

    # Hard constraint: prefer executing actions with difficulty <= max_difficulty_1_to_5 when possible.
    max_difficulty_1_to_5: int = 4
    # Allow slightly harder actions once the best observed vs_base is strong enough.
    relax_difficulty_if_best_vs_base_ge: float = 0.5
    relaxed_max_difficulty_1_to_5: int = 4

    # Utility weights (higher => more important). Utility is maximized.
    w_score: float = 3.0
    w_difficulty: float = 2.5
    w_depth: float = 0.75
    w_parent_quality: float = 1.5
    w_overall_rating: float = 0.5
    w_confidence: float = 0.25
    # Small positive bias to keep exploring root branches occasionally (but not dominate).
    w_root_explore: float = 0.15


@dataclass
class WorldModelConfig:
    enabled: bool = True
    max_chars_per_block: int = 6000
    # Guardrail: cap number of NEW nodes (insert+split children) applied per WM edit script.
    max_new_nodes_per_edit: int = 3
    # Deterministic selection policy for choosing next action.
    selection_policy: WorldModelSelectionPolicy = field(default_factory=WorldModelSelectionPolicy)


class WorldModelManager:
    """Keeps a per-definition world model JSON blob and refines it across rounds."""

    def __init__(
        self,
        *,
        llm_call: LLMCall,
        target_gpu: str,
        language: str,
        config: WorldModelConfig | None = None,
    ):
        self._llm_call = llm_call
        self._target_gpu = target_gpu
        self._language = language
        self._cfg = config or WorldModelConfig()
        self._world_models: Dict[str, str] = {}
        # Debug/reporting: last edit-ops application summary (applied vs skipped).
        self._last_apply_ops_report: dict | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def get(self, definition_name: str) -> Optional[str]:
        return self._world_models.get(definition_name)

    def set(self, definition_name: str, world_model_json: str) -> None:
        s = (world_model_json or "").strip()
        if not s:
            return
        self._world_models[definition_name] = s

    def ensure_initialized(
        self,
        *,
        definition_name: str,
        definition_text: str,
        reference_text: Optional[str] = None,
        current_code_excerpt: Optional[str] = None,
        eval_result: Optional[EvalResult] = None,
        seed_root_solution_id: Optional[str] = None,
        seed_root_solution_name: Optional[str] = None,
        seed_root_round_index: Optional[int] = None,
    ) -> Optional[str]:
        if not self._cfg.enabled:
            return None
        name = str(definition_name or "").strip()
        if not name:
            return None
        definition_text = str(definition_text or "")
        if reference_text is None:
            reference_text = _extract_reference_from_definition_text(definition_text or "")

        existing = self.get(name)
        if existing:
            # Optional: if caller provides a seed solution, deterministically attach it to root
            # (best-effort) even when the WM already exists.
            if seed_root_solution_id and seed_root_solution_name and eval_result is not None:
                try:
                    obj = load_world_model_obj(existing or "")
                    if isinstance(obj, dict):
                        dt = obj.get("decision_tree")
                        if isinstance(dt, dict):
                            root_id = str(dt.get("root_id", "") or "root")
                            nodes = dt.get("nodes")
                            if isinstance(nodes, list):
                                # Find root node and attach if missing.
                                for n in nodes:
                                    if not isinstance(n, dict):
                                        continue
                                    if str(n.get("node_id") or "") != root_id:
                                        continue
                                    sr = n.get("solution_ref")
                                    sr = sr if isinstance(sr, dict) else {}
                                    sid0 = sr.get("solution_id")
                                    if isinstance(sid0, str) and sid0.strip():
                                        break
                                    n["solution_ref"] = {
                                        "solution_id": str(seed_root_solution_id),
                                        "solution_name": str(seed_root_solution_name),
                                        "parent_solution_id": None,
                                        "eval": eval_result.to_dict(include_log_excerpt=False),
                                        "round_index": (
                                            int(seed_root_round_index) if isinstance(seed_root_round_index, int) else None
                                        ),
                                    }
                                    # Also ensure active points to root in this seed mode.
                                    dt["active_leaf_id"] = root_id
                                    updated = dump_world_model_obj(obj) or existing
                                    self.set(name, updated)
                                    return updated
                except Exception:
                    pass
            return existing

        prompts = build_world_model_prompts(
            definition_text=definition_text,
            target_gpu=self._target_gpu,
            language=self._language,
            previous_world_model_json=None,
            current_code_excerpt=current_code_excerpt,
            eval_result=eval_result,
            chosen_action_text=None,
            prediction=None,
            max_chars_per_block=self._cfg.max_chars_per_block,
        )
        raw = (self._llm_call(prompts.init_prompt) or "").strip()
        parsed = try_parse_world_model_json(raw)
        if parsed:
            # Persist a bounded excerpt of the reference implementation into the root node's notes.
            # Root stays a dummy decision/choice=null node; this is just a stable "anchor" for humans and WM.
            try:
                parsed = self._maybe_embed_reference_into_root_notes_from_text(
                    reference_text=str(reference_text or ""),
                    world_model_json=parsed,
                )
            except Exception:
                pass
            # Optional: seed root solution deterministically (do not rely on LLM edits).
            if seed_root_solution_id and seed_root_solution_name and eval_result is not None:
                try:
                    obj = load_world_model_obj(parsed or "")
                    if isinstance(obj, dict):
                        dt = obj.get("decision_tree")
                        if isinstance(dt, dict):
                            root_id = str(dt.get("root_id", "") or "root")
                            nodes = dt.get("nodes")
                            if isinstance(nodes, list):
                                for n in nodes:
                                    if not isinstance(n, dict):
                                        continue
                                    if str(n.get("node_id") or "") != root_id:
                                        continue
                                    n["solution_ref"] = {
                                        "solution_id": str(seed_root_solution_id),
                                        "solution_name": str(seed_root_solution_name),
                                        "parent_solution_id": None,
                                        "eval": eval_result.to_dict(include_log_excerpt=False),
                                        "round_index": (
                                            int(seed_root_round_index) if isinstance(seed_root_round_index, int) else None
                                        ),
                                    }
                                    dt["active_leaf_id"] = root_id
                                    parsed = dump_world_model_obj(obj) or parsed
                                    break
                except Exception:
                    pass
            self.set(name, parsed)
            return parsed
        return None

    def _maybe_embed_reference_into_root_notes_from_text(
        self, *, reference_text: str, world_model_json: str
    ) -> str:
        """
        Store a bounded excerpt of the reference implementation into decision_tree root node notes.
        This makes the 'Reference Implementation' persist in WM (not just in transient prompts).
        """
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return world_model_json
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return world_model_json
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return world_model_json
        root_id = str(dt.get("root_id", "") or "root")
        ref = str(reference_text or "").strip()
        if not ref:
            return world_model_json
        # Bounded excerpt for logs/prompts.
        max_chars = 1800
        excerpt = ref if len(ref) <= max_chars else (ref[: max_chars] + "\n...<reference truncated>...\n")
        for n in nodes:
            if isinstance(n, dict) and str(n.get("node_id") or "") == root_id:
                notes = str(n.get("notes", "") or "")
                if "Reference Implementation" in notes:
                    return world_model_json
                if notes.strip():
                    n["notes"] = notes.rstrip() + "\n\nReference Implementation (excerpt):\n" + excerpt
                else:
                    n["notes"] = "Reference Implementation (excerpt):\n" + excerpt
                break
        return dump_world_model_obj(obj) or world_model_json

    def refine(
        self,
        *,
        definition_name: str,
        definition_text: str,
        reference_text: Optional[str] = None,
        chosen_action_text: Optional[str],
        current_code_excerpt: Optional[str],
        current_tree_path: Optional[str],
        eval_result: Optional[EvalResult],
        prediction: Optional[Prediction],
        round_index: Optional[int] = None,
    ) -> Optional[str]:
        if not self._cfg.enabled:
            return None
        # Policy: refine ONLY on new successful eval datapoints with performance fields.
        # Failed runs (or missing perf): do not update and do not force initialization.
        name = str(definition_name or "").strip()
        if not name:
            return None
        prev_existing = self.get(name)
        try:
            status_key = (eval_result.status if eval_result else "").strip().lower()
            has_perf = bool(
                eval_result
                and (
                    isinstance(eval_result.latency_ms, (int, float))
                    or isinstance(eval_result.speedup_factor, (int, float))
                )
            )
            if status_key != "passed" or not has_perf:
                return prev_existing
        except Exception:
            return prev_existing

        definition_text = str(definition_text or "")
        if reference_text is None:
            reference_text = _extract_reference_from_definition_text(definition_text or "")

        prev = self.ensure_initialized(definition_name=name, definition_text=definition_text, reference_text=reference_text)

        # Identify the node we just attached a PASSED solution to (the active leaf at refine start).
        # We enforce that this solved node gets at least one OPEN child action node (a continuation step),
        # otherwise the search can "stall" and jump back to unrelated root-level actions.
        solved_parent_id: Optional[str] = None
        prev_obj_for_validation: Optional[dict] = None
        try:
            prev_obj = load_world_model_obj(prev or "")
            if isinstance(prev_obj, dict):
                prev_obj_for_validation = prev_obj
                dt0 = prev_obj.get("decision_tree")
                if isinstance(dt0, dict):
                    solved_parent_id = str(dt0.get("active_leaf_id") or "").strip() or None
        except Exception:
            solved_parent_id = None
            prev_obj_for_validation = None

        def _has_open_child_action(*, world_model_json: str, parent_id: str) -> bool:
            obj = load_world_model_obj(world_model_json or "")
            if not isinstance(obj, dict):
                return False
            dt = obj.get("decision_tree")
            if not isinstance(dt, dict):
                return False
            nodes = dt.get("nodes")
            if not isinstance(nodes, list):
                return False

            def _difficulty_1_to_5(n: dict) -> Optional[int]:
                act = n.get("action") if isinstance(n.get("action"), dict) else {}
                d = act.get("difficulty_1_to_5", None)
                if d is None:
                    d = act.get("difficulty_0_to_3", None)
                    try:
                        d = (int(d) + 1) if d is not None else None
                    except Exception:
                        d = None
                try:
                    di = int(d) if d is not None else None
                except Exception:
                    di = None
                if di is None:
                    return None
                if di < 1:
                    di = 1
                if di > 5:
                    di = 5
                return di

            for n in nodes:
                if not isinstance(n, dict):
                    continue
                if str(n.get("parent_id") or "") != parent_id:
                    continue
                sr = n.get("solution_ref")
                if isinstance(sr, dict):
                    sid = sr.get("solution_id")
                    if isinstance(sid, str) and sid.strip():
                        continue
                act = n.get("action")
                if not (isinstance(act, dict) and str(act.get("title") or "").strip()):
                    continue
                # Continuation constraint: the OPEN child action must be executable (diff < 5).
                di = _difficulty_1_to_5(n)
                if di is None:
                    continue
                if di < 5:
                    return True
            return False

        def _validate_refine_edit(
            *,
            prev_world_model_json: str,
            prev_obj: Optional[dict],
            candidate_world_model_json: str,
            edits: Optional[DecisionTreeEditOps],
            solved_parent_id: Optional[str],
        ) -> list[str]:
            """
            Post-refine validation. These are "hard requirements" on the model's edit-ops:
            1) Ops must be parseable and use only supported op kinds.
            2) Root invariants: root decision/choice must not be changed.
            4) Do not drop/overwrite existing attached solutions.
            5) Continuation: after PASSED attach, solved active node must have >=1 OPEN child action node with diff < 5.
            """
            errors: list[str] = []
            # (1) ops parse and kinds
            if edits is None or not isinstance(edits.ops, list):
                errors.append("edit_ops_parse_failed: model output did not parse into {ops:[...]} JSON")
                return errors
            allowed_kinds = {"update_node", "insert_node", "split_node", "delete_node"}
            for i, op in enumerate(edits.ops):
                if not isinstance(op, dict):
                    errors.append(f"op[{i}]: not an object")
                    continue
                kind = str(op.get("op", "") or "").strip()
                if kind not in allowed_kinds:
                    errors.append(f"op[{i}]: unknown op kind '{kind}'")

            # Parse prev/candidate world model objects for invariants
            prev_obj_local = prev_obj if isinstance(prev_obj, dict) else load_world_model_obj(prev_world_model_json or "")
            cand_obj = load_world_model_obj(candidate_world_model_json or "")
            if not isinstance(prev_obj_local, dict) or not isinstance(cand_obj, dict):
                errors.append("world_model_parse_failed: prev/candidate not parseable JSON objects")
                return errors
            prev_dt = prev_obj_local.get("decision_tree")
            cand_dt = cand_obj.get("decision_tree")
            if not isinstance(prev_dt, dict) or not isinstance(cand_dt, dict):
                errors.append("missing_decision_tree: decision_tree must exist")
                return errors

            prev_nodes = prev_dt.get("nodes")
            cand_nodes = cand_dt.get("nodes")
            if not isinstance(prev_nodes, list) or not isinstance(cand_nodes, list):
                errors.append("missing_nodes: decision_tree.nodes must be a list")
                return errors

            prev_root_id = str(prev_dt.get("root_id", "") or "root")
            cand_root_id = str(cand_dt.get("root_id", "") or "root")
            if prev_root_id != cand_root_id:
                errors.append("root_id_changed: root_id must not change")

            def _by_id(nodes: list) -> dict[str, dict]:
                out: dict[str, dict] = {}
                for n in nodes:
                    if isinstance(n, dict) and n.get("node_id"):
                        out[str(n["node_id"])] = n
                return out

            prev_by_id = _by_id(prev_nodes)
            cand_by_id = _by_id(cand_nodes)
            prev_root = prev_by_id.get(prev_root_id)
            cand_root = cand_by_id.get(prev_root_id)
            if not isinstance(prev_root, dict) or not isinstance(cand_root, dict):
                errors.append("missing_root_node: root node must exist in nodes")
            else:
                # (2) root invariants: don't change decision/choice
                if (prev_root.get("decision") != cand_root.get("decision")) or (prev_root.get("choice") != cand_root.get("choice")):
                    errors.append("root_decision_choice_changed: root decision/choice must not be edited")

            # Helper: solution_id extraction
            def _sid(n: dict) -> Optional[str]:
                sr = n.get("solution_ref")
                if not isinstance(sr, dict):
                    return None
                v = sr.get("solution_id")
                return str(v).strip() if isinstance(v, str) and v.strip() else None

            # (4) preserve attached solutions
            prev_solved: dict[str, str] = {}
            for nid, n in prev_by_id.items():
                sid = _sid(n) if isinstance(n, dict) else None
                if sid:
                    prev_solved[nid] = sid
            for nid, sid in prev_solved.items():
                cn = cand_by_id.get(nid)
                if not isinstance(cn, dict):
                    errors.append(f"solution_node_missing: node_id={nid} with solution_id={sid} disappeared")
                    continue
                csid = _sid(cn)
                if csid != sid:
                    errors.append(f"solution_id_changed: node_id={nid} expected solution_id={sid} got {csid}")

            # (5) continuation rule enforcement (post-state)
            if solved_parent_id:
                if not _has_open_child_action(world_model_json=candidate_world_model_json or "", parent_id=solved_parent_id):
                    errors.append(
                        f"continuation_missing: solved active node '{solved_parent_id}' must have >=1 OPEN child action node with diff < 5"
                    )

            # (7) Avoid silent skips: treat any skipped ops as validation failure (so we retry).
            try:
                rep = getattr(self, "_last_apply_ops_report", None)
                if isinstance(rep, dict):
                    for k in [
                        "skipped_cap",
                        "skipped_missing_parent",
                        "skipped_parent_solution_id_mismatch",
                        "skipped_update_missing_node",
                        "skipped_insert_invalid_node",
                        "skipped_split_invalid_children",
                        "skipped_delete_not_allowed",
                    ]:
                        if int(rep.get(k, 0) or 0) > 0:
                            errors.append(f"edit_ops_skipped: {k}={int(rep.get(k, 0) or 0)}")
            except Exception:
                pass

            return errors

        def _fallback_insert_min_child(*, world_model_json: str, parent_id: str) -> str:
            """
            Deterministic fallback to avoid stalling: insert a minimal OPEN action child under `parent_id`.
            This is intentionally generic but still single-iteration implementable.
            """
            obj = load_world_model_obj(world_model_json or "")
            if not isinstance(obj, dict):
                return world_model_json
            dt = obj.get("decision_tree")
            if not isinstance(dt, dict):
                return world_model_json
            nodes = dt.get("nodes")
            if not isinstance(nodes, list):
                return world_model_json
            by_id: dict[str, dict] = {}
            for n in nodes:
                if isinstance(n, dict) and n.get("node_id"):
                    by_id[str(n["node_id"])] = n
            parent = by_id.get(parent_id)
            if not isinstance(parent, dict):
                return world_model_json
            # Capture parent solution_id if present so downstream constraints are satisfied.
            parent_sol = None
            srp = parent.get("solution_ref")
            if isinstance(srp, dict):
                ps = srp.get("solution_id")
                parent_sol = str(ps).strip() if isinstance(ps, str) and ps.strip() else None

            # `refine()` may be called with definition=None; use resolved name for stable ids.
            safe_def = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(name or ""))[:48] or "def"
            rid = "r" + str(round_index) if round_index is not None else "rX"
            # Ensure uniqueness.
            counter = 0
            while True:
                counter += 1
                nid = f"node_{safe_def}_{rid}_cont_{counter}"
                if nid not in by_id:
                    break

            child = {
                "node_id": nid,
                "parent_id": parent_id,
                "decision": "Continue",
                "choice": "Next-step refinement",
                "overall_rating_0_to_10": max(0.0, float(parent.get("overall_rating_0_to_10") or 0.0)),
                "confidence_0_to_1": max(0.2, float(parent.get("confidence_0_to_1") or 0.2)),
                "notes": "AUTO_CONTINUATION: add a concrete next-step action under a solved node to keep the search moving.\n"
                "Focus on one operation or one tensor axis; refine based on the latest eval bottleneck signal.",
                "impacts": parent.get("impacts") if isinstance(parent.get("impacts"), dict) else {},
                "action": {
                    "title": "Next-step refinement: target the top bottleneck operation",
                    "description": "Propose one small change that targets the primary bottleneck operation indicated by the latest eval (e.g., reduce sync, reduce redundant loads, or improve locality), without changing the overall mapping plan.",
                    "rationale": "The parent design is working; this creates a clear single-iteration follow-up to improve it.",
                    "score_0_to_1": 0.35,
                    "difficulty_1_to_5": 2,
                },
                "solution_ref": {"solution_id": None, "parent_solution_id": parent_sol, "eval": None},
                "last_updated_round": round_index,
            }
            nodes.append(child)
            return dump_world_model_obj(obj) or world_model_json

        def _render_wm_status_for_prompt(*, world_model_json: str) -> str:
            """
            Bounded, prompt-friendly status to help the model reason about:
            - best-known solution node/path
            - how much of the tree is explored
            - which open actions are executable (frontier) vs blocked
            """
            obj = load_world_model_obj(world_model_json or "")
            if not isinstance(obj, dict):
                return "WM status: (unparseable)"
            dt = obj.get("decision_tree")
            if not isinstance(dt, dict):
                return "WM status: (no decision_tree)"
            nodes = dt.get("nodes")
            nodes = nodes if isinstance(nodes, list) else []
            root_id = str(dt.get("root_id", "") or "root")
            active_id = str(dt.get("active_leaf_id", "") or root_id)

            def _sid(n: dict) -> Optional[str]:
                sr = n.get("solution_ref")
                if not isinstance(sr, dict):
                    return None
                v = sr.get("solution_id")
                return str(v).strip() if isinstance(v, str) and v.strip() else None

            def _perf_str_from_eval(ev: dict) -> str:
                if not isinstance(ev, dict):
                    return "(no-eval)"
                mvb = ev.get("mean_vs_baseline_factor")
                sp = ev.get("speedup_factor")
                lat = ev.get("latency_ms")
                parts: list[str] = []
                if isinstance(mvb, (int, float)) and float(mvb) > 0:
                    parts.append(f"vs_base={float(mvb):.3g}x")
                elif isinstance(sp, (int, float)) and float(sp) > 0:
                    parts.append(f"speedup={float(sp):.3g}x")
                if isinstance(lat, (int, float)) and float(lat) > 0:
                    parts.append(f"lat={float(lat):.3g}ms")
                return " ".join(parts) if parts else "(no-perf)"

            # Basic exploration stats (kept minimal; do not rely on node ids alone).
            attached = 0
            for n in nodes:
                if isinstance(n, dict) and _sid(n):
                    attached += 1

            # Best known node by mean_vs_baseline_factor (preferred) else speedup_factor
            best_nid = None
            best_score = -1.0
            best_latency = None
            best_eval: Optional[dict] = None
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                sr = n.get("solution_ref")
                if not isinstance(sr, dict):
                    continue
                ev = sr.get("eval")
                if not isinstance(ev, dict):
                    continue
                status = str(ev.get("status") or "").strip().lower()
                if status != "passed":
                    continue
                mvb = ev.get("mean_vs_baseline_factor")
                sp = ev.get("speedup_factor")
                score = None
                if isinstance(mvb, (int, float)) and float(mvb) > 0:
                    score = float(mvb)
                elif isinstance(sp, (int, float)) and float(sp) > 0:
                    score = float(sp)
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_nid = str(n.get("node_id") or "") or None
                    lat = ev.get("latency_ms")
                    best_latency = float(lat) if isinstance(lat, (int, float)) else None
                    best_eval = ev

            # Under-explored: root children with no attached solution in their subtree.
            by_id: dict[str, dict] = {}
            children: dict[str, list[str]] = {}
            for n in nodes:
                if not (isinstance(n, dict) and n.get("node_id")):
                    continue
                nid = str(n["node_id"])
                by_id[nid] = n
                pid = n.get("parent_id")
                if pid is not None:
                    children.setdefault(str(pid), []).append(nid)

            def _subtree_has_solution(start_id: str) -> bool:
                stack = [start_id]
                seen: set[str] = set()
                while stack:
                    cur = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    node = by_id.get(cur)
                    if isinstance(node, dict) and _sid(node):
                        return True
                    for ch in children.get(cur, []):
                        stack.append(ch)
                return False

            under_explored: list[str] = []
            for nid in children.get(root_id, []):
                if not _subtree_has_solution(nid):
                    n = by_id.get(nid, {})
                    label = str(n.get("choice") or "").strip() or str(n.get("decision") or "").strip() or nid
                    under_explored.append(f"{nid}:{label}")
            under_explored = under_explored[:4]

            lines: list[str] = []
            # Show paths, not just ids.
            try:
                active_path = self.get_tree_path_text(definition_name=name, node_id=active_id)
                if active_path.strip():
                    lines.append("- active_path:")
                    lines.append(active_path)
            except Exception:
                lines.append(f"- active_leaf_id={active_id}")
            if best_nid:
                lines.append(f"- best_node: node_id={best_nid} perf={_perf_str_from_eval(best_eval or {})}")
                try:
                    best_path = self.get_tree_path_text(definition_name=name, node_id=best_nid)
                    if best_path.strip():
                        lines.append("- best_path:")
                        lines.append(best_path)
                except Exception:
                    pass
            else:
                lines.append("- best_node_id=(none yet)")
            if under_explored:
                lines.append("- under_explored_root_children (no attached solution in subtree): " + ", ".join(under_explored))
            return "\n".join(lines).strip()

        # Request an ops-only edit script and apply deterministically.
        # Hard validation: after attaching a PASSED solution, ensure the solved node has >=1 OPEN child action.
        # If not, retry refine edits a few times to force the continuation step to appear.
        candidate = prev
        max_refine_retries = 3
        last_validation_errors: list[str] = []
        for attempt in range(max_refine_retries):
            # Provide an explicit retry note to steer the model if validation failed previously.
            chosen_action_for_prompt = chosen_action_text
            if attempt > 0 and last_validation_errors:
                # Keep this compact to preserve token budget.
                err_lines = "\n".join(f"- {e}" for e in last_validation_errors[:12])
                suffix = (
                    "\n\n[REFINE_VALIDATION_FAILED]\n"
                    "Your previous edit script violated hard requirements. Fix them in this retry.\n"
                    "Errors:\n"
                    f"{err_lines}\n"
                    "Do NOT output commentary; output JSON edit script only."
                )
                chosen_action_for_prompt = (str(chosen_action_for_prompt or "").strip() + suffix).strip()

            edit_prompt = build_decision_tree_edit_prompt(
                world_model_json=candidate,
                definition_text=definition_text,
                baseline_targets_text=None,
                debug_and_improve_round=None,
                debug_and_improve_max_rounds=5,
                target_gpu=self._target_gpu,
                language=self._language,
                current_code_excerpt=current_code_excerpt,
                current_tree_path=current_tree_path,
                wm_status_text=_render_wm_status_for_prompt(world_model_json=candidate or ""),
                open_frontier_nodes_text=self._render_open_frontier_nodes_for_prompt(
                    world_model_json=candidate or "", max_items=10
                ),
                chosen_action_text=chosen_action_for_prompt,
                prediction=prediction,
                eval_result=eval_result,
                max_chars=self._cfg.max_chars_per_block,
            )
            raw = (self._llm_call(edit_prompt) or "").strip()
            edits = try_parse_decision_tree_edit_ops(raw)
            if edits is not None:
                candidate = self._apply_decision_tree_ops(
                    definition_name=name,
                    world_model_json=candidate,
                    edits=edits,
                    round_index=round_index,
                )

            # Post-refine validation (hard requirements). Retry if any fail.
            last_validation_errors = _validate_refine_edit(
                prev_world_model_json=prev,
                prev_obj=prev_obj_for_validation,
                candidate_world_model_json=candidate or "",
                edits=edits,
                solved_parent_id=solved_parent_id,
            )
            if not last_validation_errors:
                break
        # Final hard guard: if the model still didn't comply, deterministically insert a minimal child.
        if solved_parent_id and not _has_open_child_action(world_model_json=candidate or "", parent_id=solved_parent_id):
            candidate = _fallback_insert_min_child(world_model_json=candidate or "", parent_id=solved_parent_id)

        # Always merge computed signals (deterministic), if we have a candidate object.
        try:
            merged = merge_computed_signals(
                world_model_json=candidate,
                round_index=round_index,
                eval_result=eval_result,
            )
            if merged:
                candidate = merged
        except Exception:
            pass

        if candidate:
            self.set(name, candidate)
        return candidate

    def propose_action_nodes(
        self,
        *,
        definition_name: str,
        definition_text: str,
        reference_text: Optional[str] = None,
        current_code_excerpt: Optional[str],
        current_tree_path: Optional[str],
        baseline_targets_text: Optional[str],
        round_index: Optional[int],
    ) -> Optional[str]:
        """
        Propose/refresh OPEN action leaf nodes (unsolved) in the decision tree.
        This is used for the seed step (before any eval datapoint exists) and can also be called as maintenance.
        It does NOT require a PASSED eval.
        """
        if not self._cfg.enabled:
            return None
        name = str(definition_name or "").strip()
        if not name:
            return None
        definition_text = str(definition_text or "")
        if reference_text is None:
            reference_text = _extract_reference_from_definition_text(definition_text or "")

        prev = self.ensure_initialized(definition_name=name, definition_text=definition_text, reference_text=reference_text)
        if not prev:
            return prev

        def _has_open_frontier_action_with_score_gt(*, world_model_json: str, threshold: float) -> bool:
            """
            Validate that the executable open frontier contains at least one "high-quality" action.
            This prevents the WM from stalling by emitting only low-scoring actions.
            """
            obj = load_world_model_obj(world_model_json or "")
            if not isinstance(obj, dict):
                return False
            dt = obj.get("decision_tree")
            if not isinstance(dt, dict):
                return False
            nodes = dt.get("nodes")
            if not isinstance(nodes, list):
                return False
            root_id = str(dt.get("root_id", "") or "root")
            by_id: dict[str, dict] = {}
            for n in nodes:
                if isinstance(n, dict) and n.get("node_id"):
                    by_id[str(n["node_id"])] = n

            def _sid(n: dict) -> Optional[str]:
                sr = n.get("solution_ref")
                if not isinstance(sr, dict):
                    return None
                v = sr.get("solution_id")
                return str(v).strip() if isinstance(v, str) and v.strip() else None

            def _score01(n: dict) -> float:
                act = n.get("action") if isinstance(n.get("action"), dict) else {}
                try:
                    s01 = float(act.get("score_0_to_1", 0.0))
                except Exception:
                    s01 = 0.0
                if s01 < 0.0:
                    s01 = 0.0
                if s01 > 1.0:
                    s01 = 1.0
                return s01

            thr = float(threshold)
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                if _sid(n) is not None:
                    continue
                act = n.get("action")
                if not (isinstance(act, dict) and str(act.get("title") or "").strip()):
                    continue
                pid = n.get("parent_id")
                if pid is None:
                    continue
                pid_s = str(pid)
                if pid_s != root_id:
                    parent = by_id.get(pid_s)
                    if not isinstance(parent, dict):
                        continue
                    if _sid(parent) is None:
                        continue
                if _score01(n) > thr:
                    return True
            return False

        # If we already have enough open frontier action nodes AND the best node has open children,
        # don't spam the tree.  The minimum of 3 applies to *frontier* nodes (executable open actions
        # whose parent already has a solution or is root), not just any open action node.
        try:
            frontier_ok = self._count_open_frontier_action_nodes(world_model_json=prev) >= 3
            best_nid = self._find_best_node_id(world_model_json=prev)
            best_has_children = (
                best_nid is None  # no best node yet => nothing to ensure
                or self._node_has_open_child_action(world_model_json=prev, parent_id=best_nid)
            )
            if frontier_ok and best_has_children:
                return prev
        except Exception:
            pass
        # Determine the best node so we can steer the LLM to create children under it.
        best_nid_for_children = self._find_best_node_id(world_model_json=prev)

        candidate = prev
        max_retries = 2
        last_err: Optional[str] = None
        for attempt in range(max_retries + 1):
            frontier_text = self._render_open_frontier_nodes_for_prompt(world_model_json=candidate, max_items=10)
            status_note = None
            # Prompt steering: always ask for at least one high-score OPEN frontier action.
            best_children_hint = ""
            if best_nid_for_children and not self._node_has_open_child_action(
                world_model_json=candidate or "", parent_id=best_nid_for_children
            ):
                best_children_hint = (
                    f"\nCRITICAL: The best-performing node (node_id={best_nid_for_children}) has NO open child action nodes. "
                    f"You MUST insert at least one OPEN action child under node_id={best_nid_for_children} to continue exploring from the best solution."
                )
            if attempt == 0:
                status_note = (
                    "[ACTION_NODE_REQUIREMENT]\n"
                    "Ensure there is at least one executable OPEN frontier action with score_0_to_1 > 0.5.\n"
                    "Prefer concrete, single-iteration actions (not vague ideas)."
                    + best_children_hint
                )
            elif last_err:
                status_note = (
                    "[ACTION_NODE_VALIDATION_FAILED]\n"
                    f"{last_err}\n"
                    "Fix by inserting/updating at least one OPEN frontier action with score_0_to_1 > 0.5.\n"
                    "Do NOT output commentary; output JSON edit script only."
                    + best_children_hint
                )
            edit_prompt = build_decision_tree_edit_prompt(
                world_model_json=candidate,
                definition_text=definition_text,
                baseline_targets_text=baseline_targets_text,
                debug_and_improve_round=None,
                debug_and_improve_max_rounds=5,
                target_gpu=self._target_gpu,
                language=self._language,
                current_code_excerpt=current_code_excerpt,
                current_tree_path=current_tree_path,
                wm_status_text=status_note,
                open_frontier_nodes_text=frontier_text,
                chosen_action_text=None,
                prediction=None,
                eval_result=None,
                max_chars=self._cfg.max_chars_per_block,
            )
            raw = (self._llm_call(edit_prompt) or "").strip()
            edits = try_parse_decision_tree_edit_ops(raw)
            if edits is None:
                return prev
            candidate = self._apply_decision_tree_ops(
                definition_name=name, world_model_json=candidate, edits=edits, round_index=round_index
            )
            # Hard validation: require >=1 open frontier action with score>0.5.
            if _has_open_frontier_action_with_score_gt(world_model_json=candidate or "", threshold=0.5):
                last_err = None
                break
            last_err = "need >=1 OPEN frontier action node with score_0_to_1 > 0.5"
        # If the model still didn't comply, proceed anyway: selection will pick the best available frontier action.

        # Final hard guard: if the best node still has no open children, deterministically insert one.
        if best_nid_for_children and not self._node_has_open_child_action(
            world_model_json=candidate or "", parent_id=best_nid_for_children
        ):
            candidate = self._fallback_insert_best_node_child(
                world_model_json=candidate or "", parent_id=best_nid_for_children, round_index=round_index
            )

        if candidate:
            self.set(name, candidate)
        return candidate

    def note_action_too_hard(
        self,
        *,
        definition_name: str,
        definition_text: str,
        reference_text: Optional[str] = None,
        chosen_action_text: str | None,
        current_code_excerpt: str | None,
        current_tree_path: str | None,
        eval_result: EvalResult | None,
        debug_and_improve_round: int,
        debug_and_improve_max_rounds: int = 5,
        baseline_targets_text: str | None = None,
        round_index: int | None = None,
    ) -> Optional[str]:
        """
        Record that an action appears too hard (e.g., after exhausting debug_and_improve attempts).
        This bypasses the PASSED-only gating in `refine()` and lets WM downgrade/rewrite action metadata,
        or insert a smaller recovery ladder.
        """
        if not self._cfg.enabled:
            return None
        name = str(definition_name or "").strip()
        if not name:
            return None
        definition_text = str(definition_text or "")
        if reference_text is None:
            reference_text = _extract_reference_from_definition_text(definition_text or "")

        prev = self.ensure_initialized(definition_name=name, definition_text=definition_text, reference_text=reference_text)
        if not prev:
            return prev
        frontier_text = self._render_open_frontier_nodes_for_prompt(world_model_json=prev, max_items=10)
        edit_prompt = build_decision_tree_edit_prompt(
            world_model_json=prev,
            definition_text=definition_text,
            baseline_targets_text=baseline_targets_text,
            debug_and_improve_round=debug_and_improve_round,
            debug_and_improve_max_rounds=debug_and_improve_max_rounds,
            target_gpu=self._target_gpu,
            language=self._language,
            current_code_excerpt=current_code_excerpt,
            current_tree_path=current_tree_path,
            wm_status_text=None,
            open_frontier_nodes_text=frontier_text,
            chosen_action_text=chosen_action_text,
            prediction=None,
            eval_result=eval_result,
            max_chars=self._cfg.max_chars_per_block,
        )
        raw = (self._llm_call(edit_prompt) or "").strip()
        edits = try_parse_decision_tree_edit_ops(raw)
        if edits is None:
            return prev
        candidate = self._apply_decision_tree_ops(
            definition_name=name,
            world_model_json=prev,
            edits=edits,
            round_index=round_index,
        )
        if candidate:
            self.set(name, candidate)
        return candidate

    def choose_next_action_node_id(self, *, definition_name: str) -> Optional[str]:
        """
        Deterministically pick the next open action node to execute.

        Important: The node does NOT need to be a structural leaf in the decision tree.
        We select from the "open frontier":
        - node has an `action` and has NO attached `solution_ref.solution_id`
        - node's parent HAS a `solution_ref.solution_id` (or parent==root)
        This naturally supports multi-step action chains: step2 won't be chosen until step1 gets a solution attached.

        Ranking: action.score_0_to_1 (desc), then difficulty_1_to_5 (asc), then overall_rating_0_to_10 (desc).
        """
        name = str(definition_name or "").strip()
        if not name:
            return None
        wm = self.get(name)
        obj = load_world_model_obj(wm or "")
        if obj is None:
            return None
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return None
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return None
        root_id = str(dt.get("root_id", "") or "root")
        by_id: dict[str, dict] = {}
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                by_id[str(n["node_id"])] = n

        def _sid(n: dict) -> Optional[str]:
            sr = n.get("solution_ref")
            if not isinstance(sr, dict):
                return None
            v = sr.get("solution_id")
            return str(v).strip() if isinstance(v, str) and v.strip() else None

        def _best_vs_base() -> float:
            """
            Best observed mean_vs_baseline_factor among attached PASSED solutions in the tree.
            Returns -1.0 if none.
            """
            best = -1.0
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                sr = n.get("solution_ref")
                if not isinstance(sr, dict):
                    continue
                ev = sr.get("eval")
                if not isinstance(ev, dict):
                    continue
                status = str(ev.get("status", "") or "").strip().lower()
                if status != "passed":
                    continue
                vb = ev.get("mean_vs_baseline_factor", None)
                try:
                    vb_f = float(vb)
                except Exception:
                    continue
                if vb_f > best:
                    best = vb_f
            return best

        best_vb = _best_vs_base()
        pol = getattr(self._cfg, "selection_policy", None) or WorldModelSelectionPolicy()
        max_allowed_difficulty = int(getattr(pol, "max_difficulty_1_to_5", 3) or 3)
        try:
            if best_vb >= float(getattr(pol, "relax_difficulty_if_best_vs_base_ge", 0.9) or 0.9):
                max_allowed_difficulty = int(getattr(pol, "relaxed_max_difficulty_1_to_5", 4) or 4)
        except Exception:
            pass
        if max_allowed_difficulty < 1:
            max_allowed_difficulty = 1
        if max_allowed_difficulty > 5:
            max_allowed_difficulty = 5

        depth_cache: dict[str, int] = {}

        def _depth(nid: str) -> int:
            if nid in depth_cache:
                return depth_cache[nid]
            n = by_id.get(nid)
            if not isinstance(n, dict):
                depth_cache[nid] = 0
                return 0
            pid = n.get("parent_id")
            if pid is None:
                depth_cache[nid] = 0
                return 0
            d = 1 + _depth(str(pid))
            depth_cache[nid] = d
            return d

        cands: list[dict] = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if _sid(n) is not None:
                continue
            act = n.get("action")
            has_action = bool(isinstance(act, dict) and str(act.get("title") or "").strip())
            if not has_action:
                continue
            pid = n.get("parent_id")
            if pid is None:
                continue
            pid_s = str(pid)
            if pid_s != root_id:
                parent = by_id.get(pid_s)
                if not isinstance(parent, dict):
                    continue
                if _sid(parent) is None:
                    continue
            cands.append(n)
        if not cands:
            return None

        def _difficulty_1_to_5(n: dict) -> int:
            """
            Preferred difficulty scale is [1..5]. We accept legacy [0..3] and map it to [1..4] by +1.
            """
            act = n.get("action") if isinstance(n.get("action"), dict) else {}
            d = act.get("difficulty_1_to_5", None)
            if d is None:
                d = act.get("difficulty_0_to_3", None)
                try:
                    d = (int(d) + 1) if d is not None else 3
                except Exception:
                    d = 3
            try:
                di = int(d)
            except Exception:
                di = 3
            if di < 1:
                di = 1
            if di > 5:
                di = 5
            return di

        # Difficulty gating: execute only actions with difficulty<=max_allowed_difficulty when possible.
        filtered = [n for n in cands if _difficulty_1_to_5(n) <= max_allowed_difficulty]
        cands_eff = filtered if filtered else cands

        def _score01(n: dict) -> float:
            """
            Action-local score in [0,1]. If missing/unparseable, return 0.
            """
            act = n.get("action") if isinstance(n.get("action"), dict) else {}
            try:
                s01 = float(act.get("score_0_to_1", 0.0))
            except Exception:
                s01 = 0.0
            if s01 < 0.0:
                s01 = 0.0
            if s01 > 1.0:
                s01 = 1.0
            return s01

        def _conf(n: dict) -> float:
            try:
                return float(n.get("confidence_0_to_1", 0.0) or 0.0)
            except Exception:
                return 0.0

        def _rating01(n: dict) -> float:
            try:
                return float(n.get("overall_rating_0_to_10", 0.0)) / 10.0
            except Exception:
                return 0.0

        # Deterministic selection with stable tie-breakers.
        cands_eff.sort(
            key=lambda n: (
                -_score01(n),
                _difficulty_1_to_5(n),
                -_rating01(n),
                str(n.get("node_id") or ""),
            )
        )
        nid = str(cands_eff[0].get("node_id") or "").strip()
        return nid or None

    # Backward-compatible alias (no Definition object support).
    def choose_next_action_leaf_id(self, *, definition_name: str) -> Optional[str]:
        return self.choose_next_action_node_id(definition_name=definition_name)

    def _count_open_action_nodes(self, *, world_model_json: str) -> int:
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return 0
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return 0
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return 0
        count = 0
        for n in nodes:
            if not isinstance(n, dict):
                continue
            sr = n.get("solution_ref")
            sid = None
            if isinstance(sr, dict):
                v = sr.get("solution_id")
                sid = str(v).strip() if isinstance(v, str) and v.strip() else None
            if sid is not None:
                continue
            act = n.get("action")
            if isinstance(act, dict) and str(act.get("title") or "").strip():
                count += 1
        return count

    def _count_open_frontier_action_nodes(self, *, world_model_json: str) -> int:
        """
        Count executable open action nodes ("frontier"):
        - node has action.title, no solution_id
        - parent has a solution_id (or parent==root)
        """
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return 0
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return 0
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return 0
        root_id = str(dt.get("root_id", "") or "root")
        by_id: dict[str, dict] = {}
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                by_id[str(n["node_id"])] = n

        def _sid(n: dict) -> Optional[str]:
            sr = n.get("solution_ref")
            if not isinstance(sr, dict):
                return None
            v = sr.get("solution_id")
            return str(v).strip() if isinstance(v, str) and v.strip() else None

        cnt = 0
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if _sid(n) is not None:
                continue
            act = n.get("action")
            if not (isinstance(act, dict) and str(act.get("title") or "").strip()):
                continue
            pid = n.get("parent_id")
            if pid is None:
                continue
            pid_s = str(pid)
            if pid_s != root_id:
                parent = by_id.get(pid_s)
                if not isinstance(parent, dict):
                    continue
                if _sid(parent) is None:
                    continue
            cnt += 1
        return cnt

    def _find_best_node_id(self, *, world_model_json: str) -> Optional[str]:
        """
        Find the node_id of the best solved node strictly by ATTACHED solution eval score:
        solution_ref.eval.metrics.score (higher is better).
        No node-local fallback signals are used.
        """
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return None
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return None
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return None

        best_nid: Optional[str] = None
        best_key: tuple[float, str] | None = None
        for n in nodes:
            if not isinstance(n, dict):
                continue
            sr = n.get("solution_ref")
            if not isinstance(sr, dict):
                continue
            sid = sr.get("solution_id")
            if not (isinstance(sid, str) and sid.strip()):
                continue

            ev = sr.get("eval")
            if not isinstance(ev, dict):
                continue
            m = ev.get("metrics")
            if not isinstance(m, dict):
                continue
            s = m.get("score")
            if not isinstance(s, (int, float)):
                continue
            sol_score = float(s)
            nid = str(n.get("node_id") or "").strip()
            if not nid:
                continue
            key = (sol_score, nid)
            if best_key is None or key > best_key:
                best_key = key
                best_nid = nid
        return best_nid

    def _node_has_open_child_action(self, *, world_model_json: str, parent_id: str) -> bool:
        """
        Check whether a given node has at least one OPEN child action node
        (child with action.title but no solution_id).
        """
        obj = load_world_model_obj(world_model_json or "")
        if not isinstance(obj, dict):
            return False
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return False
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return False

        for n in nodes:
            if not isinstance(n, dict):
                continue
            if str(n.get("parent_id") or "") != parent_id:
                continue
            # Must not already have a solution attached
            sr = n.get("solution_ref")
            if isinstance(sr, dict):
                sid = sr.get("solution_id")
                if isinstance(sid, str) and sid.strip():
                    continue
            act = n.get("action")
            if isinstance(act, dict) and str(act.get("title") or "").strip():
                return True
        return False

    def _fallback_insert_best_node_child(
        self,
        *,
        world_model_json: str,
        parent_id: str,
        round_index: Optional[int] = None,
    ) -> str:
        """
        Deterministic fallback: insert a minimal OPEN action child under the best-performing node
        so the search can continue exploring from the best solution.
        """
        obj = load_world_model_obj(world_model_json or "")
        if not isinstance(obj, dict):
            return world_model_json
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return world_model_json
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return world_model_json
        by_id: dict[str, dict] = {}
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                by_id[str(n["node_id"])] = n
        parent = by_id.get(parent_id)
        if not isinstance(parent, dict):
            return world_model_json
        # Capture parent solution_id if present.
        parent_sol = None
        srp = parent.get("solution_ref")
        if isinstance(srp, dict):
            ps = srp.get("solution_id")
            parent_sol = str(ps).strip() if isinstance(ps, str) and ps.strip() else None

        rid = "r" + str(round_index) if round_index is not None else "rX"
        counter = 0
        while True:
            counter += 1
            nid = f"node_best_cont_{rid}_{counter}"
            if nid not in by_id:
                break

        child = {
            "node_id": nid,
            "parent_id": parent_id,
            "decision": "Continue from best",
            "choice": "Next-step refinement of best solution",
            "overall_rating_0_to_10": max(0.0, float(parent.get("overall_rating_0_to_10") or 0.0)),
            "confidence_0_to_1": max(0.2, float(parent.get("confidence_0_to_1") or 0.2)),
            "notes": "AUTO_BEST_CONTINUATION: the best-performing node had no open children. "
            "Inserting a continuation action to keep exploring from the best solution.",
            "impacts": parent.get("impacts") if isinstance(parent.get("impacts"), dict) else {},
            "action": {
                "title": "Refine best solution: target the top bottleneck",
                "description": "Propose one small change that targets the primary bottleneck indicated by the latest eval on the best solution, without changing the overall mapping plan.",
                "rationale": "The best solution should always have a continuation path for further improvement.",
                "score_0_to_1": 0.6,
                "difficulty_1_to_5": 2,
            },
            "solution_ref": {"solution_id": None, "parent_solution_id": parent_sol, "eval": None},
            "last_updated_round": round_index,
        }
        nodes.append(child)
        return dump_world_model_obj(obj) or world_model_json

    def _apply_decision_tree_ops(
        self,
        *,
        definition_name: str,
        world_model_json: Optional[str],
        edits: DecisionTreeEditOps,
        round_index: Optional[int],
    ) -> Optional[str]:
        """
        Deterministically apply update/insert/split ops to the decision tree.
        This is where nodes are inserted/split/updated (not in the generator loop).

        TODO(world-model): enforce path-consistency deterministically.
        Today, we rely on the LLM prompt to avoid adding descendant nodes that contradict earlier commitments
        on the same root->leaf path. Consider adding explicit per-node 'commitments' and validating edits here.
        """
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return world_model_json
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return dump_world_model_obj(obj)

        root_id = str(dt.get("root_id", "") or "root")
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            nodes = []
            dt["nodes"] = nodes

        by_id: dict[str, dict] = {}
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                by_id[str(n["node_id"])] = n

        # id generator
        counter = 0
        safe_def = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(definition_name or ""))[:48] or "def"
        def _gen_id(prefix: str) -> str:
            nonlocal counter
            counter += 1
            rid = "r" + str(round_index) if round_index is not None else "rX"
            return f"{prefix}_{safe_def}_{rid}_{counter}"

        def _ensure_node_id(nid: str) -> Optional[dict]:
            return by_id.get(str(nid)) if nid is not None else None

        def _apply_patch(node: dict, patch: dict) -> None:
            if not isinstance(patch, dict):
                return
            # Root is dummy: don't allow changing decision/choice on root.
            is_root = (str(node.get("node_id")) == root_id) or (node.get("parent_id") is None)
            if (not is_root) and ("decision" in patch):
                node["decision"] = str(patch.get("decision") or "").strip()
            if (not is_root) and ("choice" in patch):
                node["choice"] = str(patch.get("choice") or "").strip()
            if "overall_rating_0_to_10" in patch:
                node["overall_rating_0_to_10"] = patch.get("overall_rating_0_to_10")
            if "confidence_0_to_1" in patch:
                node["confidence_0_to_1"] = patch.get("confidence_0_to_1")
            if "notes" in patch:
                node["notes"] = str(patch.get("notes") or "").strip()
            if "last_updated_round" in patch:
                node["last_updated_round"] = patch.get("last_updated_round")
            if "impacts" in patch and isinstance(patch.get("impacts"), dict):
                node.setdefault("impacts", {})
                if isinstance(node["impacts"], dict):
                    # shallow merge impacts dict
                    for k, v in patch["impacts"].items():
                        if isinstance(v, dict):
                            node["impacts"].setdefault(k, {})
                            if isinstance(node["impacts"].get(k), dict):
                                node["impacts"][k].update(v)
                            else:
                                node["impacts"][k] = v
            # Optional: solution_ref patch
            if "solution_ref" in patch and isinstance(patch.get("solution_ref"), dict):
                node.setdefault("solution_ref", {})
                if isinstance(node.get("solution_ref"), dict):
                    node["solution_ref"].update(patch["solution_ref"])
            # Optional: action patch (structured next-action metadata)
            if "action" in patch and isinstance(patch.get("action"), dict):
                node.setdefault("action", {})
                if isinstance(node.get("action"), dict):
                    # shallow merge action dict; caller should keep it minimal
                    node["action"].update(patch["action"])

        # Apply ops
        # Guardrail: avoid the model exploding the tree with many low-signal nodes in one edit-script.
        # We cap the number of NEW nodes added per edit application deterministically.
        max_new_nodes_per_edit = int(getattr(self._cfg, "max_new_nodes_per_edit", 8) or 8)
        if max_new_nodes_per_edit < 0:
            max_new_nodes_per_edit = 0
        new_nodes_added = 0
        # Lightweight stats for debugging why trees don't change as expected.
        applied_updates = 0
        applied_inserts = 0
        applied_split_children = 0
        applied_deletes = 0
        skipped_cap = 0
        skipped_missing_parent = 0
        skipped_parent_solution_id_mismatch = 0
        skipped_update_missing_node = 0
        skipped_insert_invalid_node = 0
        skipped_split_invalid_children = 0
        skipped_delete_not_allowed = 0
        for op in edits.ops:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "") or "").strip()
            if kind == "update_node":
                nid = str(op.get("node_id", "") or "").strip()
                node = _ensure_node_id(nid)
                if node is None:
                    skipped_update_missing_node += 1
                    continue
                _apply_patch(node, op.get("patch", {}))
                applied_updates += 1
            elif kind == "insert_node":
                if new_nodes_added >= max_new_nodes_per_edit:
                    skipped_cap += 1
                    continue
                parent_id = str(op.get("parent_id", "") or "").strip()
                parent = _ensure_node_id(parent_id)
                if parent is None:
                    skipped_missing_parent += 1
                    continue
                # Enforce parent_solution_id requirement if parent has solution_id.
                parent_sol = None
                if isinstance(parent.get("solution_ref"), dict):
                    parent_sol = parent["solution_ref"].get("solution_id")
                provided_parent_sol = op.get("parent_solution_id")
                if parent_sol:
                    # Be tolerant if the model forgets parent_solution_id: auto-fill it.
                    # Still reject explicit mismatches.
                    if not (isinstance(provided_parent_sol, str) and provided_parent_sol.strip()):
                        provided_parent_sol = str(parent_sol)
                        op["parent_solution_id"] = str(parent_sol)
                    if not (isinstance(provided_parent_sol, str) and provided_parent_sol.strip() == str(parent_sol)):
                        skipped_parent_solution_id_mismatch += 1
                        continue
                node_in = op.get("node", {})
                if not isinstance(node_in, dict):
                    skipped_insert_invalid_node += 1
                    continue
                nid = str(node_in.get("node_id", "") or "").strip() or _gen_id("node")
                if nid in by_id:
                    nid = _gen_id("node")
                new_node = dict(node_in)
                new_node["node_id"] = nid
                new_node["parent_id"] = parent_id
                # Set solution_ref.parent_solution_id for traceability.
                new_node.setdefault("solution_ref", {})
                if isinstance(new_node.get("solution_ref"), dict):
                    new_node["solution_ref"].setdefault("parent_solution_id", str(parent_sol) if parent_sol else None)
                nodes.append(new_node)
                by_id[nid] = new_node
                new_nodes_added += 1
                applied_inserts += 1
            elif kind == "split_node":
                nid = str(op.get("node_id", "") or "").strip()
                node = _ensure_node_id(nid)
                if node is None:
                    skipped_missing_parent += 1
                    continue
                # Update parent node (decision) if provided
                _apply_patch(node, op.get("parent_patch", {}))
                children = op.get("children", [])
                if not isinstance(children, list):
                    skipped_split_invalid_children += 1
                    continue
                for ch in children[:8]:
                    if new_nodes_added >= max_new_nodes_per_edit:
                        skipped_cap += 1
                        break
                    if not isinstance(ch, dict):
                        continue
                    cid = str(ch.get("node_id", "") or "").strip() or _gen_id("child")
                    if cid in by_id:
                        cid = _gen_id("child")
                    new_node = dict(ch)
                    new_node["node_id"] = cid
                    new_node["parent_id"] = nid
                    # Propagate parent_solution_id from split parent if present.
                    psol = None
                    if isinstance(node.get("solution_ref"), dict):
                        psol = node["solution_ref"].get("solution_id")
                    new_node.setdefault("solution_ref", {})
                    if isinstance(new_node.get("solution_ref"), dict):
                        new_node["solution_ref"].setdefault("parent_solution_id", str(psol) if psol else None)
                    nodes.append(new_node)
                    by_id[cid] = new_node
                    new_nodes_added += 1
                    applied_split_children += 1
            elif kind == "delete_node":
                nid = str(op.get("node_id", "") or "").strip()
                node = _ensure_node_id(nid)
                if node is None:
                    skipped_delete_not_allowed += 1
                    continue
                # Safety: root cannot be deleted; only OPEN leaf nodes without solution_id.
                if nid == root_id:
                    skipped_delete_not_allowed += 1
                    continue
                # Must have no solution
                sr = node.get("solution_ref")
                if isinstance(sr, dict):
                    sid = sr.get("solution_id")
                    if isinstance(sid, str) and sid.strip():
                        skipped_delete_not_allowed += 1
                        continue
                # Must be a leaf (no children)
                has_child = False
                for maybe_child in nodes:
                    if isinstance(maybe_child, dict) and str(maybe_child.get("parent_id") or "") == nid:
                        has_child = True
                        break
                if has_child:
                    skipped_delete_not_allowed += 1
                    continue
                try:
                    nodes.remove(node)
                    by_id.pop(nid, None)
                    applied_deletes += 1
                except Exception:
                    skipped_delete_not_allowed += 1
                    pass

        # Update active leaf if requested
        if edits.active_leaf_id and edits.active_leaf_id in by_id:
            dt["active_leaf_id"] = edits.active_leaf_id

        # One-line summary helps diagnose "why didn't refine add nodes?" from logs.
        try:
            report = {
                "ops": int(len(edits.ops)),
                "updates": int(applied_updates),
                "inserts": int(applied_inserts),
                "split_children": int(applied_split_children),
                "deletes": int(applied_deletes),
                "new_nodes_added": int(new_nodes_added),
                "max_new_nodes_per_edit": int(max_new_nodes_per_edit),
                "skipped_cap": int(skipped_cap),
                "skipped_missing_parent": int(skipped_missing_parent),
                "skipped_parent_solution_id_mismatch": int(skipped_parent_solution_id_mismatch),
                "skipped_update_missing_node": int(skipped_update_missing_node),
                "skipped_insert_invalid_node": int(skipped_insert_invalid_node),
                "skipped_split_invalid_children": int(skipped_split_invalid_children),
                "skipped_delete_not_allowed": int(skipped_delete_not_allowed),
            }
            self._last_apply_ops_report = report
            print(
                "[WM] apply_edit_ops:"
                f" ops={report['ops']}"
                f" updates={report['updates']}"
                f" inserts={report['inserts']}"
                f" split_children={report['split_children']}"
                f" deletes={report['deletes']}"
                f" new_nodes_added={report['new_nodes_added']}/{report['max_new_nodes_per_edit']}"
                f" skipped_cap={report['skipped_cap']}"
                f" skipped_missing_parent={report['skipped_missing_parent']}"
                f" skipped_parent_solution_id_mismatch={report['skipped_parent_solution_id_mismatch']}"
                f" skipped_update_missing_node={report['skipped_update_missing_node']}"
                f" skipped_insert_invalid_node={report['skipped_insert_invalid_node']}"
                f" skipped_split_invalid_children={report['skipped_split_invalid_children']}"
                f" skipped_delete_not_allowed={report['skipped_delete_not_allowed']}"
            )
            if (
                report["skipped_cap"]
                or report["skipped_missing_parent"]
                or report["skipped_parent_solution_id_mismatch"]
                or report["skipped_update_missing_node"]
                or report["skipped_insert_invalid_node"]
                or report["skipped_split_invalid_children"]
                or report["skipped_delete_not_allowed"]
            ):
                print("[WM][WARN] edit-ops had skipped operations; refine will retry if this was a refine call.")
        except Exception:
            pass

        # Normalize + return
        return dump_world_model_obj(obj)

    def attach_solution_to_active_leaf(
        self,
        *,
        definition_name: str,
        solution_id: str,
        solution_name: str,
        eval_result: EvalResult,
        round_index: Optional[int],
    ) -> Optional[str]:
        """
        Attach a solution reference to the current active leaf node in the decision tree.
        This is deterministic and enables backtracking to any node's solution.
        """
        name = str(definition_name or "").strip()
        if not name:
            return None
        wm = self.get(name)
        obj = load_world_model_obj(wm or "")
        if obj is None:
            return wm
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return wm
        active = str(dt.get("active_leaf_id", "") or dt.get("root_id", "") or "root")
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return wm
        target = None
        for n in nodes:
            if isinstance(n, dict) and str(n.get("node_id") or "") == active:
                target = n
                break
        if target is None:
            return wm
        target.setdefault("solution_ref", {})
        if isinstance(target.get("solution_ref"), dict):
            # Preserve any existing parent_solution_id set when the node was inserted.
            existing_parent = target["solution_ref"].get("parent_solution_id")
            target["solution_ref"].update(
                {
                    "solution_id": solution_id,
                    "solution_name": solution_name,
                    "parent_solution_id": existing_parent,
                    "eval": eval_result.to_dict(include_log_excerpt=False),
                    "round_index": round_index,
                }
            )
        updated = dump_world_model_obj(obj)
        self.set(name, updated)
        return updated

    def get_active_leaf_solution_ref(
        self, *, definition_name: str
    ) -> dict:
        """Return the active leaf node's solution_ref (may be empty)."""
        name = str(definition_name or "").strip()
        if not name:
            return {}
        wm = self.get(name)
        obj = load_world_model_obj(wm or "")
        if obj is None:
            return {}
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return {}
        active = str(dt.get("active_leaf_id", "") or dt.get("root_id", "") or "root")
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return {}
        for n in nodes:
            if isinstance(n, dict) and str(n.get("node_id") or "") == active:
                sr = n.get("solution_ref")
                return sr if isinstance(sr, dict) else {}
        return {}

    def get_active_leaf_id(
        self, *, definition_name: str
    ) -> str:
        """Return the active leaf node id (falls back to root)."""
        name = str(definition_name or "").strip()
        if not name:
            return "root"
        wm = self.get(name)
        obj = load_world_model_obj(wm or "")
        if obj is None:
            return "root"
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return "root"
        return str(dt.get("active_leaf_id", "") or dt.get("root_id", "") or "root")

    def set_active_leaf_id(
        self, *, definition_name: str, node_id: str
    ) -> Optional[str]:
        """Deterministically set the decision tree's active leaf id (no LLM)."""
        name = str(definition_name or "").strip()
        if not name:
            return None
        wm = self.get(name)
        obj = load_world_model_obj(wm or "")
        if obj is None:
            return wm
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return wm
        nid = str(node_id or "").strip()
        if not nid:
            return wm
        dt["active_leaf_id"] = nid
        updated = dump_world_model_obj(obj)
        self.set(name, updated)
        return updated

    def get_tree_path_text(
        self, *, definition_name: str, node_id: Optional[str] = None
    ) -> str:
        """
        Return a compact text path root -> ... -> node_id (or active leaf), suitable for prompts.
        Includes decision/choice only (no heavy fields).
        """
        name = str(definition_name or "").strip()
        if not name:
            return ""
        wm = self.get(name)
        obj = load_world_model_obj(wm or "")
        if obj is None:
            return ""
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return ""
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return ""
        by_id: dict[str, dict] = {}
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                by_id[str(n["node_id"])] = n
        root_id = str(dt.get("root_id", "") or "root")
        target = str(node_id or "").strip() or str(dt.get("active_leaf_id", "") or root_id)
        if target not in by_id:
            target = root_id
        # Walk up to root
        path: list[dict] = []
        cur = by_id.get(target)
        guard = 0
        while isinstance(cur, dict) and guard < 64:
            path.append(cur)
            pid = cur.get("parent_id")
            if pid is None:
                break
            cur = by_id.get(str(pid))
            guard += 1
        path.reverse()
        # Render
        parts: list[str] = []
        for n in path:
            nid = str(n.get("node_id") or "")
            dec = n.get("decision")
            ch = n.get("choice")
            if dec is None and ch is None:
                parts.append(f"{nid}: <root>")
            else:
                d = str(dec or "").strip()
                c = str(ch or "").strip()
                if d and c:
                    parts.append(f"{nid}: {d} -> {c}")
                elif d:
                    parts.append(f"{nid}: {d}")
                else:
                    parts.append(f"{nid}: {c}")
        return "\n".join(f"- {p}" for p in parts).strip()

    def get_solution_ref_for_node(self, *, definition_name: str, node_id: str) -> dict:
        """Return a specific node's solution_ref (may be empty)."""
        name = str(definition_name or "").strip()
        if not name:
            return {}
        wm = self.get(name)
        obj = load_world_model_obj(wm or "")
        if obj is None:
            return {}
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return {}
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return {}
        nid = str(node_id or "").strip()
        for n in nodes:
            if isinstance(n, dict) and str(n.get("node_id") or "") == nid:
                sr = n.get("solution_ref")
                return sr if isinstance(sr, dict) else {}
        return {}

    def run(  # pragma: no cover
        self,
        *,
        current_code_excerpt: str,
        current_active_node_id: str,
        eval_result: Optional[EvalResult],
        baseline_targets_text: Optional[str] = None,
        world_model_json: Optional[str] = None,
        round_index: Optional[int] = None,
    ) -> ActionRanking:
        """
        DEPRECATED: Action ranking via `run()` has been superseded by "open action nodes" in the decision tree.
        - The WM proposes action candidates by inserting/updating UNSOLVED leaf nodes with `node.action`.
        - The system chooses a node to execute via `choose_next_action_leaf_id()`.
        """
        return ActionRanking(candidates=[], ranking=[], raw_model_output="(deprecated)")

    def _decision_tree_node_ids(self, *, world_model_json: str) -> set[str]:
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return set()
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return set()
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return set()
        out: set[str] = set()
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                out.add(str(n["node_id"]))
        return out

    def _decision_tree_nodes_by_id(self, *, world_model_json: str) -> dict[str, dict]:
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return {}
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return {}
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return {}
        by_id: dict[str, dict] = {}
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                by_id[str(n["node_id"])] = n
        return by_id

    def get_node_obj(self, *, definition_name: str, node_id: str) -> Optional[dict]:
        name = str(definition_name or "").strip()
        if not name:
            return None
        wm = self.get(name)
        obj = load_world_model_obj(wm or "")
        if obj is None:
            return None
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return None
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return None
        nid = str(node_id or "").strip()
        for n in nodes:
            if isinstance(n, dict) and str(n.get("node_id") or "") == nid:
                return n
        return None

    def _render_open_frontier_nodes_for_prompt(self, *, world_model_json: str, max_items: int = 10) -> str:
        """
        Render "highest open frontier" nodes for the action-ranking prompt.
        Frontier node = node has NO attached solution_id AND its parent HAS a solution_id.
        These are ready-to-fill leaves (or pending decision nodes) that can anchor next actions.
        """
        by_id = self._decision_tree_nodes_by_id(world_model_json=world_model_json)
        if not by_id:
            return ""

        # depth cache
        depth_cache: dict[str, int] = {}

        def _depth(nid: str) -> int:
            if nid in depth_cache:
                return depth_cache[nid]
            n = by_id.get(nid)
            if not isinstance(n, dict):
                depth_cache[nid] = 0
                return 0
            pid = n.get("parent_id")
            if pid is None:
                depth_cache[nid] = 0
                return 0
            d = 1 + _depth(str(pid))
            depth_cache[nid] = d
            return d

        def _solution_id(n: dict) -> Optional[str]:
            sr = n.get("solution_ref")
            if not isinstance(sr, dict):
                return None
            v = sr.get("solution_id")
            return str(v).strip() if isinstance(v, str) and v.strip() else None

        def _eval_perf_str_from_node(n: Optional[dict]) -> str:
            if not isinstance(n, dict):
                return "(no-parent)"
            sr = n.get("solution_ref")
            if not isinstance(sr, dict):
                return "(no-parent-solution)"
            ev = sr.get("eval")
            if not isinstance(ev, dict):
                return "(no-parent-eval)"
            mvb = ev.get("mean_vs_baseline_factor")
            sp = ev.get("speedup_factor")
            lat = ev.get("latency_ms")
            parts: list[str] = []
            if isinstance(mvb, (int, float)) and float(mvb) > 0:
                parts.append(f"vs_base={float(mvb):.3g}x")
            elif isinstance(sp, (int, float)) and float(sp) > 0:
                parts.append(f"speedup={float(sp):.3g}x")
            if isinstance(lat, (int, float)) and float(lat) > 0:
                parts.append(f"lat={float(lat):.3g}ms")
            return " ".join(parts) if parts else "(no-perf)"

        def _parent_solution_id(n: dict) -> Optional[str]:
            sr = n.get("solution_ref")
            if not isinstance(sr, dict):
                return None
            v = sr.get("parent_solution_id")
            return str(v).strip() if isinstance(v, str) and v.strip() else None

        frontier: list[dict] = []
        for nid, n in by_id.items():
            if not isinstance(n, dict):
                continue
            if _solution_id(n) is not None:
                continue
            pid = n.get("parent_id")
            if pid is None:
                continue
            parent = by_id.get(str(pid))
            if not isinstance(parent, dict):
                continue
            if _solution_id(parent) is None and str(pid) != "root":
                continue
            frontier.append(n)

        if not frontier:
            return ""

        # highest first = lowest depth
        frontier.sort(key=lambda n: _depth(str(n.get("node_id") or "")))
        lines: list[str] = []
        for n in frontier[: max(1, int(max_items))]:
            nid = str(n.get("node_id") or "").strip()
            pid = str(n.get("parent_id") or "").strip()
            dec = str(n.get("decision") or "").strip()
            choice = str(n.get("choice") or "").strip()
            psid = _parent_solution_id(n) or ""
            parent = by_id.get(pid) if pid else None
            parent_perf = "(root)" if pid == "root" else _eval_perf_str_from_node(parent)
            # Optional: print action difficulty for quick triage in prompts/logs.
            diff_s = ""
            try:
                act = n.get("action") if isinstance(n.get("action"), dict) else {}
                d = act.get("difficulty_1_to_5", act.get("difficulty_0_to_3", None))
                if d is not None:
                    di = int(d)
                    # Legacy 0..3 -> 1..4
                    if "difficulty_1_to_5" not in act and "difficulty_0_to_3" in act:
                        di = di + 1
                    if di < 1:
                        di = 1
                    if di > 5:
                        di = 5
                    diff_s = str(di)
            except Exception:
                diff_s = ""
            note = str(n.get("notes") or "").strip().replace("\n", " ")
            if len(note) > 140:
                note = note[:140] + "..."
            lines.append(
                f"- node_id={nid} | parent_id={pid} | parent_solution_id={psid or '(unknown)'}"
                + f" | parent_perf={parent_perf}"
                + (f" | difficulty_1_to_5={diff_s}" if diff_s else "")
                + f" | decision={dec or '(empty)'} | choice={choice or '(empty)'} | notes={note or '(empty)'}"
            )
        return "\n".join(lines).strip()

    def _solution_id_by_node_id(self, *, world_model_json: str) -> dict[str, Optional[str]]:
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return {}
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return {}
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return {}
        out: dict[str, Optional[str]] = {}
        for n in nodes:
            if not isinstance(n, dict) or not n.get("node_id"):
                continue
            nid = str(n["node_id"])
            sr = n.get("solution_ref")
            sid = None
            if isinstance(sr, dict):
                v = sr.get("solution_id")
                sid = str(v).strip() if isinstance(v, str) and v.strip() else None
            out[nid] = sid
        return out

    def _ensure_action_child_nodes(
        self,
        *,
        definition_name: str,
        world_model_json: str,
        base_solution_id_by_node_id: dict[str, Optional[str]],
        candidates: list[ActionCandidate],
        round_index: Optional[int],
    ) -> dict[str, str]:
        """
        Deterministically reserve one leaf node per action candidate under `base_node_id`.
        This enables action ranking items to specify `attach_to_node_id` without mutating the tree via LLM ops.
        """
        obj = load_world_model_obj(world_model_json or "")
        if obj is None:
            return {}
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return {}
        nodes = dt.get("nodes")
        if not isinstance(nodes, list):
            return {}
        # Index nodes
        by_id: dict[str, dict] = {}
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                by_id[str(n["node_id"])] = n
        rid = f"r{int(round_index)}" if isinstance(round_index, int) else "rX"
        mapping: dict[str, str] = {}
        safe_def = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(definition_name or ""))[:48] or "def"
        for c in candidates:
            aid = str(c.action_id or "").strip()
            if not aid:
                continue
            root_id = str(dt.get("root_id", "") or "root")
            base_id = str(c.base_node_id or "").strip() or root_id
            if base_id not in by_id:
                base_id = root_id
            base_sid = base_solution_id_by_node_id.get(base_id)
            # If the model selected an existing frontier leaf under this base, reuse it.
            existing_attach = str(getattr(c, "attach_to_node_id", None) or "").strip()
            if existing_attach and existing_attach in by_id:
                n = by_id[existing_attach]
                if isinstance(n, dict) and str(n.get("parent_id") or "") == base_id:
                    sr = n.get("solution_ref")
                    sid = None
                    psid = None
                    if isinstance(sr, dict):
                        v = sr.get("solution_id")
                        sid = str(v).strip() if isinstance(v, str) and v.strip() else None
                        v2 = sr.get("parent_solution_id")
                        psid = str(v2).strip() if isinstance(v2, str) and v2.strip() else None
                    if sid is None:
                        mapping[aid] = existing_attach
                        mapping[f"{aid}__base_solution_id"] = psid or (str(base_sid) if isinstance(base_sid, str) else "")
                        continue

            # Deterministic node id; stable across calls within same round (include base node for uniqueness).
            safe_base = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in base_id)[:24]
            safe_aid = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in aid)[:24]
            nid = f"action_{safe_def}_{rid}_{safe_base}_{safe_aid}"
            mapping[aid] = nid
            # Also return the base solution id for this action (for filling ranking items deterministically).
            mapping[f"{aid}__base_solution_id"] = str(base_sid) if isinstance(base_sid, str) and base_sid.strip() else ""
            if nid in by_id:
                continue
            new_node = {
                "node_id": nid,
                "parent_id": base_id,
                "decision": "Next action",
                "choice": f"{c.title} ({aid})",
                "notes": str(c.description or "").strip()[:400],
                "last_updated_round": int(round_index) if isinstance(round_index, int) else 0,
                "solution_ref": {"solution_id": None, "parent_solution_id": base_sid, "eval": {}},
            }
            nodes.append(new_node)
            by_id[nid] = new_node

        updated = dump_world_model_obj(obj)
        self.set(str(definition_name), updated)
        return mapping

