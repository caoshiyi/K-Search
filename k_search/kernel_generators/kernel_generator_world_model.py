"""World-model aware kernel generators.

These classes keep all world-model-related logic out of `kernel_generator.py`.
They reuse the base generator helpers (code cleaning, solution creation, trace selection)
and only override prompt construction to inject the persistent world model JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pathlib import Path

from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
)
from k_search.kernel_generators.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    RestoredCheckpoint,
)
from k_search.kernel_generators.kernel_generator import KernelGenerator
from k_search.kernel_generators.llm_clients import (
    LLMProviderFatalError,
    llm_log_context,
)
from k_search.tasks.task_base import code_from_solution
from k_search.kernel_generators.kernel_generator_prompts import (
    get_prompt_from_definition_text,
)
from k_search.kernel_generators.world_model_prompts import (
    get_debug_and_improve_from_spec_prompt_from_text,
    get_debug_generated_code_prompt_from_text,
    get_generate_code_from_action_prompt_from_text,
    get_generate_code_from_spec_with_action_prompt_from_text,
    get_improve_from_spec_prompt_from_text,
    get_improve_generated_code_prompt_from_text,
)
from k_search.kernel_generators.world_model_manager import (
    WorldModelConfig,
    WorldModelManager,
    WorldModelSelectionPolicy,
)
from k_search.kernel_generators.strategy_injection import StrategyCatalogEntry
from k_search.tasks.task_base import EvalResult
from k_search.kernel_generators.world_model import (
    Prediction,
    dump_world_model_obj,
    load_world_model_obj,
    render_chosen_action_node_block,
    render_open_action_nodes_block,
    render_world_model_section,
    render_world_model_status,
)
from k_search.utils.solution_db import SolutionDB
from k_search.utils.paths import (
    get_ksearch_artifacts_dir,
    get_ksearch_run_dir,
    get_run_id,
    get_run_world_model_dir,
    get_task_id,
)
from k_search.telemetry.context import TelemetryContext, build_attempt_dir
from k_search.telemetry.diagnostics import diagnose_tool_protocol_failure
from k_search.telemetry.narrative import RunNarrativeLogger


class MissingStrategyFileError(RuntimeError):
    pass


class NoExecutableStrategyNodeError(RuntimeError):
    def __init__(self, blocked: list[dict[str, Any]]) -> None:
        super().__init__("no executable strategy-backed world-model action nodes")
        self.blocked = list(blocked or [])


def resolve_strategy_catalog_entry(
    node: dict[str, Any],
    strategy_catalog: list[StrategyCatalogEntry],
) -> StrategyCatalogEntry | None:
    """Resolve a world-model node to a markdown-backed catalog strategy.

    Child or title-only nodes are intentionally not inferred from parents.
    The only legacy fallback is pure sN -> catalog[N-1].
    """
    if not isinstance(node, dict) or not strategy_catalog:
        return None
    action = node.get("action") if isinstance(node.get("action"), dict) else {}
    strategy_ref = (
        action.get("strategy_ref")
        if isinstance(action.get("strategy_ref"), dict)
        else {}
    )
    ref_id = str(strategy_ref.get("id") or "").strip()
    ref_markdown = str(strategy_ref.get("markdown_ref") or "").strip()
    node_id = str(node.get("node_id") or "").strip()

    entry: StrategyCatalogEntry | None = None
    for candidate in strategy_catalog:
        if ref_id and candidate.id == ref_id:
            entry = candidate
            break
        if ref_markdown and candidate.markdown_ref == ref_markdown:
            entry = candidate
            break

    if entry is None and re.fullmatch(r"s\d+", node_id):
        idx = int(node_id[1:]) - 1
        if 0 <= idx < len(strategy_catalog):
            entry = strategy_catalog[idx]

    if entry is None or not str(entry.markdown_ref or "").strip():
        return None
    return entry


def _node_solution_id(node: dict[str, Any]) -> str | None:
    sr = node.get("solution_ref")
    if not isinstance(sr, dict):
        return None
    sid = sr.get("solution_id")
    return str(sid).strip() if isinstance(sid, str) and sid.strip() else None


def _strategy_requires(entry: StrategyCatalogEntry) -> tuple[str, ...]:
    raw = getattr(entry, "requires", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    return ()


def _strategy_allow_reexecute(entry: StrategyCatalogEntry) -> bool:
    return bool(getattr(entry, "allow_reexecute", False))


def _action_is_closed_for_strategy_selection(node: dict[str, Any]) -> bool:
    action = node.get("action") if isinstance(node.get("action"), dict) else {}
    for status in (node.get("status"), action.get("status"), action.get("state")):
        if str(status or "").strip().lower() in {
            "too_hard",
            "blocked",
            "closed",
            "deferred",
            "skipped",
        }:
            return True
    return bool(node.get("too_hard") is True or action.get("too_hard") is True)


def _open_frontier_action_nodes(
    world_model_obj: dict[str, Any],
) -> list[dict[str, Any]]:
    dt = world_model_obj.get("decision_tree")
    if not isinstance(dt, dict):
        return []
    nodes = dt.get("nodes")
    if not isinstance(nodes, list):
        return []
    root_id = str(dt.get("root_id", "") or "root")
    by_id = {
        str(node["node_id"]): node
        for node in nodes
        if isinstance(node, dict) and node.get("node_id")
    }
    frontier: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if _node_solution_id(node) is not None:
            continue
        action = node.get("action")
        if not isinstance(action, dict) or not str(action.get("title") or "").strip():
            continue
        if _action_is_closed_for_strategy_selection(node):
            continue
        parent_id = node.get("parent_id")
        if parent_id is None:
            continue
        parent_id_s = str(parent_id)
        if parent_id_s != root_id:
            parent = by_id.get(parent_id_s)
            if not isinstance(parent, dict) or _node_solution_id(parent) is None:
                continue
        frontier.append(node)
    return frontier


def _frontier_action_event_items(
    world_model_obj: dict[str, Any], *, max_items: int = 20
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for node in _open_frontier_action_nodes(world_model_obj)[: max(0, int(max_items))]:
        action = node.get("action") if isinstance(node.get("action"), dict) else {}
        items.append(
            {
                "node_id": node.get("node_id"),
                "parent_id": node.get("parent_id"),
                "title": action.get("title") or node.get("title"),
                "decision": action.get("decision") or node.get("decision"),
                "difficulty_1_to_5": action.get("difficulty_1_to_5"),
                "score_0_to_1": action.get("score_0_to_1"),
            }
        )
    return items


def _adopted_strategy_ids_from_world_model(
    world_model_obj: dict[str, Any],
    strategy_catalog: list[StrategyCatalogEntry],
) -> list[str]:
    dt = world_model_obj.get("decision_tree")
    nodes = dt.get("nodes") if isinstance(dt, dict) else None
    if not isinstance(nodes, list):
        return []
    adopted: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or _node_solution_id(node) is None:
            continue
        entry = resolve_strategy_catalog_entry(node, strategy_catalog)
        if entry is None or entry.id in seen:
            continue
        seen.add(entry.id)
        adopted.append(entry.id)
    return adopted


def _strategy_lineage_from_world_model(
    world_model_obj: dict[str, Any],
    strategy_catalog: list[StrategyCatalogEntry],
) -> list[dict[str, Any]]:
    dt = world_model_obj.get("decision_tree")
    nodes = dt.get("nodes") if isinstance(dt, dict) else None
    if not isinstance(nodes, list):
        return []
    lineage: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        sid = _node_solution_id(node)
        if sid is None:
            continue
        entry = resolve_strategy_catalog_entry(node, strategy_catalog)
        if entry is None or entry.id in seen:
            continue
        sr = (
            node.get("solution_ref")
            if isinstance(node.get("solution_ref"), dict)
            else {}
        )
        ev = sr.get("eval") if isinstance(sr.get("eval"), dict) else {}
        seen.add(entry.id)
        lineage.append(
            {
                "strategy_id": entry.id,
                "action_node_id": str(node.get("node_id") or ""),
                "solution_id": sid,
                "candidate_id": sr.get("candidate_id"),
                "candidate_manifest_path": sr.get("candidate_manifest_path"),
                "adopted_round": sr.get("round_index"),
                "eval_status": ev.get("status"),
                "mean_latency_us": ev.get("latency_us"),
                "latency_ms": ev.get("latency_ms"),
            }
        )
    return lineage


def _adopted_strategy_node_ids_from_world_model(
    world_model_obj: dict[str, Any],
    strategy_catalog: list[StrategyCatalogEntry],
) -> dict[str, str]:
    dt = world_model_obj.get("decision_tree")
    nodes = dt.get("nodes") if isinstance(dt, dict) else None
    if not isinstance(nodes, list):
        return {}
    out: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict) or _node_solution_id(node) is None:
            continue
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        entry = resolve_strategy_catalog_entry(node, strategy_catalog)
        if entry is not None and entry.id not in out:
            out[entry.id] = node_id
    return out


def _score_0_to_1(node: dict[str, Any]) -> float:
    action = node.get("action") if isinstance(node.get("action"), dict) else {}
    try:
        score = float(action.get("score_0_to_1", 0.0))
    except Exception:
        score = 0.0
    return max(0.0, min(1.0, score))


def _difficulty_1_to_5(node: dict[str, Any]) -> int:
    action = node.get("action") if isinstance(node.get("action"), dict) else {}
    raw = action.get("difficulty_1_to_5", None)
    if raw is None:
        raw = action.get("difficulty_0_to_3", None)
        try:
            raw = int(raw) + 1 if raw is not None else 3
        except Exception:
            raw = 3
    try:
        value = int(raw)
    except Exception:
        value = 3
    return max(1, min(5, value))


def _rating_0_to_1(node: dict[str, Any]) -> float:
    try:
        return float(node.get("overall_rating_0_to_10", 0.0)) / 10.0
    except Exception:
        return 0.0


def _max_allowed_strategy_difficulty(
    *,
    world_model_obj: dict[str, Any],
    selection_policy: WorldModelSelectionPolicy,
) -> int:
    best_vs_base = -1.0
    dt = world_model_obj.get("decision_tree")
    nodes = dt.get("nodes") if isinstance(dt, dict) else None
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            sr = node.get("solution_ref")
            ev = (
                sr.get("eval")
                if isinstance(sr, dict) and isinstance(sr.get("eval"), dict)
                else None
            )
            if (
                not isinstance(ev, dict)
                or str(ev.get("status", "") or "").strip().lower() != "passed"
            ):
                continue
            try:
                best_vs_base = max(
                    best_vs_base, float(ev.get("mean_vs_baseline_factor"))
                )
            except Exception:
                pass
    max_allowed = int(getattr(selection_policy, "max_difficulty_1_to_5", 3) or 3)
    try:
        if best_vs_base >= float(
            getattr(selection_policy, "relax_difficulty_if_best_vs_base_ge", 0.9) or 0.9
        ):
            max_allowed = int(
                getattr(selection_policy, "relaxed_max_difficulty_1_to_5", 4) or 4
            )
    except Exception:
        pass
    return max(1, min(5, max_allowed))


def _sort_executable_strategy_nodes(
    executable: list[tuple[dict[str, Any], StrategyCatalogEntry]],
    *,
    world_model_obj: dict[str, Any],
    selection_policy: WorldModelSelectionPolicy,
) -> list[tuple[dict[str, Any], StrategyCatalogEntry]]:
    max_allowed = _max_allowed_strategy_difficulty(
        world_model_obj=world_model_obj,
        selection_policy=selection_policy,
    )
    filtered = [
        item for item in executable if _difficulty_1_to_5(item[0]) <= max_allowed
    ]
    effective = filtered if filtered else list(executable)
    effective.sort(
        key=lambda item: (
            -_score_0_to_1(item[0]),
            _difficulty_1_to_5(item[0]),
            -_rating_0_to_1(item[0]),
            str(item[0].get("node_id") or ""),
        )
    )
    return effective


def _definition_text_for_codegen_prompt(
    task: Any,
    *,
    language: str,
    has_explicit_base_code: bool,
) -> str:
    """Return task definition text sized for codegen prompts.

    AscendC action/debug prompts already carry the explicit base/current code
    and response format. When the task can provide a spec-only variant, use it
    to avoid duplicating project sources and conflicting format instructions.
    """
    get_def = getattr(task, "get_definition_text", None)
    if not callable(get_def):
        raise RuntimeError(
            f"Task '{getattr(task, 'name', '')}' does not provide get_definition_text(); "
            "cannot build world-model prompts without a definition."
        )

    if has_explicit_base_code:
        try:
            text = get_def(
                language=str(language),
                include_sources=False,
                include_format=False,
            )
            return str(text or "").strip()
        except TypeError:
            pass

    return str(get_def(language=str(language)) or "").strip()


class WorldModelKernelGeneratorWithBaseline(KernelGenerator):
    """Baseline-aware generator variant that maintains and injects a persistent world model."""

    def _strategy_entry_for_node(
        self, node_obj: dict[str, Any] | None
    ) -> StrategyCatalogEntry | None:
        if not isinstance(node_obj, dict) or not self._strategy_catalog:
            return None
        return resolve_strategy_catalog_entry(node_obj, self._strategy_catalog)

    def _strategy_text_for_node(self, node_obj: dict[str, Any] | None) -> str:
        """Render full markdown strategy text for a selected action node."""
        entry = self._strategy_entry_for_node(node_obj)
        if entry is None:
            return ""

        from k_search.kernel_generators.strategy_injection import (
            render_strategy_action_text,
        )

        return render_strategy_action_text(entry=entry)

    def _block_strategy_action_node(
        self,
        *,
        definition_name: str,
        node_id: str,
        reason: str,
        policy: str,
    ) -> None:
        wm_json = self._wm.get(definition_name)
        obj = load_world_model_obj(wm_json or "")
        if obj is None:
            return
        dt = obj.get("decision_tree")
        nodes = dt.get("nodes") if isinstance(dt, dict) else None
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict) or str(node.get("node_id") or "") != str(
                node_id
            ):
                continue
            action = node.get("action") if isinstance(node.get("action"), dict) else {}
            action["status"] = "blocked"
            action["blocked_reason"] = reason
            action["blocked_policy"] = policy
            node["action"] = action
            node["blocked_reason"] = reason
            node["blocked_policy"] = policy
            node["notes"] = f"Blocked by K-Search policy: {reason}; policy={policy}."
            break
        dumped = dump_world_model_obj(obj)
        if dumped:
            self._wm.set(definition_name, dumped)

    def _annotate_strategy_action_node_block(
        self,
        *,
        definition_name: str,
        node_id: str,
        reason: str,
        policy: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        wm_json = self._wm.get(definition_name)
        obj = load_world_model_obj(wm_json or "")
        if obj is None:
            return
        dt = obj.get("decision_tree")
        nodes = dt.get("nodes") if isinstance(dt, dict) else None
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict) or str(node.get("node_id") or "") != str(
                node_id
            ):
                continue
            action = node.get("action") if isinstance(node.get("action"), dict) else {}
            action["blocked_reason"] = reason
            action["blocked_policy"] = policy
            if detail:
                action["blocked_detail"] = dict(detail)
            node["action"] = action
            node["blocked_reason"] = reason
            node["blocked_policy"] = policy
            if detail:
                node["blocked_detail"] = dict(detail)
            node["notes"] = (
                f"Temporarily blocked by K-Search policy: {reason}; policy={policy}."
            )
            break
        dumped = dump_world_model_obj(obj)
        if dumped:
            self._wm.set(definition_name, dumped)

    def _persist_strategy_state_artifact(
        self,
        *,
        task: Any | None,
        run_id: str | None,
        world_model_obj: dict[str, Any],
        blocked_actions: list[dict[str, Any]] | None = None,
    ) -> None:
        if task is None or self._strategy_catalog is None:
            return
        try:
            root = get_run_world_model_dir(
                base_dir=self._artifacts_dir,
                task_name=str(getattr(task, "name", "") or ""),
                run_id=run_id,
            )
            state_path = root / "strategy_state.json"
            adopted = _adopted_strategy_ids_from_world_model(
                world_model_obj, self._strategy_catalog
            )
            lineage = _strategy_lineage_from_world_model(
                world_model_obj, self._strategy_catalog
            )
            current_parent_solution_id = None
            dt = world_model_obj.get("decision_tree")
            active = str(dt.get("active_leaf_id") or "") if isinstance(dt, dict) else ""
            nodes = dt.get("nodes") if isinstance(dt, dict) else None
            if active and isinstance(nodes, list):
                for node in nodes:
                    if (
                        isinstance(node, dict)
                        and str(node.get("node_id") or "") == active
                    ):
                        current_parent_solution_id = _node_solution_id(node)
                        break
            payload = {
                "schema_version": 1,
                "current_parent_solution_id": current_parent_solution_id,
                "adopted_strategy_ids": adopted,
                "strategy_lineage": lineage,
                "blocked_actions": list(blocked_actions or []),
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception:
            pass

    def _persist_blocked_actions_artifact(
        self,
        *,
        task: Any | None,
        run_id: str | None,
        round_index: int | None,
        blocked_actions: list[dict[str, Any]],
        no_executable: bool = False,
    ) -> None:
        if task is None or not blocked_actions:
            return
        try:
            root = get_run_world_model_dir(
                base_dir=self._artifacts_dir,
                task_name=str(getattr(task, "name", "") or ""),
                run_id=run_id,
            )
            path = root / "blocked_actions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for item in blocked_actions:
                    payload = {
                        "round": round_index,
                        "action_node_id": item.get("node_id"),
                        "strategy_id": item.get("strategy_id"),
                        "reason": item.get("reason"),
                        "policy": item.get("policy", "hard_strategy_dependency_gating"),
                    }
                    if "missing" in item:
                        payload["missing"] = list(item.get("missing") or [])
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
                if no_executable:
                    counts: dict[str, int] = {}
                    for item in blocked_actions:
                        reason = str(item.get("reason") or "unknown")
                        counts[reason] = counts.get(reason, 0) + 1
                    f.write(
                        json.dumps(
                            {
                                "event": "no_executable_strategy_node",
                                "round": round_index,
                                "blocked_count": len(blocked_actions),
                                "blocked_reasons": counts,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        except Exception:
            pass

    def _strategy_context_for_node(
        self,
        *,
        definition_name: str,
        node_obj: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(node_obj, dict) or self._strategy_catalog is None:
            return None
        entry = resolve_strategy_catalog_entry(node_obj, self._strategy_catalog)
        if entry is None:
            return None
        wm_obj = load_world_model_obj(self._wm.get(definition_name) or "")
        adopted = (
            _adopted_strategy_ids_from_world_model(wm_obj, self._strategy_catalog)
            if isinstance(wm_obj, dict)
            else []
        )
        adopted_set = set(adopted)
        requires = list(_strategy_requires(entry))
        parent_id = str(node_obj.get("parent_id") or "")
        parent_solution_id = None
        if parent_id:
            try:
                sr = self._wm.get_solution_ref_for_node(
                    definition_name=definition_name, node_id=parent_id
                )
                parent_solution_id = (
                    sr.get("solution_id") if isinstance(sr, dict) else None
                )
            except Exception:
                parent_solution_id = None
        return {
            "strategy_id": entry.id,
            "strategy_markdown_ref": entry.markdown_ref,
            "requires": requires,
            "dependencies_satisfied": all(req in adopted_set for req in requires),
            "parent_strategy_lineage": adopted,
            "parent_solution_id": parent_solution_id,
            "parent_branch_id": parent_id or None,
            "action_node_id": str(node_obj.get("node_id") or ""),
        }

    def _mark_candidate_manifest_adopted(
        self,
        *,
        manifest_path: str | None,
        adoption_reason: str,
    ) -> None:
        if not manifest_path:
            return
        try:
            p = Path(manifest_path)
            if not p.is_file():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            data["adopted"] = True
            data["adoption_reason"] = adoption_reason
            p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass

    def _adopt_passed_candidate_to_active_leaf(
        self,
        *,
        task: Any | None,
        run_id: str | None,
        action_node_id: str | None,
        solution: Any | None,
        eval_result: EvalResult | None,
        round_index: int,
        code_text: str,
        candidate_id: str | None = None,
        candidate_manifest_path: str | None = None,
        changed_paths: list[str] | None = None,
        diff_summary: str | None = None,
        adoption_reason: str = "selected_as_current_parent",
    ) -> str | None:
        """Commit a passed candidate to WM lineage immediately after evaluation."""
        if (
            task is None
            or solution is None
            or eval_result is None
            or self._solution_db is None
        ):
            return None
        if not bool(getattr(eval_result, "is_passed", lambda: False)()):
            return None
        node_id = str(action_node_id or "").strip()
        if not node_id:
            return None
        definition_name = str(getattr(task, "name", "") or "")
        if not definition_name:
            return None

        try:
            solution_id = solution.hash() if hasattr(solution, "hash") else None
        except Exception:
            solution_id = None
        existing_ref: dict[str, Any] = {}
        try:
            existing = self._wm.get_solution_ref_for_node(
                definition_name=definition_name, node_id=node_id
            )
            existing_ref = existing if isinstance(existing, dict) else {}
        except Exception:
            existing_ref = {}

        if not solution_id or str(existing_ref.get("solution_id") or "") != str(
            solution_id
        ):
            rec = self._solution_db.add(
                solution=solution,
                eval_result=eval_result,
                code_text=str(code_text or ""),
                parent_solution_id=None,
            )
            solution_id = rec.solution_id
            self._wm.set_active_leaf_id(
                definition_name=definition_name, node_id=node_id
            )
            self._wm.attach_solution_to_active_leaf(
                definition_name=definition_name,
                solution_id=rec.solution_id,
                solution_name=rec.solution_name,
                eval_result=eval_result,
                round_index=round_index,
                candidate_id=candidate_id,
                candidate_manifest_path=candidate_manifest_path,
                changed_paths=changed_paths,
                diff_summary=diff_summary,
            )

        self._mark_candidate_manifest_adopted(
            manifest_path=candidate_manifest_path,
            adoption_reason=adoption_reason,
        )
        self._persist_world_model_snapshot(task=task, run_id=run_id)
        wm_after_attach = load_world_model_obj(self._wm.get(definition_name) or "")
        if isinstance(wm_after_attach, dict):
            self._persist_strategy_state_artifact(
                task=task,
                run_id=run_id,
                world_model_obj=wm_after_attach,
            )
        return str(solution_id or "") or None

    def _ensure_selected_strategy_parent_lineage(
        self,
        *,
        definition_name: str,
        selected_node_id: str,
        selected_strategy: StrategyCatalogEntry,
    ) -> None:
        requires = _strategy_requires(selected_strategy)
        if not requires:
            return
        wm_obj = load_world_model_obj(self._wm.get(definition_name) or "")
        if wm_obj is None:
            return
        dt = wm_obj.get("decision_tree")
        nodes = dt.get("nodes") if isinstance(dt, dict) else None
        if not isinstance(nodes, list):
            return
        adopted_node_ids = _adopted_strategy_node_ids_from_world_model(
            wm_obj,
            self._strategy_catalog or [],
        )
        target_parent_id = adopted_node_ids.get(requires[-1])
        if not target_parent_id or target_parent_id == selected_node_id:
            return
        by_id = {
            str(node.get("node_id") or ""): node
            for node in nodes
            if isinstance(node, dict) and node.get("node_id")
        }
        selected = by_id.get(selected_node_id)
        target_parent = by_id.get(target_parent_id)
        if not isinstance(selected, dict) or not isinstance(target_parent, dict):
            return
        if str(selected.get("parent_id") or "") == target_parent_id:
            return

        old_parent_id = str(selected.get("parent_id") or "")
        old_parent = by_id.get(old_parent_id)
        if isinstance(old_parent, dict) and isinstance(
            old_parent.get("children"), list
        ):
            old_parent["children"] = [
                child_id
                for child_id in old_parent["children"]
                if str(child_id) != selected_node_id
            ]
        selected["parent_id"] = target_parent_id
        children = target_parent.get("children")
        if not isinstance(children, list):
            children = []
        if selected_node_id not in [str(child_id) for child_id in children]:
            children.append(selected_node_id)
        target_parent["children"] = children
        selected["parent_strategy_lineage"] = list(
            _adopted_strategy_ids_from_world_model(wm_obj, self._strategy_catalog or [])
        )
        selected["parent_lineage_repaired_by"] = "hard_strategy_dependency_gating"
        dumped = dump_world_model_obj(wm_obj)
        if dumped:
            self._wm.set(definition_name, dumped)

    def _choose_executable_strategy_action_node_id(
        self,
        *,
        definition_name: str,
        task: Any | None = None,
        run_id: str | None = None,
        round_index: int | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        blocked: list[dict[str, Any]] = []
        if self._strategy_catalog is None:
            return (
                self._wm.choose_next_action_node_id(definition_name=definition_name),
                blocked,
            )
        wm_json = self._wm.get(definition_name)
        wm_obj = load_world_model_obj(wm_json or "")
        if wm_obj is None:
            return None, blocked

        adopted_strategy_ids = set(
            _adopted_strategy_ids_from_world_model(wm_obj, self._strategy_catalog)
        )
        executable: list[tuple[dict[str, Any], StrategyCatalogEntry]] = []
        hard_policy = "hard_strategy_dependency_gating"
        file_policy = "only_catalog_backed_strategy_nodes_are_executable"

        for node in _open_frontier_action_nodes(wm_obj):
            node_id = str(node.get("node_id") or "")
            entry = resolve_strategy_catalog_entry(node, self._strategy_catalog)
            if entry is None:
                item = {
                    "node_id": node_id,
                    "reason": "strategy_file_required_but_missing",
                    "policy": file_policy,
                }
                blocked.append(item)
                self._block_strategy_action_node(
                    definition_name=definition_name,
                    node_id=node_id,
                    reason=item["reason"],
                    policy=item["policy"],
                )
                continue

            if entry.id in adopted_strategy_ids and not _strategy_allow_reexecute(
                entry
            ):
                item = {
                    "node_id": node_id,
                    "strategy_id": entry.id,
                    "reason": "strategy_already_adopted",
                    "policy": hard_policy,
                }
                blocked.append(item)
                self._block_strategy_action_node(
                    definition_name=definition_name,
                    node_id=node_id,
                    reason=item["reason"],
                    policy=item["policy"],
                )
                continue

            missing = [
                req
                for req in _strategy_requires(entry)
                if req not in adopted_strategy_ids
            ]
            if missing:
                item = {
                    "node_id": node_id,
                    "strategy_id": entry.id,
                    "reason": "missing_prerequisites",
                    "missing": missing,
                    "policy": hard_policy,
                }
                blocked.append(item)
                self._annotate_strategy_action_node_block(
                    definition_name=definition_name,
                    node_id=node_id,
                    reason=item["reason"],
                    policy=item["policy"],
                    detail={"missing": missing},
                )
                continue

            executable.append((node, entry))

        wm_obj_after_blocks = (
            load_world_model_obj(self._wm.get(definition_name) or "") or wm_obj
        )
        self._persist_blocked_actions_artifact(
            task=task,
            run_id=run_id,
            round_index=round_index,
            blocked_actions=blocked,
            no_executable=not executable,
        )
        self._persist_strategy_state_artifact(
            task=task,
            run_id=run_id,
            world_model_obj=wm_obj_after_blocks,
            blocked_actions=blocked,
        )
        try:
            _nar = getattr(self, "_narrative", None)
            if _nar is not None and blocked:
                _nar.blocked_actions(
                    round_num=round_index,
                    actions=blocked,
                    no_executable=not executable,
                )
        except Exception:
            pass
        if not executable:
            raise NoExecutableStrategyNodeError(blocked)

        policy = (
            getattr(getattr(self._wm, "_cfg", None), "selection_policy", None)
            or WorldModelSelectionPolicy()
        )
        selected = _sort_executable_strategy_nodes(
            executable,
            world_model_obj=wm_obj_after_blocks,
            selection_policy=policy,
        )[0]
        selected_node, selected_strategy = selected
        selected_id = str(selected_node.get("node_id") or "").strip()
        if selected_id:
            self._ensure_selected_strategy_parent_lineage(
                definition_name=definition_name,
                selected_node_id=selected_id,
                selected_strategy=selected_strategy,
            )
            wm_obj_after_parent = load_world_model_obj(
                self._wm.get(definition_name) or ""
            )
            if isinstance(wm_obj_after_parent, dict):
                self._persist_strategy_state_artifact(
                    task=task,
                    run_id=run_id,
                    world_model_obj=wm_obj_after_parent,
                    blocked_actions=blocked,
                )
        return selected_id or None, blocked

    def _default_world_model_path(
        self, *, task: Any, run_id: str | None = None, include_run: bool = True
    ) -> Optional[Path]:
        try:
            if include_run:
                return (
                    get_run_world_model_dir(
                        base_dir=self._artifacts_dir,
                        task_name=str(getattr(task, "name", "") or ""),
                        run_id=run_id,
                    )
                    / "world_model.json"
                )
            root = get_ksearch_artifacts_dir(
                base_dir=self._artifacts_dir,
                task_name=str(getattr(task, "name", "") or ""),
                include_run=False,
            )
            return root / "world_model" / "world_model.json"
        except Exception:
            return None

    def _persist_world_model_snapshot(
        self, *, task: Any, run_id: str | None = None
    ) -> None:
        """Best-effort: persist the current WM JSON to disk so future runs can resume."""
        try:
            # Save to current run directory
            p = self._default_world_model_path(
                task=task, run_id=run_id, include_run=True
            )
            if p is not None:
                wm_s = str(
                    self._wm.get(str(getattr(task, "name", "") or "")) or ""
                ).strip()
                if wm_s:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(wm_s, encoding="utf-8")
        except Exception:
            pass

    def _resume_world_model_from_snapshot(
        self, *, task: Any, ref: str, run_id: str | None = None
    ) -> None:
        """
        Load+normalize a world model JSON snapshot and set it into the in-memory WorldModelManager.
        """
        wm_ref = str(ref or "").strip()
        if not wm_ref:
            return
        if wm_ref.lower() == "auto":
            # Try run-specific path first, then fall back to top-level
            p = self._default_world_model_path(
                task=task, run_id=run_id, include_run=True
            )
            if p is None or not p.exists():
                p = self._default_world_model_path(
                    task=task, run_id=None, include_run=False
                )
            if p is None or not p.exists():
                raise FileNotFoundError(
                    "continue-from-world-model=auto but no world_model.json found"
                )
        else:
            p = Path(wm_ref).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"World model JSON not found: {p}")
        raw_wm = p.read_text(encoding="utf-8")
        obj = load_world_model_obj(raw_wm or "")
        if obj is None:
            raise ValueError(
                f"Invalid world model JSON (could not parse/normalize): {p}"
            )
        self._wm.set(str(getattr(task, "name", "") or ""), dump_world_model_obj(obj))

    def _restore_world_model_from_checkpoint(
        self, *, task: Any, restored: RestoredCheckpoint
    ) -> None:
        raw_wm = Path(restored.world_model_path).read_text(encoding="utf-8")
        obj = load_world_model_obj(raw_wm or "")
        if obj is None:
            raise ValueError(
                f"Invalid checkpoint world model JSON: {restored.world_model_path}"
            )
        self._wm.set(str(getattr(task, "name", "") or ""), raw_wm)

    def _save_cycle_checkpoint_if_enabled(self, **kwargs: Any) -> None:
        cfg = getattr(self, "_checkpoint_config", CheckpointConfig())
        if self._checkpoint_manager is None:
            return
        if not (bool(cfg.enabled) or bool(cfg.resume_from)):
            return
        if str(cfg.every or "cycle") != "cycle":
            return
        self._checkpoint_manager.save_cycle_checkpoint(
            **kwargs,
            llm_provider=str(getattr(self, "llm_provider", "") or ""),
            model_name=str(getattr(self, "model_name", "") or ""),
            language=str(self.language),
            target_gpu=str(self.target_gpu),
            resume_mode=str(cfg.resume_mode or "new-run"),
        )

    def _save_attempt_checkpoint_if_enabled(self, **kwargs: Any) -> None:
        cfg = getattr(self, "_checkpoint_config", CheckpointConfig())
        if self._checkpoint_manager is None:
            return
        if not (bool(cfg.enabled) or bool(cfg.resume_from)):
            return
        if str(cfg.every or "cycle") != "attempt":
            return
        self._checkpoint_manager.save_attempt_checkpoint(
            **kwargs,
            llm_provider=str(getattr(self, "llm_provider", "") or ""),
            model_name=str(getattr(self, "model_name", "") or ""),
            language=str(self.language),
            target_gpu=str(self.target_gpu),
            resume_mode=str(cfg.resume_mode or "new-run"),
        )

    def __init__(
        self,
        *args,
        enable_world_model: bool = True,
        # Default higher to allow passing full kernel.cu into WM prompts (we avoid truncating code).
        world_model_max_chars: int = 50000,
        artifacts_dir: str | None = None,
        wm_max_difficulty: int | None = None,
        # Strategy injection: load external strategy catalog and seed WM with strategy-derived nodes.
        strategy_file: str | None = None,
        strategy_form: str | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._world_model_max_chars = int(world_model_max_chars)
        self._artifacts_dir = artifacts_dir
        self._strategy_file = strategy_file
        self._strategy_form = strategy_form
        self._checkpoint_config = checkpoint_config or CheckpointConfig()
        self._checkpoint_manager: CheckpointManager | None = None
        self._restored_checkpoint: RestoredCheckpoint | None = None

        # Load strategy catalog if provided.
        self._strategy_catalog: list[Any] | None = None
        if strategy_file:
            from k_search.kernel_generators.strategy_injection import (
                load_strategy_catalog,
            )

            if strategy_form not in (None, "", "natural_language"):
                raise ValueError(
                    "Only natural_language strategy form is supported. "
                    "Use markdown_ref in strategy catalog."
                )
            self._strategy_catalog = load_strategy_catalog(strategy_file)
            self._strategy_form = "natural_language"
            print(
                f"[STRATEGY] Loaded {len(self._strategy_catalog)} strategies from {strategy_file}, form=natural_language"
            )

        def _llm_call(prompt: str) -> str:
            with llm_log_context(flow="world_model", phase="world_model_manager"):
                return self.llm_client.generate(prompt)

        selection_policy = WorldModelSelectionPolicy()
        if wm_max_difficulty is not None:
            selection_policy.max_difficulty_1_to_5 = int(wm_max_difficulty)

        self._wm = WorldModelManager(
            llm_call=_llm_call,
            target_gpu=self.target_gpu,
            language=self.language,
            config=WorldModelConfig(
                enabled=bool(enable_world_model),
                max_chars_per_block=self._world_model_max_chars,
                selection_policy=selection_policy,
            ),
        )
        # Lazy-init; we need TraceSet.root to choose a persistence location.
        self._solution_db: Optional[SolutionDB] = None

    def generate(  # type: ignore[override]
        self,
        task: Any,
        max_opt_rounds: int = 10,
        baseline_solution: Optional[
            str
        ] = None,  # handled by task; kept for signature compatibility
        *,
        wm_stagnation_window: int = 5,
        num_debug_and_improve_rounds: int = 5,
        continue_from_solution: Optional[str] = None,
        continue_from_world_model: Optional[str] = None,
        continue_from_run: Optional[str] = None,  # New: resume from a historical run
        run_id: Optional[str] = None,  # New: explicit run_id override
        # Workload selection is owned by the Task; configure it when constructing `task`.
    ) -> Any:
        """
        Baseline-aware generator with persistent world model injection/refinement.
        This is a lightly modified copy of KernelGenerator.generate().
        """
        import random
        import time

        try:
            max_dai = int(num_debug_and_improve_rounds)
        except Exception:
            max_dai = 5
        if max_dai < 1:
            max_dai = 1

        # Determine run_id: either explicit, from continue_from_run, or generate new
        from k_search.utils.paths import get_run_id

        if continue_from_run:
            effective_run_id = continue_from_run
        elif run_id:
            effective_run_id = run_id
        else:
            effective_run_id = get_run_id()

        # Store run_id in task for downstream artifact and telemetry lineage.
        setattr(task, "_ksearch_run_id", effective_run_id)

        checkpoint_resume = str(self._checkpoint_config.resume_from or "").strip()
        restored_checkpoint: RestoredCheckpoint | None = None
        if bool(self._checkpoint_config.enabled) or checkpoint_resume:
            checkpoint_every = str(self._checkpoint_config.every or "cycle")
            if checkpoint_every not in {"cycle", "attempt"}:
                raise ValueError("--checkpoint-every must be 'cycle' or 'attempt'")
            self._checkpoint_manager = CheckpointManager(
                artifacts_dir=self._artifacts_dir,
                task_name=str(getattr(task, "name", "") or ""),
                task_id=get_task_id(),
                run_id=effective_run_id,
                config=self._checkpoint_config,
            )
            if checkpoint_resume:
                ref = self._checkpoint_manager.resolve(
                    checkpoint_resume,
                    policy=self._checkpoint_config.resume_policy,
                )
                restored_checkpoint = self._checkpoint_manager.restore_to_run(
                    ref,
                    target_run_id=effective_run_id,
                )
                self._restored_checkpoint = restored_checkpoint
                self._restore_world_model_from_checkpoint(
                    task=task, restored=restored_checkpoint
                )
                if continue_from_solution:
                    _emit_msg = "--resume-from-checkpoint specified; ignoring continue_from_solution"
                    print(f"[WARN] {_emit_msg}", flush=True)
                    continue_from_solution = None
                if continue_from_world_model:
                    _emit_msg = "--resume-from-checkpoint specified; ignoring continue_from_world_model"
                    print(f"[WARN] {_emit_msg}", flush=True)
                    continue_from_world_model = None

        def _stage(msg: str) -> None:
            m = (msg or "").strip()
            if not m:
                return
            print(f"\n[STAGE] {m}", flush=True)

        def _emit(text: str) -> None:
            t = (text or "").strip("\n")
            if not t:
                return
            print(t, flush=True)

        def _code_for_wm_from_raw(raw: Any) -> str:
            return task.code_for_world_model_from_raw(raw=raw, language=self.language)

        def _wm_guardrail(s: Any) -> str:
            """
            Emergency-only guardrail: avoid pathological multi-MB prompts.
            This is NOT a normal cap; typical kernel.cu should pass through unchanged.
            """
            try:
                ss = s if isinstance(s, str) else str(s or "")
                if len(ss) > 200000:
                    return ss[:200000] + "\n...<truncated for safety>...\n"
                return ss
            except Exception:
                return str(s or "")

        # Run-level narrative summary log (human-readable timeline + events.jsonl),
        # under the unified run logs dir <base>/logs/<task>/<run_id>/.
        try:
            _task_name = str(getattr(task, "name", "") or "")
            _run_id = effective_run_id
            self._narrative = RunNarrativeLogger(
                get_ksearch_run_dir(
                    base_dir=self._artifacts_dir, task_name=_task_name, run_id=_run_id
                ),
                meta={
                    "run_id": _run_id,
                    "task_name": _task_name,
                    "model_name": str(getattr(self, "model_name", "") or ""),
                    "llm_provider": str(
                        getattr(self, "llm_provider", "")
                        or getattr(getattr(self, "llm_client", None), "provider", "")
                        or ""
                    ),
                    "target_gpu": str(self.target_gpu),
                    "language": str(self.language),
                    "max_opt_rounds": int(max_opt_rounds),
                    "reference_latency_ms": getattr(task, "reference_latency_ms", None),
                    "artifacts_dir": str(self._artifacts_dir or ""),
                },
            )
            self._narrative.run_start()
        except Exception:
            self._narrative = None

        # Init SolutionDB under k-search artifacts dir (task-agnostic; never use dataset root).
        if self._solution_db is None:
            _stage("init SolutionDB")
            try:
                if (
                    restored_checkpoint is not None
                    and restored_checkpoint.solution_db_path is not None
                ):
                    db_path = restored_checkpoint.solution_db_path
                else:
                    db_path = (
                        get_run_world_model_dir(
                            base_dir=self._artifacts_dir,
                            task_name=str(getattr(task, "name", "") or ""),
                            run_id=effective_run_id,
                        )
                        / "solution_db.jsonl"
                    )
            except Exception:
                db_path = (
                    get_run_world_model_dir(
                        base_dir=self._artifacts_dir,
                        task_name=None,
                        run_id=effective_run_id,
                    )
                    / "solution_db.jsonl"
                )
            self._solution_db = SolutionDB(
                jsonl_path=db_path,
                max_excerpt_chars=self._world_model_max_chars,
            )

        get_def = getattr(task, "get_definition_text", None)
        if callable(get_def):
            definition_text = str(get_def(language=str(self.language)) or "").strip()
            if not definition_text:
                raise RuntimeError(
                    f"Task '{getattr(task, 'name', '')}' returned empty definition text; "
                    "cannot build world-model prompts without a definition."
                )
        else:
            raise RuntimeError(
                f"Task '{getattr(task, 'name', '')}' does not provide get_definition_text(); "
                "cannot build world-model prompts without a definition."
            )

        # Optional: resume world model from a JSON snapshot on disk.
        wm_ref = str(continue_from_world_model or "").strip()
        _wm_already_restored_from_snapshot = False
        if wm_ref and restored_checkpoint is None:
            self._resume_world_model_from_snapshot(
                task=task, ref=wm_ref, run_id=effective_run_id
            )
            _emit(render_world_model_status(self._wm.get(task.name)))
            self._persist_world_model_snapshot(task=task, run_id=effective_run_id)
            _wm_already_restored_from_snapshot = True

        # Optional W&B support
        try:
            import wandb  # type: ignore
        except Exception:  # pragma: no cover
            wandb = None

        # Seed initial code
        current_code = None
        current_raw_code = None
        if restored_checkpoint is not None:
            _stage(f"resume from checkpoint={restored_checkpoint.checkpoint_id}")
            if restored_checkpoint.current_solution is not None:
                current_code, current_raw_code = code_from_solution(
                    self.language,
                    restored_checkpoint.current_solution,
                )
            self._persist_world_model_snapshot(task=task, run_id=effective_run_id)
            _emit(render_world_model_status(self._wm.get(task.name)))
        elif continue_from_solution:
            _stage(f"resume from solution={continue_from_solution}")
            base_sol = task.get_solution(continue_from_solution)
            if base_sol is None:
                raise ValueError(
                    f"Solution '{continue_from_solution}' not found in TraceSet"
                )
            if base_sol.definition != task.name:
                raise ValueError(
                    f"Solution '{continue_from_solution}' does not belong to definition '{task.name}'"
                )
            current_code, current_raw_code = code_from_solution(self.language, base_sol)

            # Evaluate the continued-from solution first (best-effort), then initialize the WM with
            # root attached to this solution. This makes base-vs-cycle comparisons meaningful immediately
            # and lets the LLM "see" the base code+perf at init time.
            _stage(f"seed eval for continue-from solution: {base_sol.name}")
            seed_eval = task.seed_eval_for_base_solution(
                base_solution=base_sol,
            )

            try:
                if self._solution_db is not None:
                    rec_seed = self._solution_db.add(
                        solution=base_sol,
                        eval_result=seed_eval,
                        code_text=str(current_raw_code or ""),
                        parent_solution_id=None,
                    )
                    # Initialize WM (or seed existing WM) with root attached to the continued-from solution.
# Skip this if WM was already restored from a snapshot (which preserves the full tree state).
                    if not _wm_already_restored_from_snapshot:
                        try:
                            wm_code = _wm_guardrail(_code_for_wm_from_raw(current_raw_code))
                        except Exception:
                            wm_code = None
                        with llm_log_context(
                            operator=str(getattr(task, "name", "") or ""),
                            flow="world_model",
                            round_index=0,
                            stage="world_model_seed_init",
                            language=str(self.language),
                            target_gpu=str(self.target_gpu),
                        ):
                            self._wm.ensure_initialized(
                                definition_name=task.name,
                                definition_text=definition_text,
                                current_code_excerpt=(str(wm_code) if isinstance(wm_code, str) and wm_code.strip() else None),
                                eval_result=seed_eval,
                                seed_root_solution_id=str(rec_seed.solution_id),
                                seed_root_solution_name=str(rec_seed.solution_name),
                                seed_root_round_index=0,
                            )
                        _emit("[WM] Initialized+seeded root from continue_from_solution (code+eval).")
                        _emit(render_world_model_status(self._wm.get(task.name)))
                        self._persist_world_model_snapshot(task=task, run_id=effective_run_id)
                    else:
                        _emit("[WM] Skipping ensure_initialized because WM was restored from snapshot.")
                        # WM snapshot already has correct state; no need to modify
                        _emit(render_world_model_status(self._wm.get(task.name)))
            except LLMProviderFatalError as exc:
                _emit(
                    f"[ERROR] world model seed init failed with fatal provider error: {exc}"
                )
                raise
            except Exception as exc:
                _emit(
                    f"[WARN] world model seed init failed: {type(exc).__name__}: {exc}"
                )
        else:
            _stage("initialize world model")
            t0 = time.perf_counter()

            # Strategy injection: if strategy_file is provided, build WM from catalog
            # instead of using LLM-generated initialization.
            if self._strategy_catalog is not None and self._strategy_form:
                from k_search.kernel_generators.strategy_injection import (
                    build_wm_from_strategies,
                )
                _emit(f"[STRATEGY] Building WM from strategy catalog ({len(self._strategy_catalog)} strategies, form=natural_language)")
                wm_obj = build_wm_from_strategies(
                    strategy_catalog=self._strategy_catalog,
                    definition_name=task.name,
                    kernel_summary=str(definition_text[:500] or ""),
                )
                wm_json = dump_world_model_obj(wm_obj)
                self._wm.set(task.name, wm_json)
                wm = wm_json
                _emit("[STRATEGY] WM built from catalog (no LLM init call needed)")
            else:
                with llm_log_context(
                    operator=str(getattr(task, "name", "") or ""),
                    flow="world_model",
                    round_index=0,
                    stage="world_model_init",
                    language=str(self.language),
                    target_gpu=str(self.target_gpu),
                ):
                    wm = self._wm.ensure_initialized(
                        definition_name=task.name, definition_text=definition_text
                    )

            dt = time.perf_counter() - t0
            _emit(render_world_model_status(wm))
            _emit(f"[STAGE] world model init latency: {dt:.2f}s")
            self._persist_world_model_snapshot(task=task, run_id=effective_run_id)
            try:
                _nar = getattr(self, "_narrative", None)
                if _nar is not None:
                    wm_obj_for_events = load_world_model_obj(wm or "")
                    _nar.world_model_init(
                        summary=render_open_action_nodes_block(wm, max_items=8),
                        actions=(
                            _frontier_action_event_items(wm_obj_for_events, max_items=8)
                            if isinstance(wm_obj_for_events, dict)
                            else []
                        ),
                    )
            except Exception:
                pass

            # NOTE: We intentionally do NOT attach `baseline_solution` code to the WM tree.
            # Baseline is used for targets/vs_base evaluation, but should be hidden from the model.
            # The root node's "reference implementation" is embedded as an excerpt in root.notes (spec anchor),
            # and the first PASSED generated kernel becomes the practical base for future actions.
            # Do NOT generate code here; the main loop runs explicit action cycles (N attempts each).
            current_code = None
            current_raw_code = None

        # Use the simpler explicit action-cycle loop (v2). This is much easier to reason about:
        # choose action -> attempt 1 (spec/base + action) -> attempts 2..N (debug_and_improve) -> attach+refine/too-hard.
        return self._generate_world_model_cycles_v2(
            task=task,
            max_opt_rounds=max_opt_rounds,
            wm_stagnation_window=wm_stagnation_window,
            max_dai=max_dai,
            initial_raw_code=(
                current_raw_code if isinstance(current_raw_code, str) else None
            ),
            run_id=effective_run_id,
            start_round=(
                restored_checkpoint.start_round
                if restored_checkpoint is not None
                else 1
            ),
            restored_best_solution=(
                restored_checkpoint.best_solution
                if restored_checkpoint is not None
                else None
            ),
            restored_best_eval=(
                restored_checkpoint.best_eval
                if restored_checkpoint is not None
                else None
            ),
            restored_best_score=(
                restored_checkpoint.best_score
                if restored_checkpoint is not None
                else None
            ),
        )
        # (legacy loop removed; v2 runs all optimization rounds)

    def _generate_world_model_cycles_v2(
        self,
        *,
        task: Any,
        max_opt_rounds: int,
        wm_stagnation_window: int = 5,
        max_dai: int,
        initial_raw_code: Optional[str] = None,
        run_id: Optional[str] = None,
        start_round: int = 1,
        restored_best_solution: Optional[Any] = None,
        restored_best_eval: Optional[EvalResult] = None,
        restored_best_score: Optional[float] = None,
    ) -> Any:
        """
        Simpler state machine:
        - cycle start: pick an open action node
        - attempt 1: generate from (spec+action) if parent=root else (base+action) if parent has solution
        - attempts 2..N: debug_and_improve using logs from the previous attempt
        - cycle end: attach+refine best PASSED in this cycle; else mark action too hard
        """
        effective_run_id = str(
            run_id or getattr(task, "_ksearch_run_id", None) or get_run_id()
        )
        agentic_task_name = (
            str(
                getattr(task, "name", "")
                or getattr(task, "definition_name", "")
                or "ascendc"
            ).strip()
            or "ascendc"
        )
        get_def = getattr(task, "get_definition_text", None)
        if callable(get_def):
            definition_text = str(get_def(language=str(self.language)) or "").strip()
            if not definition_text:
                raise RuntimeError(
                    f"Task '{getattr(task, 'name', '')}' returned empty definition text; "
                    "cannot build world-model prompts without a definition."
                )
        else:
            raise RuntimeError(
                f"Task '{getattr(task, 'name', '')}' does not provide get_definition_text(); "
                "cannot build world-model prompts without a definition."
            )
        baseline_targets_text = str(
            getattr(task, "get_baseline_targets_text", lambda: "")() or ""
        ).strip()

        try:
            import wandb  # type: ignore
        except Exception:  # pragma: no cover
            wandb = None

        def _stage(msg: str) -> None:
            m = (msg or "").strip()
            if not m:
                return
            print(f"\n[STAGE] {m}", flush=True)

        def _emit(text: str) -> None:
            t = (text or "").strip("\n")
            if not t:
                return
            print(t, flush=True)

        def _append_baseline_hint(p: str) -> str:
            if not baseline_targets_text:
                return p
            return (
                p
                + "\n\nPerformance targets (lower is better):\n"
                + baseline_targets_text
                + "\n- Optimize for overall mean latency across the listed workloads while maintaining correctness."
            )

        def _code_format_text() -> str:
            hook = getattr(task, "get_code_format_text", None)
            if not callable(hook):
                return ""
            try:
                return str(
                    hook(language=str(self.language), target_gpu=str(self.target_gpu))
                    or ""
                ).strip()
            except Exception:
                return ""

        def _emit_kernel_cu(cleaned: object) -> None:
            """Print the generated CUDA kernel.cu to stdout (bounded for readability)."""
            if (self.language or "").lower() != "cuda":
                return
            if not isinstance(cleaned, dict):
                return
            cu = cleaned.get("kernel.cu")
            if not isinstance(cu, str) or not cu.strip():
                return
            s = cu.strip()
            _emit("\n[GENERATED kernel.cu]\n" + s + "\n[/GENERATED kernel.cu]\n")

        def _code_for_wm_from_raw(raw: Any) -> str:
            return task.code_for_world_model_from_raw(raw=raw, language=self.language)

        def _wm_guardrail(s: Any) -> str:
            """
            Emergency-only guardrail: avoid pathological multi-MB prompts.
            This is NOT a normal cap; typical kernel.cu should pass through unchanged.
            """
            try:
                ss = s if isinstance(s, str) else str(s or "")
                if len(ss) > 200000:
                    return ss[:200000] + "\n...<truncated for safety>...\n"
                return ss
            except Exception:
                return str(s or "")

        def _code_for_codegen_prompt_from_raw(raw: Any) -> str:
            try:
                code = _wm_guardrail(_code_for_wm_from_raw(raw))
            except Exception:
                code = ""
            if str(code or "").strip():
                return str(code)
            return str(raw or "")

        best_solution: Optional[Any] = restored_best_solution
        best_eval: Optional[EvalResult] = restored_best_eval
        best_score: float = (
            float(restored_best_score) if restored_best_score is not None else -1.0
        )
        run_failed: bool = False

        current_raw_code: Any = str(initial_raw_code or "")
        last_solution: Optional[Any] = restored_best_solution

        # Walk action cycles. Each cycle keeps trying the SAME chosen action node until:
        # - we see no improvements for `stagnation_window` consecutive rounds, OR
        # - we hit max_opt_rounds.
        cycle_start_round = int(start_round or 1)
        while cycle_start_round <= max_opt_rounds:
            try:
                stagnation_window = int(wm_stagnation_window)
            except Exception:
                stagnation_window = 5
            if stagnation_window < 1:
                stagnation_window = 1

            _stage(
                f"world model: select next action (cycle start @ round {cycle_start_round})"
            )
            # When strategy injection is active, skip LLM propose_action_nodes.
            # The action nodes are already seeded from the strategy catalog.
            if self._strategy_catalog is None:
                try:
                    wm_code = _wm_guardrail(_code_for_wm_from_raw(current_raw_code))
                    with llm_log_context(
                        operator=str(getattr(task, "name", "") or ""),
                        flow="world_model",
                        round_index=cycle_start_round,
                        stage="world_model_propose_actions",
                        language=str(self.language),
                        target_gpu=str(self.target_gpu),
                    ):
                        self._wm.propose_action_nodes(
                            definition_name=task.name,
                            definition_text=definition_text,
                            current_code_excerpt=(
                                str(wm_code) if str(wm_code).strip() else None
                            ),
                            current_tree_path=self._wm.get_tree_path_text(
                                definition_name=task.name
                            ),
                            baseline_targets_text=baseline_targets_text,
                            round_index=cycle_start_round,
                        )
                except LLMProviderFatalError as exc:
                    _emit(
                        f"[ERROR] world model action proposal failed with fatal provider error: {exc}"
                    )
                    raise
                except Exception as exc:
                    _emit(
                        f"[WARN] world model action proposal failed: {type(exc).__name__}: {exc}"
                    )
            wm_json = self._wm.get(task.name)
            _emit(render_world_model_status(wm_json))
            _emit(render_open_action_nodes_block(wm_json, max_items=8))

            blocked_strategy_nodes: list[dict[str, Any]] = []
            try:
                if self._strategy_catalog is not None:
                    chosen_leaf, blocked_strategy_nodes = (
                        self._choose_executable_strategy_action_node_id(
                            definition_name=task.name,
                            task=task,
                            run_id=effective_run_id,
                            round_index=cycle_start_round,
                        )
                    )
                else:
                    chosen_leaf = self._wm.choose_next_action_node_id(
                        definition_name=task.name
                    )
            except NoExecutableStrategyNodeError as exc:
                blocked_strategy_nodes = list(exc.blocked or [])
                _emit(
                    "[WARN] No executable strategy-backed open action nodes found; stopping."
                )
                chosen_leaf = None
            except Exception:
                chosen_leaf = None
            if not chosen_leaf:
                _emit("[WARN] No executable open action nodes found; stopping.")
                break

            self._wm.set_active_leaf_id(definition_name=task.name, node_id=chosen_leaf)
            node_obj = self._wm.get_node_obj(
                definition_name=task.name, node_id=chosen_leaf
            )
            chosen_action_text = None
            strategy_text = ""
            selected_strategy_entry = self._strategy_entry_for_node(node_obj)
            blk = render_chosen_action_node_block(node_obj or {})
            if blk.strip():
                chosen_action_text = blk.strip()
                strategy_text = self._strategy_text_for_node(node_obj)
                if strategy_text:
                    chosen_action_text = (
                        chosen_action_text
                        + "\n\nReferenced strategy document:\n"
                        + strategy_text
                    )
                _emit(chosen_action_text)

            try:
                _nar = getattr(self, "_narrative", None)
                if _nar is not None:
                    _no = node_obj if isinstance(node_obj, dict) else {}
                    _act = (
                        _no.get("action") if isinstance(_no.get("action"), dict) else {}
                    )
                    _nar.action_selected(
                        node_id=chosen_leaf,
                        title=_act.get("title") or _no.get("choice"),
                        decision=_no.get("decision"),
                        difficulty=_act.get("difficulty_1_to_5"),
                        score=_act.get("score_0_to_1"),
                        round_num=cycle_start_round,
                    )
            except Exception:
                pass

            parent_id = str((node_obj or {}).get("parent_id") or "root")
            parent_is_root = parent_id == "root"
            selected_strategy_context = self._strategy_context_for_node(
                definition_name=task.name,
                node_obj=node_obj,
            )
            base_raw_code = ""
            base_score: float = (
                -1.0
            )  # comparable to cycle_best_score (task-defined score)
            base_eval: Optional[EvalResult] = None
            if self._solution_db is not None:
                sr = self._wm.get_solution_ref_for_node(
                    definition_name=task.name, node_id=parent_id
                )
                sid = sr.get("solution_id") if isinstance(sr, dict) else None
                # base_score from stored WM eval (no extra benchmarking)
                try:
                    ev = sr.get("eval") if isinstance(sr, dict) else None
                    if isinstance(ev, dict):
                        # Best-effort reconstruct EvalResult so we can surface base perf in prompts.
                        try:
                            base_eval = EvalResult(
                                status=str(ev.get("status", "") or ""),
                                latency_ms=(
                                    float(ev["latency_ms"])
                                    if isinstance(ev.get("latency_ms"), (int, float))
                                    else None
                                ),
                                reference_latency_ms=(
                                    float(ev["reference_latency_ms"])
                                    if isinstance(
                                        ev.get("reference_latency_ms"), (int, float)
                                    )
                                    else None
                                ),
                                mean_vs_baseline_factor=(
                                    float(ev["mean_vs_baseline_factor"])
                                    if isinstance(
                                        ev.get("mean_vs_baseline_factor"), (int, float)
                                    )
                                    else None
                                ),
                                speedup_factor=(
                                    float(ev["speedup_factor"])
                                    if isinstance(
                                        ev.get("speedup_factor"), (int, float)
                                    )
                                    else None
                                ),
                                log_excerpt=str(ev.get("log_excerpt", "") or ""),
                                metrics=(
                                    ev.get("metrics")
                                    if isinstance(ev.get("metrics"), dict)
                                    else {}
                                ),
                            )
                        except Exception:
                            base_eval = None
                        m = (
                            ev.get("metrics")
                            if isinstance(ev.get("metrics"), dict)
                            else None
                        )
                        sc = m.get("score") if isinstance(m, dict) else None
                        base_score = float(sc) if isinstance(sc, (int, float)) else -1.0
                except Exception:
                    base_score = -1.0
                if isinstance(sid, str) and sid.strip():
                    recb = self._solution_db.get(sid)
                    if recb is not None and recb.code:
                        base_raw_code = recb.code
            if not str(base_raw_code or "").strip():
                get_baseline_code = getattr(task, "get_baseline_code_for_codegen", None)
                if callable(get_baseline_code):
                    try:
                        base_raw_code = str(
                            get_baseline_code(language=str(self.language)) or ""
                        ).strip()
                    except TypeError:
                        try:
                            base_raw_code = str(get_baseline_code() or "").strip()
                        except Exception:
                            base_raw_code = ""
                    except Exception:
                        base_raw_code = ""

            prediction = None
            try:
                act = (
                    (node_obj or {}).get("action")
                    if isinstance((node_obj or {}).get("action"), dict)
                    else {}
                )
                expected_speedup = (
                    act.get("expected_speedup")
                    if isinstance(act.get("expected_speedup"), dict)
                    else None
                )
                evb = (
                    expected_speedup.get("factor")
                    if isinstance(expected_speedup, dict)
                    and expected_speedup.get("factor") is not None
                    else act.get("expected_vs_baseline_factor", None)
                )
                prediction = (
                    Prediction(
                        expected_vs_baseline_factor=float(evb),
                        confidence=0.5,
                        rationale=str(act.get("rationale") or "").strip(),
                    )
                    if evb is not None
                    else None
                )
            except Exception:
                prediction = None

            cycle_best_solution: Optional[Any] = None
            cycle_best_eval: Optional[EvalResult] = None
            cycle_best_raw: str = ""
            cycle_best_wm_code: str = ""
            cycle_best_score: float = -1.0
            cycle_best_round: int = 0
            cycle_best_candidate_id: str | None = None
            cycle_best_manifest_path: str | None = None
            cycle_best_changed_paths: list[str] = []
            cycle_best_diff_summary: str = ""
            cycle_best_project_snapshot: Any | None = None
            cycle_best_session_id: str | None = None
            cycle_best_adopted_solution_id: str | None = None
            # Multi-turn SDK session/worktree owner for agentic AscendC (cycle-level).
            agentic_cycle_cm: Any = None
            agentic_cycle: Any = None
            # End the cycle only after this many consecutive non-improving rounds.
            no_improve_streak: int = 0
            # End the cycle if we keep failing to beat the parent/base score for too long (once we have any PASSED solution).
            no_improve_over_base_streak: int = 0
            rounds_consumed: int = 0

            # Last attempt summary (for next debug prompt)
            last_eval: Optional[EvalResult] = None
            round_eval: Optional[EvalResult] = None

            try:
                while True:
                    if cycle_start_round + rounds_consumed > max_opt_rounds:
                        break
                    attempt_idx = rounds_consumed + 1
                    round_num = cycle_start_round + rounds_consumed
                    print(f"\n=== Optimization Round {round_num}/{max_opt_rounds} ===")
                    _emit(
                        f"[CYCLE] action_node_id={chosen_leaf} attempt={attempt_idx} "
                        f"parent_is_root={'yes' if parent_is_root else 'no'} "
                        f"base_code={'yes' if bool(str(base_raw_code or '').strip()) else 'no'}"
                    )

                    _stage(
                        f"codegen: attempt {attempt_idx} (round {round_num}) "
                        f"no_improve_streak={no_improve_streak}/{stagnation_window} "
                        f"no_improve_over_base={no_improve_over_base_streak}/{stagnation_window}"
                    )
                    if not chosen_action_text:
                        raise RuntimeError(
                            "World-model generator expected a chosen action, but chosen_action_text is empty. "
                            f"(round={round_num} attempt={attempt_idx})"
                        )
                    # --- Agentic AscendC codegen branch ---
                    if self._should_use_ascendc_agentic_codegen(task):
                        from k_search.kernel_generators.ascendc_agentic_codegen import (
                            _build_fix_prompt,
                        )

                        trace_excerpt = str(
                            getattr(
                                task, "get_last_round_trace_logs_for_prompt", lambda: ""
                            )()
                            or ""
                        )
                        perf_lines: list[str] = []
                        if last_eval is not None:
                            perf_lines.extend(
                                last_eval.perf_summary_lines(prefix="last_attempt")
                            )
                        if base_eval is not None:
                            perf_lines.extend(
                                base_eval.perf_summary_lines(prefix="base")
                            )
                        perf_summary = "\n".join(perf_lines).strip()

                        if attempt_idx == 1:
                            agentic_mode = "action"
                            agentic_action = str(chosen_action_text or "")
                            base_solution_for_agentic = None
                            if isinstance(base_raw_code, str) and base_raw_code.strip():
                                try:
                                    base_solution_for_agentic = (
                                        task.solution_from_raw_code_for_agentic(
                                            raw_code=base_raw_code,
                                            round_num=round_num,
                                            model_name=str(self.model_name),
                                            target_gpu=str(self.target_gpu),
                                            language=str(self.language),
                                        )
                                    )
                                except Exception:
                                    base_solution_for_agentic = None
                        else:
                            last_attempt_passed = bool(
                                last_eval is not None
                                and getattr(last_eval, "is_passed", lambda: False)()
                            )
                            agentic_mode = "improve" if last_attempt_passed else "debug"
                            agentic_action = (
                                str(chosen_action_text or "")
                                + "\n\nContinue the same action. If the previous attempt failed, fix it first. "
                                "If it passed, improve latency without broadening scope."
                            )
                            base_solution_for_agentic = (
                                last_solution or cycle_best_solution
                            )

                        definition_hook = getattr(
                            task, "get_agentic_definition_text", None
                        )
                        if callable(definition_hook):
                            agentic_definition = str(
                                definition_hook(language=str(self.language)) or ""
                            )
                        else:
                            agentic_definition = _definition_text_for_codegen_prompt(
                                task,
                                language=str(self.language),
                                has_explicit_base_code=False,
                            )
                        request = AscendCAgenticCodegenRequest(
                            definition_text=agentic_definition,
                            action_text=agentic_action,
                            trace_logs=trace_excerpt,
                            perf_summary=perf_summary,
                            target_gpu=str(self.target_gpu),
                            round_num=int(round_num),
                            attempt_idx=int(attempt_idx),
                            mode=agentic_mode,  # type: ignore[arg-type]
                            run_id=effective_run_id,
                            task_name=agentic_task_name,
                            parent_candidate_id=cycle_best_candidate_id,
                            action_node_id=str(chosen_leaf) if chosen_leaf else None,
                            eval_result=last_eval,
                            canonical_strategy_markdown_path=(
                                Path(selected_strategy_entry.markdown_path)
                                if selected_strategy_entry is not None
                                else None
                            ),
                            strategy_summary=(
                                str(blk or "").strip()
                                if str(blk or "").strip()
                                else str(chosen_action_text or "").strip()
                            ),
                            strategy_context=selected_strategy_context,
                            blocked_strategy_nodes=blocked_strategy_nodes,
                        )
                        try:
                            if attempt_idx == 1:
                                # Attempt 1: open a new multi-turn cycle.
                                agentic_cycle_cm = self._agentic_runner().open_cycle(
                                    task=task,
                                    request=request,
                                    base_solution=base_solution_for_agentic,
                                )
                                agentic_cycle = agentic_cycle_cm.__enter__()
                                result = agentic_cycle.run_initial()
                            else:
                                # Attempt 2+: continue in the same session with a fix prompt
                                if agentic_cycle is None:
                                    # No existing session (first attempt failed to produce one);
                                    # fall back to a fresh session.
                                    agentic_cycle_cm = (
                                        self._agentic_runner().open_cycle(
                                            task=task,
                                            request=request,
                                            base_solution=base_solution_for_agentic,
                                        )
                                    )
                                    agentic_cycle = agentic_cycle_cm.__enter__()
                                    result = agentic_cycle.run_initial()
                                else:
                                    agentic_cycle.request = request
                                    if agentic_mode == "improve":
                                        improve_prompt = (
                                            f"Original action intent: {agentic_action}\n\n"
                                            f"Previous attempt passed evaluation.\n\n"
                                            f"Performance summary:\n{perf_summary or '(no performance summary available)'}\n\n"
                                            "Continue only if there is evidence for a focused latency improvement. "
                                            "If the implementation already matches the design and there is no clear "
                                            "performance opportunity, preserve the current implementation and explain that."
                                        )
                                        result = agentic_cycle.continue_improve(
                                            improve_prompt
                                        )
                                    else:
                                        fix_base = _build_fix_prompt(
                                            eval_result=last_eval,
                                            fix_round=attempt_idx - 1,
                                        )
                                        fix_prompt = (
                                            f"Original action intent: {agentic_action}\n\n"
                                            f"{fix_base}\n\n"
                                            f"Continue the same action. Fix the failure first, then improve."
                                        )
                                        result = agentic_cycle.continue_fix(fix_prompt)
                        except LLMProviderFatalError as exc:
                            _emit(
                                f"[ERROR] fatal LLM provider error during agentic codegen: {exc}"
                            )
                            if agentic_cycle_cm is not None:
                                agentic_cycle_cm.__exit__(
                                    type(exc), exc, exc.__traceback__
                                )
                                agentic_cycle_cm = None
                                agentic_cycle = None
                            raise
                        except (TimeoutError, ValueError, RuntimeError) as exc:
                            if agentic_cycle_cm is not None:
                                agentic_cycle_cm.__exit__(
                                    type(exc), exc, exc.__traceback__
                                )
                                agentic_cycle_cm = None
                                agentic_cycle = None
                            diagnosis = diagnose_tool_protocol_failure(
                                trace_path=(
                                    build_attempt_dir(
                                        TelemetryContext(
                                            run_id=effective_run_id,
                                            task_name=agentic_task_name,
                                            definition=getattr(
                                                task, "definition_name", None
                                            )
                                            or agentic_task_name,
                                            flow="agentic_codegen_multi_turn",
                                            stage=str(agentic_mode),
                                            round_index=int(round_num),
                                            attempt_index=int(attempt_idx),
                                            action_node_id=str(chosen_leaf or ""),
                                            model_name=str(self.model_name),
                                            provider="claude-agent",
                                            target_gpu=str(self.target_gpu),
                                            language=str(self.language),
                                        )
                                    )
                                    / "agent_trace.jsonl"
                                )
                            )
                            if self._allow_ascendc_agentic_legacy_fallback():
                                _emit(
                                    f"[WARN] agentic codegen failed for action_node_id={chosen_leaf} "
                                    f"round={round_num}; falling back to legacy prompt path: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                            else:
                                msg = (
                                    f"agentic codegen failed for action_node_id={chosen_leaf} "
                                    f"round={round_num}: {type(exc).__name__}: {exc}"
                                )
                                if diagnosis is not None:
                                    msg = (
                                        f"{msg}\n\n"
                                        f"diagnosis={diagnosis.reason}; retryable={diagnosis.retryable}; "
                                        f"unknown_tool={diagnosis.unknown_tool}; stage={diagnosis.stage or agentic_mode}\n"
                                        f"recovery_prompt: {diagnosis.recovery_prompt}"
                                    )
                                _emit(f"[WARN] {msg}")
                                metrics: dict[str, Any] = {
                                    "score_name": "codegen",
                                    "score": -1.0,
                                }
                                if diagnosis is not None:
                                    metrics.update(diagnosis.to_dict())
                                round_eval = EvalResult(
                                    status="codegen_failed",
                                    log_excerpt=msg,
                                    metrics=metrics,
                                )
                                last_eval = round_eval
                                try:
                                    _nar = getattr(self, "_narrative", None)
                                    if _nar is not None:
                                        _nar.eval_result(
                                            round_num=round_num, eval_result=round_eval
                                        )
                                except Exception:
                                    pass
                                rounds_consumed = max(rounds_consumed, attempt_idx)
                                if (
                                    diagnosis is not None
                                    and diagnosis.retryable
                                    and (cycle_start_round + rounds_consumed)
                                    <= max_opt_rounds
                                ):
                                    _emit(
                                        f"[RETRY] retryable tool protocol error in agentic attempt {attempt_idx}; "
                                        f"continuing with recovery prompt context."
                                    )
                                    continue
                                if diagnosis is not None:
                                    run_failed = True
                                    try:
                                        _nar = getattr(self, "_narrative", None)
                                        if _nar is not None:
                                            _nar.run_failure(
                                                reason=diagnosis.reason,
                                                error_type="ToolProtocolError",
                                                error_message=diagnosis.message,
                                                retryable=diagnosis.retryable,
                                                stage=diagnosis.stage
                                                or str(agentic_mode),
                                                detail=diagnosis.recovery_prompt,
                                                extra={
                                                    "unknown_tool": diagnosis.unknown_tool,
                                                    "tool_use_id": diagnosis.tool_use_id,
                                                },
                                            )
                                    except Exception:
                                        pass
                                else:
                                    run_failed = True
                                    try:
                                        _nar = getattr(self, "_narrative", None)
                                        if _nar is not None:
                                            _nar.run_failure(
                                                reason="codegen_failed",
                                                error_type=type(exc).__name__,
                                                error_message=str(exc),
                                                retryable=False,
                                                stage=str(agentic_mode),
                                                detail=msg,
                                            )
                                    except Exception:
                                        pass
                                break
                        else:
                            solution = result.solution
                            current_code, current_raw_code = code_from_solution(
                                self.language, solution
                            )
                            last_solution = solution
                            current_wm_code = _wm_guardrail(
                                _code_for_wm_from_raw(current_raw_code)
                            )
                            _emit(
                                f"[LLM] agentic ascendc result round={round_num} "
                                f"prompt_chars={result.prompt_chars} "
                                f"changed_files={','.join(result.changed_paths)} "
                                f"project_path={result.project_path}"
                            )
                            _stage(f"use agentic worktree eval (round {round_num})")
                            round_eval = result.eval_result
                            all_passed = bool(
                                getattr(round_eval, "is_passed", lambda: False)()
                            )
                            round_score = float(
                                getattr(round_eval, "score", lambda: -1.0)()
                            )
                            last_eval = round_eval
                            try:
                                _nar = getattr(self, "_narrative", None)
                                if _nar is not None:
                                    _details = []
                                    if isinstance(result.artifact_paths, dict):
                                        for _k in (
                                            "transcript_path",
                                            "prompt_path",
                                            "manifest_path",
                                        ):
                                            _v = result.artifact_paths.get(_k)
                                            if _v:
                                                _details.append(str(_v))
                                    _nar.llm_codegen(
                                        round_num=round_num,
                                        attempt=attempt_idx,
                                        mode=f"agentic/{agentic_mode}",
                                        changed_paths=result.changed_paths,
                                        diff=str(result.diff_text or ""),
                                        detail_paths=_details,
                                    )
                                    _nar.eval_result(
                                        round_num=round_num, eval_result=round_eval
                                    )
                            except Exception:
                                pass
                            if all_passed and round_score > best_score:
                                best_score = float(round_score)
                                best_eval = round_eval
                                best_solution = solution
                                from k_search.kernel_generators.memory import (
                                    save_code_map_if_adopted,
                                    save_knowledge_if_adopted,
                                )

                                save_code_map_if_adopted(
                                    task=task,
                                    code_map_text=getattr(
                                        result, "code_map_text", None
                                    ),
                                    adopted=True,
                                    solution_id=(
                                        solution.hash()
                                        if hasattr(solution, "hash")
                                        else None
                                    ),
                                    parent_solution_id=(
                                        selected_strategy_context.get(
                                            "parent_solution_id"
                                        )
                                        if isinstance(selected_strategy_context, dict)
                                        else None
                                    ),
                                    candidate_id=(
                                        result.candidate_patch.candidate_id
                                        if result.candidate_patch is not None
                                        else None
                                    ),
                                    action_node_id=(
                                        str(chosen_leaf) if chosen_leaf else None
                                    ),
                                    strategy_id=(
                                        selected_strategy_context.get("strategy_id")
                                        if isinstance(selected_strategy_context, dict)
                                        else None
                                    ),
                                    branch_id=str(chosen_leaf) if chosen_leaf else None,
                                    eval_status=str(
                                        getattr(round_eval, "status", "") or ""
                                    ),
                                    speedup_vs_parent=(
                                        round_eval.metrics.get("speedup_vs_parent")
                                        if isinstance(
                                            getattr(round_eval, "metrics", None), dict
                                        )
                                        else None
                                    ),
                                    created_round=int(round_num),
                                    created_attempt=int(attempt_idx),
                                )
                                save_knowledge_if_adopted(
                                    task=task,
                                    knowledge_text=getattr(
                                        result, "knowledge_text", None
                                    ),
                                    adopted=True,
                                )
                            if all_passed:
                                if round_score > cycle_best_score:
                                    cycle_best_score = float(round_score)
                                    cycle_best_eval = round_eval
                                    cycle_best_solution = solution
                                    cycle_best_raw = str(current_raw_code or "")
                                    cycle_best_wm_code = str(current_wm_code or "")
                                    cycle_best_round = int(round_num)
                                    cycle_best_candidate_id = (
                                        result.candidate_patch.candidate_id
                                        if result.candidate_patch is not None
                                        else None
                                    )
                                    cycle_best_manifest_path = (
                                        (result.artifact_paths or {}).get(
                                            "manifest_path"
                                        )
                                        if isinstance(result.artifact_paths, dict)
                                        else None
                                    )
                                    cycle_best_changed_paths = list(
                                        result.changed_paths or []
                                    )
                                    cycle_best_diff_summary = str(
                                        result.diff_text or ""
                                    )[:4000]
                                    cycle_best_project_snapshot = getattr(
                                        result, "project_snapshot", None
                                    )
                                    cycle_best_session_id = getattr(
                                        result, "session_id", None
                                    )
                                    cycle_best_adopted_solution_id = self._adopt_passed_candidate_to_active_leaf(
                                        task=task,
                                        run_id=effective_run_id,
                                        action_node_id=str(chosen_leaf or ""),
                                        solution=cycle_best_solution,
                                        eval_result=cycle_best_eval,
                                        round_index=cycle_best_round,
                                        code_text=cycle_best_raw,
                                        candidate_id=cycle_best_candidate_id,
                                        candidate_manifest_path=cycle_best_manifest_path,
                                        changed_paths=cycle_best_changed_paths,
                                        diff_summary=cycle_best_diff_summary,
                                    )
                                    no_improve_streak = 0
                                else:
                                    no_improve_streak += 1
                            else:
                                no_improve_streak += 1
                            if cycle_best_solution is not None and base_score > 0:
                                if cycle_best_score > base_score:
                                    no_improve_over_base_streak = 0
                                else:
                                    no_improve_over_base_streak += 1
                            self._save_attempt_checkpoint_if_enabled(
                                task=task,
                                round_index=round_num,
                                cycle_start_round=cycle_start_round,
                                attempt_idx=attempt_idx,
                                action_node_id=str(chosen_leaf or ""),
                                next_round=round_num + 1,
                                next_attempt_idx=attempt_idx + 1,
                                world_model_json=self._wm.get(task.name),
                                solution_db_path=(
                                    self._solution_db.jsonl_path
                                    if self._solution_db is not None
                                    else None
                                ),
                                best_solution=best_solution,
                                best_eval=best_eval,
                                best_score=best_score,
                                current_solution=solution,
                                current_eval=round_eval,
                                last_solution=last_solution,
                                last_eval=last_eval,
                                candidate_manifest_path=(
                                    (result.artifact_paths or {}).get("manifest_path")
                                    if isinstance(result.artifact_paths, dict)
                                    else None
                                ),
                                candidate_diff=str(result.diff_text or ""),
                                project_snapshot=getattr(
                                    result, "project_snapshot", None
                                ),
                                claude_session={
                                    "schema_version": 1,
                                    "session_id": getattr(result, "session_id", None),
                                    "cwd": None,
                                    "resume_supported": bool(
                                        getattr(result, "session_id", None)
                                    ),
                                    "file_checkpointing_enabled": False,
                                    "session_store_enabled": False,
                                    "session_store_kind": None,
                                    "notes": "Claude state is auxiliary; K-Search manifest is authoritative.",
                                },
                                max_opt_rounds=max_opt_rounds,
                                wm_stagnation_window=wm_stagnation_window,
                                wm_max_difficulty=getattr(
                                    getattr(
                                        getattr(self._wm, "_cfg", None),
                                        "selection_policy",
                                        None,
                                    ),
                                    "max_difficulty_1_to_5",
                                    None,
                                ),
                            )
                            rounds_consumed += 1
                            if (
                                no_improve_streak >= stagnation_window
                                or no_improve_over_base_streak >= stagnation_window
                            ):
                                break
                            continue
                    elif attempt_idx == 1:
                        # If the action's parent has an attached solution (including root when continuing),
                        # start from that base_code; otherwise fall back to spec+action.
                        if isinstance(base_raw_code, str) and base_raw_code.strip():
                            codegen_definition_text = (
                                _definition_text_for_codegen_prompt(
                                    task,
                                    language=str(self.language),
                                    has_explicit_base_code=True,
                                )
                            )
                            prompt = get_generate_code_from_action_prompt_from_text(
                                self.language,
                                definition_text=codegen_definition_text,
                                base_code=base_raw_code,
                                action_text=chosen_action_text,
                                code_format=_code_format_text(),
                                target_gpu=self.target_gpu,
                            )
                        else:
                            codegen_definition_text = (
                                _definition_text_for_codegen_prompt(
                                    task,
                                    language=str(self.language),
                                    has_explicit_base_code=False,
                                )
                            )
                            prompt = get_generate_code_from_spec_with_action_prompt_from_text(
                                self.language,
                                definition_text=codegen_definition_text,
                                action_text=chosen_action_text,
                                code_format=_code_format_text(),
                                target_gpu=self.target_gpu,
                            )
                    else:
                        if parent_is_root or not base_raw_code:
                            has_passed_in_cycle = cycle_best_solution is not None
                            # Reference base shown in prompts: prefer whichever is better by score (base_score vs cycle_best_score).
                            # If parent is root (no base score), fall back to cycle_best if present.
                            base_for_debug = "(no base code; start from spec)"
                            if isinstance(base_raw_code, str) and base_raw_code.strip():
                                base_for_debug = base_raw_code
                            if (
                                isinstance(cycle_best_raw, str)
                                and cycle_best_raw.strip()
                                and (base_score <= 0 or cycle_best_score > base_score)
                            ):
                                base_for_debug = cycle_best_raw

                            # Perf summary should match the code we include as `base_code` in the prompt.
                            base_perf_eval: Optional[EvalResult] = None
                            if (
                                isinstance(base_for_debug, str)
                                and base_for_debug.strip()
                            ):
                                if (
                                    base_for_debug == cycle_best_raw
                                    and cycle_best_eval is not None
                                ):
                                    base_perf_eval = cycle_best_eval
                                elif (
                                    base_for_debug == base_raw_code
                                    and base_eval is not None
                                ):
                                    base_perf_eval = base_eval

                            perf_summary_lines: list[str] = []
                            if last_eval is not None:
                                perf_summary_lines.extend(
                                    last_eval.perf_summary_lines(prefix="last_attempt")
                                )
                            if base_perf_eval is not None:
                                perf_summary_lines.extend(
                                    base_perf_eval.perf_summary_lines(prefix="base")
                                )
                            perf_summary = "\n".join(perf_summary_lines).strip()
                            current_code_for_prompt = _code_for_codegen_prompt_from_raw(
                                current_raw_code
                            )
                            codegen_definition_text = (
                                _definition_text_for_codegen_prompt(
                                    task,
                                    language=str(self.language),
                                    has_explicit_base_code=bool(
                                        str(base_for_debug or "").strip()
                                        and not str(base_for_debug).startswith(
                                            "(no base code"
                                        )
                                    ),
                                )
                            )
                            if not has_passed_in_cycle:
                                prompt = (
                                    get_debug_and_improve_from_spec_prompt_from_text(
                                        self.language,
                                        definition_text=codegen_definition_text,
                                        trace_logs=str(
                                            getattr(
                                                task,
                                                "get_last_round_trace_logs_for_prompt",
                                                lambda: "",
                                            )()
                                            or ""
                                        ),
                                        current_code=current_code_for_prompt,
                                        action_text=str(chosen_action_text or ""),
                                        code_format=_code_format_text(),
                                        debug_round=min(attempt_idx, max_dai),
                                        max_rounds=max_dai,
                                        target_gpu=self.target_gpu,
                                        perf_summary=perf_summary,
                                        base_code=base_for_debug,
                                    )
                                )
                            else:
                                prompt = get_improve_from_spec_prompt_from_text(
                                    self.language,
                                    definition_text=codegen_definition_text,
                                    trace_logs=str(
                                        getattr(
                                            task,
                                            "get_last_round_trace_logs_for_prompt",
                                            lambda: "",
                                        )()
                                        or ""
                                    ),
                                    current_code=current_code_for_prompt,
                                    code_format=_code_format_text(),
                                    debug_round=min(attempt_idx, max_dai),
                                    max_rounds=max_dai,
                                    target_gpu=self.target_gpu,
                                    perf_summary=perf_summary,
                                    base_code=base_for_debug,
                                )
                        else:
                            has_passed_in_cycle = cycle_best_solution is not None
                            # Reference base shown in prompts: prefer whichever is better by score (base_score vs cycle_best_score).
                            base_for_debug = base_raw_code
                            if (
                                isinstance(cycle_best_raw, str)
                                and cycle_best_raw.strip()
                                and (base_score <= 0 or cycle_best_score > base_score)
                            ):
                                base_for_debug = cycle_best_raw

                            base_perf_eval: Optional[EvalResult] = None
                            if (
                                isinstance(base_for_debug, str)
                                and base_for_debug.strip()
                            ):
                                if (
                                    base_for_debug == cycle_best_raw
                                    and cycle_best_eval is not None
                                ):
                                    base_perf_eval = cycle_best_eval
                                elif (
                                    base_for_debug == base_raw_code
                                    and base_eval is not None
                                ):
                                    base_perf_eval = base_eval

                            perf_summary_lines: list[str] = []
                            if last_eval is not None:
                                perf_summary_lines.extend(
                                    last_eval.perf_summary_lines(prefix="last_attempt")
                                )
                            if base_perf_eval is not None:
                                perf_summary_lines.extend(
                                    base_perf_eval.perf_summary_lines(prefix="base")
                                )
                            perf_summary = "\n".join(perf_summary_lines).strip()
                            current_code_for_prompt = _code_for_codegen_prompt_from_raw(
                                current_raw_code
                            )
                            codegen_definition_text = (
                                _definition_text_for_codegen_prompt(
                                    task,
                                    language=str(self.language),
                                    has_explicit_base_code=bool(
                                        str(base_for_debug or "").strip()
                                    ),
                                )
                            )
                            if not has_passed_in_cycle:
                                prompt = get_debug_generated_code_prompt_from_text(
                                    self.language,
                                    definition_text=codegen_definition_text,
                                    trace_logs=str(
                                        getattr(
                                            task,
                                            "get_last_round_trace_logs_for_prompt",
                                            lambda: "",
                                        )()
                                        or ""
                                    ),
                                    base_code=base_for_debug,
                                    buggy_code=current_code_for_prompt,
                                    action_text=str(chosen_action_text or ""),
                                    code_format=_code_format_text(),
                                    debug_round=min(attempt_idx, max_dai),
                                    max_rounds=max_dai,
                                    target_gpu=self.target_gpu,
                                    perf_summary=perf_summary,
                                )
                            else:
                                prompt = get_improve_generated_code_prompt_from_text(
                                    self.language,
                                    definition_text=codegen_definition_text,
                                    trace_logs=str(
                                        getattr(
                                            task,
                                            "get_last_round_trace_logs_for_prompt",
                                            lambda: "",
                                        )()
                                        or ""
                                    ),
                                    base_code=base_for_debug,
                                    current_code=current_code_for_prompt,
                                    code_format=_code_format_text(),
                                    debug_round=min(attempt_idx, max_dai),
                                    max_rounds=max_dai,
                                    target_gpu=self.target_gpu,
                                    perf_summary=perf_summary,
                                )

                    prompt = (
                        prompt
                        + "\n\n"
                        + render_world_model_section(
                            self._wm.get(task.name),
                            max_chars=self._world_model_max_chars,
                        )
                    )
                    prompt = _append_baseline_hint(prompt)

                    try:
                        with llm_log_context(
                            operator=str(getattr(task, "name", "") or ""),
                            flow="world_model",
                            round_index=round_num,
                            stage=(
                                "action_codegen"
                                if attempt_idx == 1
                                else "debug_codegen"
                            ),
                            action_node_id=str(chosen_leaf or ""),
                            debug_attempt=attempt_idx,
                            max_debug_attempts=max_dai,
                            max_rounds=max_opt_rounds,
                            language=str(self.language),
                            target_gpu=str(self.target_gpu),
                        ):
                            code_result = self._generate_code_from_prompt(
                                prompt, task=task
                            )
                    except LLMProviderFatalError as exc:
                        _emit(f"[ERROR] fatal LLM provider error during codegen: {exc}")
                        raise
                    except (TimeoutError, ValueError, RuntimeError) as exc:
                        msg = (
                            f"codegen failed after retries for action_node_id={chosen_leaf} "
                            f"round={round_num}: {type(exc).__name__}: {exc}"
                        )
                        _emit(f"[WARN] {msg}")
                        round_eval = EvalResult(
                            status="codegen_failed",
                            log_excerpt=msg,
                            metrics={"score_name": "codegen", "score": -1.0},
                        )
                        last_eval = round_eval
                        rounds_consumed = max(rounds_consumed, attempt_idx)
                        run_failed = True
                        try:
                            _nar = getattr(self, "_narrative", None)
                            if _nar is not None:
                                _nar.run_failure(
                                    reason="codegen_failed",
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                    retryable=False,
                                    stage=(
                                        "action_codegen"
                                        if attempt_idx == 1
                                        else "debug_codegen"
                                    ),
                                    detail=msg,
                                )
                        except Exception:
                            pass
                        break
                    current_code = code_result["cleaned"]
                    current_raw_code = code_result["raw"]
                    _emit_kernel_cu(current_code)
                    current_wm_code = (
                        (
                            current_code.get("kernel.cu")
                            if isinstance(current_code, dict)
                            else None
                        )
                        if (self.language or "").lower() == "cuda"
                        else None
                    )
                    if (
                        not isinstance(current_wm_code, str)
                        or not current_wm_code.strip()
                    ):
                        current_wm_code = _code_for_wm_from_raw(current_raw_code)
                    current_wm_code = _wm_guardrail(str(current_wm_code or ""))

                    _stage(
                        f"create Solution object from current code (round {round_num})"
                    )
                    solution = self._create_solution_from_code(
                        cleaned_code=current_code,
                        raw_code=current_raw_code,
                        task=task,
                        round_num=int(round_num),
                    )
                    last_solution = solution
                    _stage(f"evaluate solution (round {round_num})")
                    round_eval = task.run_benchmark(
                        solution=solution,
                        dump_traces=False,
                        round_num=int(round_num),
                    )
                    try:
                        _nar = getattr(self, "_narrative", None)
                        if _nar is not None:
                            _nar.llm_codegen(
                                round_num=round_num,
                                attempt=attempt_idx,
                                mode=(
                                    "action" if attempt_idx == 1 else "debug/improve"
                                ),
                                prompt=prompt,
                                response=str(current_raw_code or ""),
                            )
                            _nar.eval_result(
                                round_num=round_num, eval_result=round_eval
                            )
                    except Exception:
                        pass

                    all_passed = bool(getattr(round_eval, "is_passed", lambda: False)())
                    round_score = float(getattr(round_eval, "score", lambda: -1.0)())

                    # Save as "last attempt" for the next debug prompt
                    last_eval = round_eval

                    # If all workloads passed in this round, log a W&B artifact containing the generated code
                    # (and a WM snapshot) for traceability.
                    if (
                        all_passed
                        and wandb is not None
                        and getattr(wandb, "run", None) is not None
                    ):
                        try:
                            import tempfile

                            art_name = f"r{round_num}_code"
                            artifact = wandb.Artifact(
                                name=art_name,
                                type="generated-code",
                                metadata={
                                    "definition": task.name,
                                    "round": int(round_num),
                                    "solution": solution.name,
                                    "language": self.language,
                                    "target_gpu": self.target_gpu,
                                    "wm_enabled": True,
                                },
                            )
                            with tempfile.TemporaryDirectory() as tmpdir:
                                tmpdir_p = Path(tmpdir)
                                # Cleaned code files
                                if isinstance(current_code, dict):
                                    for filename, content in current_code.items():
                                        p = tmpdir_p / filename
                                        p.parent.mkdir(parents=True, exist_ok=True)
                                        p.write_text(str(content or ""))
                                        artifact.add_file(
                                            str(p), name=f"clean/{filename}"
                                        )
                                else:
                                    p = tmpdir_p / "main.py"
                                    p.parent.mkdir(parents=True, exist_ok=True)
                                    p.write_text(str(current_code or ""))
                                    artifact.add_file(str(p), name="clean/main.py")

                                # Raw code (as generated from the LLM before cleaning)
                                raw_path = tmpdir_p / "raw_code.txt"
                                raw_path.write_text(
                                    str(current_raw_code)
                                    if current_raw_code is not None
                                    else ""
                                )
                                artifact.add_file(
                                    str(raw_path), name="raw/raw_code.txt"
                                )

                                # World model snapshot (best-effort)
                                wm_path = tmpdir_p / "world_model.json"
                                wm_path.write_text(str(self._wm.get(task.name) or ""))
                                artifact.add_file(
                                    str(wm_path), name="wm/world_model.json"
                                )

                                # Round-level eval summary (best-effort)
                                summary_path = tmpdir_p / "round_summary.txt"
                                summary_path.write_text(
                                    (
                                        f"status={getattr(round_eval, 'status', None)}\n"
                                        f"score_name={round_eval.metrics.get('score_name')}\n"
                                        f"score_value={round_eval.metrics.get('score')}\n"
                                        f"mean_vs_baseline_factor={getattr(round_eval, 'mean_vs_baseline_factor', None)}\n"
                                        f"speedup_factor={getattr(round_eval, 'speedup_factor', None)}\n"
                                        f"latency_ms={getattr(round_eval, 'latency_ms', None)}\n"
                                    )
                                )
                                artifact.add_file(
                                    str(summary_path), name="eval/round_summary.txt"
                                )

                            wandb.log_artifact(artifact)
                        except Exception:
                            pass

                    if all_passed and round_score > best_score:
                        best_score = float(round_score)
                        best_eval = round_eval
                        best_solution = solution

                    if all_passed:
                        er = round_eval
                        score = float(getattr(er, "score", lambda: -1.0)())
                        if score > cycle_best_score:
                            cycle_best_score = float(score)
                            cycle_best_eval = er
                            cycle_best_solution = solution
                            cycle_best_raw = str(current_raw_code or "")
                            cycle_best_wm_code = str(current_wm_code or "")
                            cycle_best_round = int(round_num)
                            cycle_best_adopted_solution_id = (
                                self._adopt_passed_candidate_to_active_leaf(
                                    task=task,
                                    run_id=effective_run_id,
                                    action_node_id=str(chosen_leaf or ""),
                                    solution=cycle_best_solution,
                                    eval_result=cycle_best_eval,
                                    round_index=cycle_best_round,
                                    code_text=cycle_best_raw,
                                )
                            )
                            no_improve_streak = 0
                        else:
                            no_improve_streak += 1
                    else:
                        # Failed round (or missing perf): count as no improvement for stagnation purposes.
                        no_improve_streak += 1

                    # "Can't beat base" streak: only meaningful after we have at least one PASSED solution in this cycle,
                    # and only when the parent/base has a meaningful score.
                    if cycle_best_solution is not None and base_score > 0:
                        if cycle_best_score > base_score:
                            no_improve_over_base_streak = 0
                        else:
                            no_improve_over_base_streak += 1

                    self._save_attempt_checkpoint_if_enabled(
                        task=task,
                        round_index=round_num,
                        cycle_start_round=cycle_start_round,
                        attempt_idx=attempt_idx,
                        action_node_id=str(chosen_leaf or ""),
                        next_round=round_num + 1,
                        next_attempt_idx=attempt_idx + 1,
                        world_model_json=self._wm.get(task.name),
                        solution_db_path=(
                            self._solution_db.jsonl_path
                            if self._solution_db is not None
                            else None
                        ),
                        best_solution=best_solution,
                        best_eval=best_eval,
                        best_score=best_score,
                        current_solution=solution,
                        current_eval=round_eval,
                        last_solution=last_solution,
                        last_eval=last_eval,
                        max_opt_rounds=max_opt_rounds,
                        wm_stagnation_window=wm_stagnation_window,
                        wm_max_difficulty=getattr(
                            getattr(
                                getattr(self._wm, "_cfg", None),
                                "selection_policy",
                                None,
                            ),
                            "max_difficulty_1_to_5",
                            None,
                        ),
                    )

                    if wandb is not None and getattr(wandb, "run", None) is not None:
                        try:
                            # Log current round score as well (helps debug regressions / oscillations).
                            round_sn = None
                            try:
                                round_sn = (
                                    round_eval.metrics.get("score_name")
                                    if isinstance(
                                        getattr(round_eval, "metrics", None), dict
                                    )
                                    else None
                                )
                            except Exception:
                                round_sn = None
                            round_key = (
                                f"{task.name}/generate/{round_sn}"
                                if isinstance(round_sn, str) and round_sn
                                else f"{task.name}/generate/round_score"
                            )
                            wandb.log(
                                {
                                    round_key: (
                                        float(round_score)
                                        if (all_passed and round_score > 0)
                                        else None
                                    )
                                },
                                step=round_num,
                            )

                            # Log best-so-far score (single scalar). Before we have any PASSED solution,
                            # `best_eval` is None, so we log None under a stable key.
                            key = (
                                f"{task.name}/generate/best_{best_eval.metrics['score_name']}"
                                if best_eval is not None
                                else f"{task.name}/generate/best_score"
                            )
                            wandb.log(
                                {
                                    key: (
                                        float(best_score)
                                        if best_eval is not None
                                        else None
                                    )
                                },
                                step=round_num,
                            )
                        except Exception:
                            pass

                    rounds_consumed += 1
                    if (
                        no_improve_streak >= stagnation_window
                        or no_improve_over_base_streak >= stagnation_window
                    ):
                        break

                # Close any open multi-turn agentic cycle at cycle end.
                if agentic_cycle_cm is not None:
                    agentic_cycle_cm.__exit__(None, None, None)
                    agentic_cycle_cm = None
                    agentic_cycle = None

            finally:
                if agentic_cycle_cm is not None:
                    agentic_cycle_cm.__exit__(None, None, None)
                    agentic_cycle_cm = None
                    agentic_cycle = None

            if cycle_best_solution is not None and cycle_best_eval is not None:
                _stage(
                    f"cycle end: attach+refine best PASSED (round {cycle_best_round}, score={cycle_best_score:.3f})"
                )
                if self._solution_db is not None and not cycle_best_adopted_solution_id:
                    rec_best = self._solution_db.add(
                        solution=cycle_best_solution,
                        eval_result=cycle_best_eval,
                        code_text=str(cycle_best_raw or ""),
                        parent_solution_id=None,
                    )
                    self._wm.set_active_leaf_id(
                        definition_name=task.name, node_id=chosen_leaf
                    )
                    self._wm.attach_solution_to_active_leaf(
                        definition_name=task.name,
                        solution_id=rec_best.solution_id,
                        solution_name=rec_best.solution_name,
                        eval_result=cycle_best_eval,
                        round_index=cycle_best_round,
                        candidate_id=cycle_best_candidate_id,
                        candidate_manifest_path=cycle_best_manifest_path,
                        changed_paths=cycle_best_changed_paths,
                        diff_summary=cycle_best_diff_summary,
                    )
                    self._mark_candidate_manifest_adopted(
                        manifest_path=cycle_best_manifest_path,
                        adoption_reason="selected_as_current_parent",
                    )
                    _emit(render_world_model_status(self._wm.get(task.name)))
                    self._persist_world_model_snapshot(
                        task=task, run_id=effective_run_id
                    )
                    wm_after_attach = load_world_model_obj(
                        self._wm.get(task.name) or ""
                    )
                    if isinstance(wm_after_attach, dict):
                        self._persist_strategy_state_artifact(
                            task=task,
                            run_id=effective_run_id,
                            world_model_obj=wm_after_attach,
                        )

                with llm_log_context(
                    operator=str(getattr(task, "name", "") or ""),
                    flow="world_model",
                    round_index=cycle_best_round,
                    stage="world_model_refine",
                    action_node_id=str(chosen_leaf or ""),
                    language=str(self.language),
                    target_gpu=str(self.target_gpu),
                ):
                    self._wm.refine(
                        definition_name=task.name,
                        definition_text=definition_text,
                        chosen_action_text=chosen_action_text,
                        current_code_excerpt=_wm_guardrail(
                            str(cycle_best_wm_code or "")
                        ),
                        current_tree_path=self._wm.get_tree_path_text(
                            definition_name=task.name
                        ),
                        eval_result=cycle_best_eval,
                        prediction=prediction,
                        round_index=cycle_best_round,
                    )
                _emit(render_world_model_status(self._wm.get(task.name)))
                self._persist_world_model_snapshot(task=task, run_id=effective_run_id)
                try:
                    _nar = getattr(self, "_narrative", None)
                    if _nar is not None:
                        _nar.world_model_update(
                            kind="attach+refine",
                            round_num=cycle_best_round,
                            detail=f"attach solution to {chosen_leaf}; refine world model (score={cycle_best_score:.3f})",
                            prediction=prediction,
                        )
                except Exception:
                    pass
                self._save_cycle_checkpoint_if_enabled(
                    task=task,
                    round_index=cycle_best_round,
                    cycle_start_round=cycle_start_round,
                    action_node_id=str(chosen_leaf or ""),
                    next_round=cycle_start_round + max(1, rounds_consumed),
                    world_model_json=self._wm.get(task.name),
                    solution_db_path=(
                        self._solution_db.jsonl_path
                        if self._solution_db is not None
                        else None
                    ),
                    best_solution=best_solution,
                    best_eval=best_eval,
                    best_score=best_score,
                    current_solution=cycle_best_solution,
                    current_eval=cycle_best_eval,
                    cycle_best_solution=cycle_best_solution,
                    cycle_best_eval=cycle_best_eval,
                    cycle_best_score=cycle_best_score,
                    candidate_manifest_path=cycle_best_manifest_path,
                    candidate_diff=cycle_best_diff_summary,
                    project_snapshot=cycle_best_project_snapshot,
                    claude_session={
                        "schema_version": 1,
                        "session_id": cycle_best_session_id,
                        "cwd": None,
                        "resume_supported": bool(cycle_best_session_id),
                        "file_checkpointing_enabled": False,
                        "session_store_enabled": False,
                        "session_store_kind": None,
                        "notes": "Claude state is auxiliary; K-Search manifest is authoritative.",
                    },
                    max_opt_rounds=max_opt_rounds,
                    wm_stagnation_window=wm_stagnation_window,
                    wm_max_difficulty=getattr(
                        getattr(
                            getattr(self._wm, "_cfg", None), "selection_policy", None
                        ),
                        "max_difficulty_1_to_5",
                        None,
                    ),
                )
            else:
                _stage("cycle end: no PASSED solution; mark action too hard")
                try:
                    er_fail = None
                    if round_eval is not None:
                        # Task guarantees failed evals contain only failure status + logs (no partial perf fields).
                        er_fail = round_eval
                    with llm_log_context(
                        operator=str(getattr(task, "name", "") or ""),
                        flow="world_model",
                        round_index=cycle_start_round + max(0, rounds_consumed - 1),
                        stage="world_model_mark_too_hard",
                        action_node_id=str(chosen_leaf or ""),
                        debug_attempt=min(rounds_consumed, max_dai),
                        max_debug_attempts=max_dai,
                        language=str(self.language),
                        target_gpu=str(self.target_gpu),
                    ):
                        self._wm.note_action_too_hard(
                            definition_name=task.name,
                            definition_text=definition_text,
                            chosen_action_text=chosen_action_text,
                            current_code_excerpt=_wm_guardrail(
                                str(_code_for_wm_from_raw(current_raw_code) or "")
                            ),
                            current_tree_path=self._wm.get_tree_path_text(
                                definition_name=task.name
                            ),
                            eval_result=er_fail,
                            debug_and_improve_round=min(rounds_consumed, max_dai),
                            debug_and_improve_max_rounds=max_dai,
                            baseline_targets_text=baseline_targets_text,
                            round_index=cycle_start_round + max(0, rounds_consumed - 1),
                        )
                    _emit(render_world_model_status(self._wm.get(task.name)))
                    self._persist_world_model_snapshot(
                        task=task, run_id=effective_run_id
                    )
                    try:
                        _nar = getattr(self, "_narrative", None)
                        if _nar is not None:
                            _nar.world_model_update(
                                kind="too_hard",
                                round_num=cycle_start_round
                                + max(0, rounds_consumed - 1),
                                detail=f"action {chosen_leaf} marked too hard after {rounds_consumed} round(s)",
                            )
                    except Exception:
                        pass
                except LLMProviderFatalError as exc:
                    _emit(
                        f"[ERROR] world model too-hard update failed with fatal provider error: {exc}"
                    )
                    raise
                except Exception as exc:
                    _emit(
                        f"[WARN] world model too-hard update failed: {type(exc).__name__}: {exc}"
                    )
                self._save_cycle_checkpoint_if_enabled(
                    task=task,
                    round_index=cycle_start_round + max(0, rounds_consumed - 1),
                    cycle_start_round=cycle_start_round,
                    action_node_id=str(chosen_leaf or ""),
                    next_round=cycle_start_round + max(1, rounds_consumed),
                    world_model_json=self._wm.get(task.name),
                    solution_db_path=(
                        self._solution_db.jsonl_path
                        if self._solution_db is not None
                        else None
                    ),
                    best_solution=best_solution,
                    best_eval=best_eval,
                    best_score=best_score,
                    current_solution=last_solution,
                    current_eval=round_eval,
                    last_solution=last_solution,
                    last_eval=last_eval,
                    max_opt_rounds=max_opt_rounds,
                    wm_stagnation_window=wm_stagnation_window,
                    wm_max_difficulty=getattr(
                        getattr(
                            getattr(self._wm, "_cfg", None), "selection_policy", None
                        ),
                        "max_difficulty_1_to_5",
                        None,
                    ),
                )

            cycle_start_round += max(1, rounds_consumed)

        # Run-level narrative end marker (best-effort).
        try:
            _nar = getattr(self, "_narrative", None)
            if _nar is not None and not run_failed:
                _be = best_eval
                _nar.run_end(
                    best_round=(
                        getattr(_be, "metrics", {}).get("round")
                        if isinstance(getattr(_be, "metrics", None), dict)
                        else None
                    ),
                    latency_ms=getattr(_be, "latency_ms", None),
                    vs_baseline=getattr(_be, "mean_vs_baseline_factor", None),
                    total_rounds=cycle_start_round - 1,
                )
        except Exception:
            pass

        # Fall back to the best observed solution, else the last attempted.
        if best_solution is not None:
            return best_solution
        if last_solution is not None:
            return last_solution
        raise ValueError(
            f"[{task.name}] No solution was generated (best_solution and last_solution are None)."
        )
