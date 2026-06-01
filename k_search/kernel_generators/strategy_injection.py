"""Strategy injection module for K-Search world model.

Loads external optimization strategies from a JSON catalog file and injects
them as action nodes into the WM decision tree, enabling strategy-guided
optimization instead of purely LLM-driven search.

Three strategy forms are supported:
- "natural_language": renders the natural_language field from each strategy
- "structured_params": renders the structured_params field as formatted JSON
- "dsl": renders the dsl field from each strategy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


STRATEGY_FORMS = ("natural_language", "structured_params", "dsl")


def load_strategy_catalog(path: str | Path) -> list[dict[str, Any]]:
    """Load a strategy catalog JSON file and return the strategy list."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Strategy catalog not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    catalog = data.get("strategy_catalog", [])
    if not isinstance(catalog, list):
        raise ValueError(f"Expected 'strategy_catalog' to be a list, got {type(catalog).__name__}")
    return catalog


def render_strategy_as_action_text(
    strategy: dict[str, Any],
    form: str = "natural_language",
) -> str:
    """Render a single strategy into action_text for WM codegen prompts.

    Args:
        strategy: A strategy dict from the catalog (must have id, name, and
                  at least one of natural_language/structured_params/dsl).
        form: One of "natural_language", "structured_params", "dsl".

    Returns:
        A string suitable for injection as action_text in WM action prompts.
    """
    form = form.strip().lower()
    if form not in STRATEGY_FORMS:
        raise ValueError(f"Invalid strategy form '{form}'; must be one of {STRATEGY_FORMS}")

    sid = strategy.get("id", "unknown")
    name = strategy.get("name", "unknown")
    category = strategy.get("category", "unknown")
    difficulty = strategy.get("difficulty", 3)
    impact = strategy.get("impact", "medium")

    header = f"Strategy {sid}: {name} (category={category}, impact={impact}, difficulty={difficulty})\n"

    if form == "natural_language":
        text = str(strategy.get("natural_language", "") or "").strip()
        if not text:
            raise ValueError(f"Strategy {sid} has no natural_language field")
        base_text = header + text

    elif form == "structured_params":
        params = strategy.get("structured_params")
        if not isinstance(params, dict):
            raise ValueError(f"Strategy {sid} has no structured_params dict")
        intro = (
            f"Apply this optimization using the following structured parameters.\n"
            f"Each parameter specifies the exact change to make, with constraints and expected impact.\n"
        )
        base_text = header + intro + json.dumps(params, indent=2, ensure_ascii=False)

    elif form == "dsl":
        text = str(strategy.get("dsl", "") or "").strip()
        if not text:
            raise ValueError(f"Strategy {sid} has no dsl field")
        intro = (
            f"Apply this optimization expressed as a domain-specific language (DSL) specification.\n"
            f"The DSL defines the exact transformation, constraints, and expected outcome.\n"
        )
        base_text = header + intro + text

    else:
        raise ValueError(f"Unhandled form: {form}")

    # === Implementation Priorities Section（放在策略描述后面）===
    priorities = strategy.get("implementation_priorities")
    if isinstance(priorities, list) and priorities:
        priority_section = _render_priorities_section(priorities)
        if priority_section:
            base_text = base_text + "\n\n" + priority_section

    # Append API Reference and Anti-Pattern sections if api_references exists
    api_references = strategy.get("api_references")
    if isinstance(api_references, list) and api_references:
        api_section = _render_api_references_section(api_references)
        if api_section:
            base_text += "\n\n" + api_section
        anti_section = _render_anti_patterns_section(api_references, sid)
        if anti_section:
            base_text += "\n\n" + anti_section

    return base_text


def _render_api_references_section(api_references: list[dict[str, Any]]) -> str:
    """Render api_references into an API Reference section for prompt injection.

    Args:
        api_references: List of dicts with api_name, doc_path, summary fields.

    Returns:
        A formatted string with the API Reference section, or empty string
        if api_references is empty.
    """
    if not api_references:
        return ""

    lines = ["=== AscendC API Reference ==="]
    for ref in api_references:
        api_name = ref.get("api_name", "unknown")
        summary = ref.get("summary", "")
        doc_path = ref.get("doc_path", "")
        lines.append(f"- {api_name}: {summary}")
        if doc_path:
            lines.append(f"  Full doc at references/{doc_path}")

    lines.append(
        "If you need more details about any API, read the referenced doc files "
        "before generating code."
    )
    return "\n".join(lines)


def _render_priorities_section(priorities: list[dict[str, Any]]) -> str:
    """Render implementation priorities into a structured section.

    Args:
        priorities: List of dicts with priority, action, dependency fields.

    Returns:
        A formatted string with the Implementation Priority section,
        or empty string if priorities is empty.
    """
    if not priorities:
        return ""

    lines = ["=== Implementation Priority ==="]
    lines.append("Apply ONE change at a time, verify before stacking:")

    for p in priorities:
        priority = p.get("priority", "P?")
        action = p.get("action", "unknown action")
        dep = p.get("dependency")

        lines.append(f"{priority}: {action}")
        if dep:
            lines.append(f"     -> Requires: {dep}")

    lines.append("")
    lines.append("Constraint: Recommended 1-2 files per round.")
    lines.append("-> Single-file changes have 100% success rate in experiments.")
    lines.append("-> Do NOT combine P0+P1+P2 in one round. Each round should change ONE priority level.")

    return "\n".join(lines)


def _render_anti_patterns_section(
    api_references: list[dict[str, Any]],
    strategy_id: str,
) -> str:
    """Render anti-pattern warnings from all api_references into a unified section.

    Args:
        api_references: List of dicts that may contain anti_patterns lists.
        strategy_id: The strategy id used in the anti-pattern label format.

    Returns:
        A formatted string with aggregated anti-pattern warnings, or empty
        string if no anti_patterns exist across all api_references.
    """
    all_patterns: list[dict[str, Any]] = []
    for ref in api_references:
        patterns = ref.get("anti_patterns")
        if isinstance(patterns, list):
            all_patterns.extend(patterns)

    if not all_patterns:
        return ""

    lines = ["=== Anti-Pattern Warnings ==="]
    for ap in all_patterns:
        ap_id = ap.get("id", "unknown")
        pattern = ap.get("pattern", "unknown pattern")
        reason = ap.get("reason", "no reason given")
        source = ap.get("source", "unknown source")
        lines.append(
            f"[AP-{strategy_id}-{ap_id}] Do NOT use {pattern}. "
            f"Reason: {reason}. (Discovered from {source})"
        )
    return "\n".join(lines)


def _text_similarity_ratio(text_a: str, text_b: str) -> float:
    """Compute simple word-level overlap ratio between two strings.

    Returns a value in [0, 1]. Used for dedup: patterns with similarity > 0.8
    are considered duplicates.

    Edge cases: both empty = 1.0, one empty = 0.0.
    """
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def learn_anti_pattern_from_failure(
    strategy_catalog: list[dict[str, Any]],
    strategy_id: str,
    api_name: str,
    new_anti_pattern: dict[str, Any],
    round_index: int,
) -> list[dict[str, Any]]:
    """Learn a new anti-pattern from a failure and append to the catalog.

    Finds the strategy by strategy_id, then finds the api_reference by api_name.
    Deduplicates using _text_similarity_ratio > 0.8.
    Auto-generates id: finds max existing ap number and increments.
    Modifies catalog in-place and returns it.
    Returns unchanged if strategy/api not found or dedup'd.
    """
    # Find the strategy by strategy_id
    strategy = None
    for s in strategy_catalog:
        if s.get("id") == strategy_id:
            strategy = s
            break
    if strategy is None:
        return strategy_catalog

    # Find the api_reference by api_name
    api_references = strategy.get("api_references", [])
    api_ref = None
    for ref in api_references:
        if ref.get("api_name") == api_name:
            api_ref = ref
            break
    if api_ref is None:
        return strategy_catalog

    # Dedup: check similarity against existing patterns
    new_pattern_text = new_anti_pattern.get("pattern", "")
    existing_patterns = api_ref.get("anti_patterns", [])
    for existing in existing_patterns:
        existing_text = existing.get("pattern", "")
        if _text_similarity_ratio(new_pattern_text, existing_text) > 0.8:
            return strategy_catalog

    # Auto-generate id: find max existing ap number across all strategies
    max_ap_num = 0
    for s in strategy_catalog:
        for ref in s.get("api_references", []):
            for ap in ref.get("anti_patterns", []):
                ap_id_str = ap.get("id", "")
                if isinstance(ap_id_str, str) and ap_id_str.startswith("ap"):
                    try:
                        num = int(ap_id_str[2:])
                        if num > max_ap_num:
                            max_ap_num = num
                    except ValueError:
                        pass

    next_id = f"ap{max_ap_num + 1}"

    # Build the entry (only pattern, reason, source, discovered_at; ignore api_name_hint)
    entry = {
        "id": next_id,
        "pattern": new_anti_pattern.get("pattern", "unknown pattern"),
        "reason": new_anti_pattern.get("reason", "no reason given"),
        "source": f"round_{round_index}_failure",
        "discovered_at": round_index,
    }

    # Append to anti_patterns list
    if not isinstance(existing_patterns, list):
        existing_patterns = []
        api_ref["anti_patterns"] = existing_patterns
    existing_patterns.append(entry)

    return strategy_catalog


def _build_action_node(
    idx: int,
    strategy: dict[str, Any],
    form: str,
) -> dict[str, Any]:
    """Build a single action node dict from a strategy."""
    sid = strategy.get("id", f"s{idx}")
    name = strategy.get("name", f"strategy_{idx}")
    category = strategy.get("category", "unknown")
    difficulty = strategy.get("difficulty", 3)
    impact = strategy.get("impact", "medium")
    impact_score = {"low": 0.3, "medium": 0.6, "high": 0.8}.get(impact, 0.5)

    # Priority: expected_speedup_interval.likely > structured_params.expected_speedup.min
    expected_speedup = None

    # First: try expected_speedup_interval.likely
    interval = strategy.get("expected_speedup_interval")
    if isinstance(interval, dict):
        likely = interval.get("likely")
        if isinstance(likely, (int, float)):
            expected_speedup = float(likely)

    # Fallback: structured_params.expected_speedup.min
    if expected_speedup is None:
        sp = strategy.get("structured_params", {})
        if isinstance(sp, dict):
            es = sp.get("expected_speedup")
            if isinstance(es, dict) and "min" in es:
                expected_speedup = float(es.get("min", 1.0))

    action_text = render_strategy_as_action_text(strategy, form)
    title = f"{sid}: {name}"
    description = action_text

    node_id = f"s{idx+1}"
    return {
        "id": node_id,
        "parent_id": "root",
        "description": description[:300] if len(description) > 300 else description,
        "strategy_combination": [sid],
        "mode": "strategy_guided",
        "optimization_type": category,
        "status": "open",
        "score": None,
        "difficulty": difficulty,
        "depth": 1,
        "solution_ref": {
            "eval": {"score": None, "score_name": None, "status": ""},
            "parent_solution_id": None,
            "solution_id": None,
        },
        "parent_code_ref": None,
        "children": [],
        "profiling_insight": None,
        "profiling_evidence": None,
        "failure_type": None,
        "failure_reason": None,
        "retry_count": 0,
        "node_id": node_id,
        "node_type": "action",
        "notes": f"Strategy from catalog, rendered as '{form}'.\n\n{action_text}",
        "overall_rating_0_to_10": {"low": 5, "medium": 7, "high": 8}.get(impact, 6),
        "confidence_0_to_1": {"low": 0.5, "medium": 0.6, "high": 0.7}.get(impact, 0.6),
        "last_updated_round": 0,
        "choice": None,
        "decision": None,
        "action": {
            "title": title,
            "description": action_text,
            "rationale": f"Catalog strategy {sid} ({category}/{impact}). Expected speedup: {expected_speedup or 'unknown'}x.",
            "score_0_to_1": impact_score,
            "difficulty_1_to_5": difficulty,
            "expected_vs_baseline_factor": expected_speedup,
        },
        "impacts": _build_impacts_from_strategy(strategy),
    }


def build_wm_from_strategies(
    strategy_catalog: list[dict[str, Any]],
    form: str = "natural_language",
    *,
    definition_name: str = "multi_query_attention",
    kernel_summary: str = "",
    hw_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete world model JSON from a strategy catalog.

    Creates a decision tree with root node and strategy-derived action nodes.
    Each strategy becomes a child action node of root with the action text
    rendered in the specified form. Nodes are stored as a **list** (matching
    WM convention where each dict has a node_id field).

    Args:
        strategy_catalog: List of strategy dicts from load_strategy_catalog.
        form: Strategy rendering form ("natural_language", "structured_params", "dsl").
        definition_name: Task definition name.
        kernel_summary: Brief kernel summary for WM root.
        hw_params: Hardware parameters dict for WM root.

    Returns:
        A normalized world model dict ready for dump_world_model_obj().
    """
    if hw_params is None:
        hw_params = {
            "chip_model": "910B3",
            "ub_size_bytes": 196608,
            "core_num": 40,
            "cube_core_cnt": 20,
            "vector_core_cnt": 40,
            "peak_bw_gbps": 57.6,
            "l1_size_bytes": 524288,
            "l0a_size_bytes": 65536,
            "l0b_size_bytes": 65536,
            "l0c_size_bytes": 131072,
            "alignment_bytes": 32,
        }

    # Build root node
    children_ids = [f"s{i+1}" for i in range(len(strategy_catalog))]
    root_node = {
        "id": "root",
        "parent_id": None,
        "description": kernel_summary or f"Baseline {definition_name} kernel.",
        "strategy_combination": [],
        "mode": "strategy_guided",
        "optimization_type": "bandwidth",
        "status": "completed",
        "score": 1.0,
        "difficulty": 1,
        "depth": 0,
        "solution_ref": {
            "eval": {"score": None, "score_name": None, "status": ""},
            "parent_solution_id": None,
            "solution_id": None,
        },
        "parent_code_ref": None,
        "children": children_ids,
        "profiling_insight": None,
        "profiling_evidence": None,
        "failure_type": None,
        "failure_reason": None,
        "retry_count": 0,
        "node_id": "root",
        "node_type": "root",
        "notes": f"Strategy-guided optimization using {len(strategy_catalog)} strategies rendered as '{form}'. "
                 f"Each child node represents one strategy from the catalog.",
        "overall_rating_0_to_10": 5,
        "confidence_0_to_1": 0.7,
        "last_updated_round": 0,
        "choice": None,
        "decision": None,
        "action": {
            "title": "",
            "description": "",
            "rationale": "",
            "score_0_to_1": 0.0,
            "difficulty_1_to_5": 1,
            "expected_vs_baseline_factor": None,
        },
        "impacts": {
            "memory_bandwidth": {"rating_0_to_10": 3, "risk": "", "notes": ""},
            "register_pressure": {"rating_0_to_10": 6, "risk": "", "notes": ""},
            "compute_intensity_and_hw_fit": {"rating_0_to_10": 5, "risk": "", "notes": ""},
        },
    }

    # Build action nodes from strategies
    action_nodes = []
    for i, strategy in enumerate(strategy_catalog):
        action_nodes.append(_build_action_node(i, strategy, form))

    # Nodes as a list (matching WM convention)
    all_nodes = [root_node] + action_nodes

    wm = {
        "kernel_summary": kernel_summary or f"{definition_name} optimization via strategy catalog ({form} form)",
        "baseline_performance": {"speedup": 1.0, "time_ms": None},
        "computed_signals": {
            "round_index": 0,
            "trace": {
                "latency_ms": None,
                "mean_vs_baseline_factor": None,
                "reference_latency_ms": None,
                "speedup_factor": None,
                "status": "",
            },
        },
        "decision_tree": {
            "active_leaf_id": "root",
            "root_id": "root",
            "nodes": all_nodes,
        },
        "open_questions": [
            "Which strategy form (natural language / structured params / DSL) yields best optimization results?",
            "What is the optimal strategy ordering for maximum cumulative speedup?",
            "Can strategies be combined for synergistic effects?",
        ],
        "stagnation_count": 0,
        "stagnation_count_vs_base": 0,
        "best_score": 1.0,
        "world_model_active": True,
        "hw_params": hw_params,
        "discovered_strategies": [],
        "solution_db_path": None,
    }

    return wm


def _build_impacts_from_strategy(strategy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build impact ratings from strategy metadata."""
    category = strategy.get("category", "unknown")
    impact = strategy.get("impact", "medium")
    ratings = {"low": 3, "medium": 6, "high": 8}.get(impact, 5)

    base = {
        "memory_bandwidth": {"rating_0_to_10": ratings if category in ("memory", "tiling") else 5, "risk": "", "notes": ""},
        "register_pressure": {"rating_0_to_10": ratings if category in ("memory",) else 5, "risk": "", "notes": ""},
        "compute_intensity_and_hw_fit": {"rating_0_to_10": ratings if category in ("compute", "pipeline") else 5, "risk": "", "notes": ""},
    }

    # Override with structured_params impacts if present
    sp = strategy.get("structured_params", {})
    if isinstance(sp, dict) and "expected_speedup" in sp:
        es = sp["expected_speedup"]
        if isinstance(es, dict):
            base["memory_bandwidth"]["notes"] = f"Expected speedup range: {es}"

    return base


def get_strategy_action_text_for_node(
    strategy_catalog: list[dict[str, Any]],
    node_id: str,
    form: str = "natural_language",
) -> Optional[str]:
    """Look up a strategy by node_id and render it as action_text.

    Node IDs are formatted as s{i+1} where i is the strategy index.
    """
    if not node_id.startswith("s"):
        return None
    try:
        idx = int(node_id[1:]) - 1
    except ValueError:
        return None
    if idx < 0 or idx >= len(strategy_catalog):
        return None
    return render_strategy_as_action_text(strategy_catalog[idx], form)