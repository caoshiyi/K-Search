"""Markdown-backed natural-language strategy injection for K-Search.

Strategy catalogs are concise JSON indexes. Each entry carries searchable
metadata plus a relative reference to a markdown file containing the full
natural-language strategy body. World-model initialization uses only summary
metadata; the full markdown is loaded only after an action node is selected.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from k_search.kernel_generators.baseline_model import normalize_expected_speedup


STRATEGY_FORMS = ("natural_language",)
INLINE_STRATEGY_COMPAT_ENV = "KSEARCH_ALLOW_INLINE_STRATEGY"


class StrategyCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class StrategyCatalogEntry:
    id: str
    title: str
    summary: str
    markdown_ref: str
    markdown_path: Path
    tags: tuple[str, ...] = ()
    difficulty_1_to_5: int = 3
    score_0_to_1: float = 0.5
    expected_vs_baseline_factor: float | None = None
    requires: tuple[str, ...] = ()
    allow_reexecute: bool = False
    expected_speedup: dict[str, Any] | None = None
    inline_natural_language: str | None = None


def _allow_inline_strategy_compat() -> bool:
    value = os.getenv(INLINE_STRATEGY_COMPAT_ENV, "")
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _resolve_markdown_ref(catalog_path: Path, markdown_ref: str) -> Path:
    root = catalog_path.parent.resolve(strict=True)
    p = Path(markdown_ref)
    if p.is_absolute():
        raise ValueError(f"strategy markdown_ref must be relative: {markdown_ref}")

    resolved = (root / p).resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"strategy markdown_ref escapes catalog directory: {markdown_ref}")
    if resolved.suffix.lower() != ".md":
        raise ValueError(f"strategy markdown_ref must point to a .md file: {markdown_ref}")
    if not resolved.is_file():
        raise FileNotFoundError(f"strategy markdown file not found: {resolved}")
    return resolved


def _required_str(raw: dict[str, Any], key: str, strategy_label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"strategy {strategy_label} missing {key}")
    return value.strip()


def _coerce_optional_float(raw: dict[str, Any], key: str, strategy_id: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"strategy {strategy_id} field {key} must be numeric")
    return float(value)


def _coerce_requires(raw: dict[str, Any], strategy_id: str) -> tuple[str, ...]:
    value = raw.get("requires", ())
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise StrategyCatalogError(f"strategy {strategy_id} field requires must be a list of strategy ids")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise StrategyCatalogError(
                f"strategy {strategy_id} field requires must be a list of strategy ids"
            )
        result.append(item.strip())
    return tuple(result)


def _coerce_bool(raw: dict[str, Any], key: str, strategy_id: str, default: bool = False) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return bool(value)
    raise StrategyCatalogError(f"strategy {strategy_id} field {key} must be boolean")


def _coerce_expected_speedup(raw: dict[str, Any], strategy_id: str) -> dict[str, Any] | None:
    try:
        return normalize_expected_speedup(
            raw,
            strategy_requires=_coerce_requires(raw, strategy_id),
            allow_unspecified_legacy=("expected_speedup" not in raw),
        )
    except ValueError as exc:
        raise StrategyCatalogError(f"strategy {strategy_id} field expected_speedup invalid: {exc}") from exc


def _coerce_score(raw: dict[str, Any], strategy_id: str) -> float:
    value = raw.get("score_0_to_1", 0.5)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"strategy {strategy_id} field score_0_to_1 must be numeric")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"strategy {strategy_id} field score_0_to_1 must be in [0, 1]")
    return score


def _coerce_difficulty(raw: dict[str, Any], strategy_id: str) -> int:
    value = raw.get("difficulty_1_to_5", 3)
    if "difficulty" in raw and "difficulty_1_to_5" not in raw:
        value = raw.get("difficulty", 3)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"strategy {strategy_id} field difficulty_1_to_5 must be an integer")
    if not 1 <= value <= 5:
        raise ValueError(f"strategy {strategy_id} field difficulty_1_to_5 must be in [1, 5]")
    return int(value)


def _coerce_tags(raw: dict[str, Any], strategy_id: str) -> tuple[str, ...]:
    tags = raw.get("tags", ())
    if tags is None:
        return ()
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple)):
        raise ValueError(f"strategy {strategy_id} field tags must be a list of strings")
    result: list[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"strategy {strategy_id} field tags must be a list of strings")
        result.append(tag.strip())
    if not result and isinstance(raw.get("category"), str) and raw["category"].strip():
        result.append(raw["category"].strip())
    return tuple(result)


def _validate_strategy_dependencies(entries: list[StrategyCatalogEntry]) -> None:
    by_id = {entry.id: entry for entry in entries}
    for entry in entries:
        for req in entry.requires:
            if req == entry.id:
                raise StrategyCatalogError(f"strategy {entry.id} cannot require itself")
            if req not in by_id:
                raise StrategyCatalogError(f"strategy {entry.id} requires unknown strategy id: {req}")

    graph = {entry.id: list(entry.requires) for entry in entries}
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str, stack: list[str]) -> None:
        if node in visiting:
            cycle = " -> ".join(stack + [node])
            raise StrategyCatalogError(f"strategy dependency cycle detected: {cycle}")
        if node in visited:
            return
        visiting.add(node)
        for dep in graph.get(node, []):
            dfs(dep, stack + [node])
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        dfs(node, [])


def load_strategy_catalog(strategy_file: str | Path) -> list[StrategyCatalogEntry]:
    """Load and validate a markdown-backed natural-language strategy catalog."""
    catalog_path = Path(strategy_file).expanduser().resolve()
    if not catalog_path.exists():
        raise FileNotFoundError(f"Strategy catalog not found: {catalog_path}")
    if not catalog_path.is_file():
        raise ValueError(f"Strategy catalog path is not a file: {catalog_path}")

    with catalog_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Strategy catalog JSON must be an object")

    form = str(data.get("strategy_form", "natural_language") or "").strip().lower()
    if form != "natural_language":
        raise ValueError("Only natural_language strategy form is supported")

    allow_inline = _allow_inline_strategy_compat()
    raw_entries = data.get("strategies")
    if raw_entries is None and allow_inline and isinstance(data.get("strategy_catalog"), list):
        warnings.warn(
            "'strategy_catalog' with inline natural_language is deprecated; "
            "migrate to 'strategies' with markdown_ref.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw_entries = data.get("strategy_catalog")

    if not isinstance(raw_entries, list):
        raise ValueError(f"Expected 'strategies' to be a list, got {type(raw_entries).__name__}")

    entries: list[StrategyCatalogEntry] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(raw_entries):
        label = f"at index {idx}"
        if not isinstance(raw, dict):
            raise ValueError(f"strategy {label} must be an object")

        sid = _required_str(raw, "id", label)
        label = sid
        if sid in seen_ids:
            raise ValueError(f"duplicate strategy id: {sid}")
        seen_ids.add(sid)

        for forbidden in ("structured_params", "dsl"):
            if forbidden in raw:
                raise ValueError(
                    f"strategy {sid} uses unsupported field {forbidden}; "
                    "only natural_language markdown_ref is supported"
                )

        inline_text = None
        if "natural_language" in raw:
            if not allow_inline:
                raise ValueError(
                    f"strategy {sid} contains inline natural_language text. "
                    "Move full strategy text to markdown_ref and keep only summary in JSON."
                )
            inline_value = str(raw.get("natural_language", "") or "").strip()
            if inline_value:
                inline_text = inline_value
            warnings.warn(
                f"strategy {sid} uses deprecated inline natural_language compatibility mode; "
                "migrate to markdown_ref.",
                RuntimeWarning,
                stacklevel=2,
            )

        if allow_inline and "title" not in raw and isinstance(raw.get("name"), str):
            title = str(raw.get("name", "")).strip()
            if not title:
                raise ValueError(f"strategy {sid} missing title")
        else:
            title = _required_str(raw, "title", sid)

        summary = str(raw.get("summary", "") or "").strip()
        if not summary and allow_inline and inline_text:
            summary = inline_text[:800].strip()
        if not summary:
            raise ValueError(f"strategy {sid} missing summary")
        if len(summary) > 1200:
            raise ValueError(f"strategy {sid} summary too long; keep JSON summary concise")

        markdown_ref = str(raw.get("markdown_ref", "") or "").strip()
        if markdown_ref:
            markdown_path = _resolve_markdown_ref(catalog_path, markdown_ref)
            inline_for_entry = None
        elif allow_inline and inline_text:
            markdown_path = catalog_path
            inline_for_entry = inline_text
        else:
            raise ValueError(f"strategy {sid} missing markdown_ref")

        entries.append(
            StrategyCatalogEntry(
                id=sid,
                title=title,
                summary=summary,
                markdown_ref=markdown_ref,
                markdown_path=markdown_path,
                tags=_coerce_tags(raw, sid),
                difficulty_1_to_5=_coerce_difficulty(raw, sid),
                score_0_to_1=_coerce_score(raw, sid),
                expected_vs_baseline_factor=_coerce_optional_float(
                    raw,
                    "expected_vs_baseline_factor",
                    sid,
                ),
                requires=_coerce_requires(raw, sid),
                allow_reexecute=_coerce_bool(raw, "allow_reexecute", sid, default=False),
                expected_speedup=_coerce_expected_speedup(raw, sid),
                inline_natural_language=inline_for_entry,
            )
        )

    _validate_strategy_dependencies(entries)
    return entries


def render_strategy_action_text(
    *,
    entry: StrategyCatalogEntry,
    max_markdown_chars: int = 12000,
) -> str:
    """Read and render the selected strategy's full natural-language text."""
    if not isinstance(entry, StrategyCatalogEntry):
        raise TypeError("entry must be a StrategyCatalogEntry")
    if max_markdown_chars <= 0:
        raise ValueError("max_markdown_chars must be positive")

    if entry.inline_natural_language is not None:
        full_text = entry.inline_natural_language.strip()
    else:
        full_text = entry.markdown_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

    if len(full_text) > max_markdown_chars:
        full_text = full_text[:max_markdown_chars] + "\n\n[truncated strategy markdown]"

    expected_speedup_text = (
        json.dumps(entry.expected_speedup, ensure_ascii=False, sort_keys=True)
        if entry.expected_speedup
        else "(none)"
    )
    return (
        f"Strategy ID: {entry.id}\n"
        f"Strategy title: {entry.title}\n"
        f"Strategy summary: {entry.summary}\n"
        f"Strategy tags: {', '.join(entry.tags) if entry.tags else '(none)'}\n"
        f"Strategy difficulty: {entry.difficulty_1_to_5}/5\n\n"
        f"Expected speedup: {expected_speedup_text}\n\n"
        f"Strategy requires: {', '.join(entry.requires) if entry.requires else '(none)'}\n\n"
        "Full natural-language strategy markdown:\n"
        f"{full_text}"
    )


def render_strategy_as_action_text(
    strategy: StrategyCatalogEntry,
    form: str = "natural_language",
) -> str:
    """Compatibility wrapper around render_strategy_action_text.

    Only natural_language is supported. The old structured_params and dsl
    branches intentionally fail fast.
    """
    form = form.strip().lower()
    if form != "natural_language":
        raise ValueError(
            "Only natural_language strategy form is supported. "
            "Use markdown_ref in strategy catalog."
        )
    return render_strategy_action_text(entry=strategy)


def _render_api_references_section(api_references: list[dict[str, Any]]) -> str:
    """Render api_references into an API Reference section for prompt injection."""
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
    """Render implementation priorities into a structured section."""
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
    """Render anti-pattern warnings from all api_references into a section."""
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
    """Compute simple word-level overlap ratio between two strings."""
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
    """Learn a new anti-pattern from a failure and append to a dict catalog.

    This helper remains dict-based because it mutates experimental anti-pattern
    catalogs, not the markdown-backed production catalog entries.
    """
    strategy = None
    for s in strategy_catalog:
        if s.get("id") == strategy_id:
            strategy = s
            break
    if strategy is None:
        return strategy_catalog

    api_references = strategy.get("api_references", [])
    api_ref = None
    for ref in api_references:
        if ref.get("api_name") == api_name:
            api_ref = ref
            break
    if api_ref is None:
        return strategy_catalog

    new_pattern_text = new_anti_pattern.get("pattern", "")
    existing_patterns = api_ref.get("anti_patterns", [])
    for existing in existing_patterns:
        existing_text = existing.get("pattern", "")
        if _text_similarity_ratio(new_pattern_text, existing_text) > 0.8:
            return strategy_catalog

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
    entry = {
        "id": next_id,
        "pattern": new_anti_pattern.get("pattern", "unknown pattern"),
        "reason": new_anti_pattern.get("reason", "no reason given"),
        "source": f"round_{round_index}_failure",
        "discovered_at": round_index,
    }

    if not isinstance(existing_patterns, list):
        existing_patterns = []
        api_ref["anti_patterns"] = existing_patterns
    existing_patterns.append(entry)

    return strategy_catalog


def _build_action_node(
    idx: int,
    strategy: StrategyCatalogEntry,
    form: str = "natural_language",
) -> dict[str, Any]:
    """Build a single summary-only world-model action node from a strategy."""
    form = form.strip().lower()
    if form != "natural_language":
        raise ValueError(
            "Only natural_language strategy form is supported. "
            "Use markdown_ref in strategy catalog."
        )
    if not isinstance(strategy, StrategyCatalogEntry):
        raise TypeError("strategy must be a StrategyCatalogEntry")

    sid = strategy.id
    node_id = f"s{idx + 1}"
    optimization_type = strategy.tags[0] if strategy.tags else "strategy"
    rating = max(1, min(10, round(strategy.score_0_to_1 * 10)))
    description = strategy.summary

    return {
        "id": node_id,
        "parent_id": "root",
        "description": description[:300] if len(description) > 300 else description,
        "strategy_combination": [sid],
        "mode": "strategy_guided",
        "optimization_type": optimization_type,
        "status": "open",
        "score": None,
        "difficulty": strategy.difficulty_1_to_5,
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
        "notes": (
            "Strategy from catalog. This node stores summary metadata only; "
            "the full markdown strategy is loaded when the action is selected.\n\n"
            f"{description}"
        ),
        "overall_rating_0_to_10": rating,
        "confidence_0_to_1": strategy.score_0_to_1,
        "last_updated_round": 0,
        "choice": None,
        "decision": None,
        "action": {
            "title": strategy.title,
            "description": description,
            "rationale": (
                f"Catalog strategy {sid}. Full natural-language details are loaded from "
                f"{strategy.markdown_ref or 'inline compatibility text'} at execution time."
            ),
            "score_0_to_1": strategy.score_0_to_1,
            "difficulty_1_to_5": strategy.difficulty_1_to_5,
            "expected_speedup": dict(strategy.expected_speedup) if strategy.expected_speedup else None,
            "requires": list(strategy.requires),
            "allow_reexecute": bool(strategy.allow_reexecute),
            "strategy_ref": {
                "id": sid,
                "markdown_ref": strategy.markdown_ref,
            },
        },
        "impacts": _build_impacts_from_strategy(strategy),
    }


def build_wm_from_strategies(
    strategy_catalog: list[StrategyCatalogEntry],
    form: str = "natural_language",
    *,
    definition_name: str = "multi_query_attention",
    kernel_summary: str = "",
    hw_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete world model JSON from summary-only strategy nodes."""
    form = form.strip().lower()
    if form != "natural_language":
        raise ValueError(
            "Only natural_language strategy form is supported. "
            "Use markdown_ref in strategy catalog."
        )

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

    children_ids = [f"s{i + 1}" for i in range(len(strategy_catalog))]
    root_node = {
        "id": "root",
        "parent_id": None,
        "description": kernel_summary or f"Baseline {definition_name} kernel.",
        "strategy_combination": [],
        "mode": "strategy_guided",
        "optimization_type": "strategy",
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
        "notes": (
            f"Strategy-guided optimization using {len(strategy_catalog)} natural-language "
            "strategy summaries. Full markdown documents are loaded only for the "
            "selected action node."
        ),
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
            "expected_speedup": None,
        },
        "impacts": {
            "memory_bandwidth": {"rating_0_to_10": 3, "risk": "", "notes": ""},
            "register_pressure": {"rating_0_to_10": 6, "risk": "", "notes": ""},
            "compute_intensity_and_hw_fit": {"rating_0_to_10": 5, "risk": "", "notes": ""},
        },
    }

    action_nodes = [_build_action_node(i, strategy, form) for i, strategy in enumerate(strategy_catalog)]
    all_nodes = [root_node] + action_nodes

    return {
        "kernel_summary": kernel_summary or f"{definition_name} optimization via strategy catalog",
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
            "Which strategy summary should be executed next?",
            "What evidence confirms the selected markdown strategy improves the kernel?",
            "Can successful strategies be combined after single-strategy validation?",
        ],
        "stagnation_count": 0,
        "stagnation_count_vs_base": 0,
        "best_score": 1.0,
        "world_model_active": True,
        "hw_params": hw_params,
        "discovered_strategies": [],
        "solution_db_path": None,
    }


def _build_impacts_from_strategy(strategy: StrategyCatalogEntry) -> dict[str, dict[str, Any]]:
    """Build broad impact ratings from strategy metadata."""
    rating = max(1, min(10, round(strategy.score_0_to_1 * 10)))
    tags = {tag.lower() for tag in strategy.tags}
    expected_note = f"Expected speedup: {strategy.expected_speedup}" if strategy.expected_speedup else ""
    memory_tags = {"memory", "ub", "l1", "cache", "data-movement", "bandwidth"}
    compute_tags = {"compute", "vector", "pipeline", "tiling"}

    return {
        "memory_bandwidth": {
            "rating_0_to_10": rating if tags & memory_tags else 5,
            "risk": "",
            "notes": expected_note,
        },
        "register_pressure": {
            "rating_0_to_10": rating if "register" in tags else 5,
            "risk": "",
            "notes": "",
        },
        "compute_intensity_and_hw_fit": {
            "rating_0_to_10": rating if tags & compute_tags else 5,
            "risk": "",
            "notes": expected_note if tags & compute_tags else "",
        },
    }


def get_strategy_action_text_for_node(
    strategy_catalog: list[StrategyCatalogEntry],
    node_id: str,
    form: str = "natural_language",
) -> Optional[str]:
    """Look up a strategy by node_id and render its full markdown action text."""
    form = form.strip().lower()
    if form != "natural_language":
        raise ValueError(
            "Only natural_language strategy form is supported. "
            "Use markdown_ref in strategy catalog."
        )
    if not node_id.startswith("s"):
        return None
    try:
        idx = int(node_id[1:]) - 1
    except ValueError:
        return None
    if idx < 0 or idx >= len(strategy_catalog):
        return None
    return render_strategy_action_text(entry=strategy_catalog[idx])
