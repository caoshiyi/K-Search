from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


CodeMapReuseMode = Literal["action", "repair"]


@dataclass(frozen=True)
class CodeMapReuseContext:
    mode: CodeMapReuseMode
    parent_solution_id: str | None = None
    parent_branch_id: str | None = None
    current_candidate_id: str | None = None
    branch_id: str | None = None


@dataclass(frozen=True)
class CodeMapReuseDecision:
    reused: bool
    reason: str
    previous_meta: dict[str, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _meta_excerpt(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    keys = (
        "schema_version",
        "solution_id",
        "parent_solution_id",
        "candidate_id",
        "action_node_id",
        "strategy_id",
        "branch_id",
        "adopted",
        "eval_status",
        "speedup_vs_parent",
        "created_round",
        "created_attempt",
    )
    return {key: meta.get(key) for key in keys if key in meta}


def evaluate_code_map_reuse(
    meta: dict[str, Any] | None,
    ctx: CodeMapReuseContext,
) -> CodeMapReuseDecision:
    if not isinstance(meta, dict) or not meta:
        return CodeMapReuseDecision(False, "missing_code_map_meta", _meta_excerpt(meta))

    if ctx.mode == "repair":
        if (
            _clean(meta.get("candidate_id")) == _clean(ctx.current_candidate_id)
            and _clean(meta.get("branch_id")) == _clean(ctx.branch_id)
        ):
            return CodeMapReuseDecision(True, "same_repair_candidate", _meta_excerpt(meta))
        return CodeMapReuseDecision(False, "repair_candidate_or_branch_mismatch", _meta_excerpt(meta))

    if meta.get("adopted") is not True:
        return CodeMapReuseDecision(False, "not_adopted_or_lineage_mismatch", _meta_excerpt(meta))

    if not _clean(ctx.parent_solution_id):
        return CodeMapReuseDecision(False, "missing_parent_solution_id", _meta_excerpt(meta))
    if _clean(meta.get("solution_id")) != _clean(ctx.parent_solution_id):
        return CodeMapReuseDecision(False, "parent_solution_mismatch", _meta_excerpt(meta))

    if not _clean(ctx.parent_branch_id):
        return CodeMapReuseDecision(False, "missing_parent_branch_id", _meta_excerpt(meta))
    if _clean(meta.get("branch_id")) != _clean(ctx.parent_branch_id):
        return CodeMapReuseDecision(False, "parent_branch_mismatch", _meta_excerpt(meta))

    return CodeMapReuseDecision(True, "adopted_parent_lineage_match", _meta_excerpt(meta))


def should_reuse_code_map(meta: dict[str, Any] | None, ctx: CodeMapReuseContext) -> bool:
    return evaluate_code_map_reuse(meta, ctx).reused


def build_code_map_meta(
    *,
    solution_id: str | None = None,
    parent_solution_id: str | None = None,
    candidate_id: str | None = None,
    action_node_id: str | None = None,
    strategy_id: str | None = None,
    branch_id: str | None = None,
    adopted: bool,
    eval_status: str | None = None,
    speedup_vs_parent: float | None = None,
    created_round: int | None = None,
    created_attempt: int | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "schema_version": 1,
        "solution_id": _clean(solution_id),
        "parent_solution_id": _clean(parent_solution_id),
        "candidate_id": _clean(candidate_id),
        "action_node_id": _clean(action_node_id),
        "strategy_id": _clean(strategy_id),
        "branch_id": _clean(branch_id),
        "adopted": bool(adopted),
        "eval_status": _clean(eval_status),
        "speedup_vs_parent": speedup_vs_parent,
        "created_round": int(created_round) if created_round is not None else None,
        "created_attempt": int(created_attempt) if created_attempt is not None else None,
    }
    return {key: value for key, value in meta.items() if value is not None}
