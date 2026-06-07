from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from k_search.utils.paths import get_ksearch_artifacts_dir
from k_search.kernel_generators.code_map_lineage import (
    CodeMapReuseContext,
    build_code_map_meta,
    should_reuse_code_map,
)


@dataclass(frozen=True)
class MemoryKind:
    """Describes one persisted memory type (code_map today; plan/review later)."""

    name: str
    filename: str
    gated_writeback: bool = True
    metadata_filename: str | None = None


CODE_MAP = MemoryKind(
    name="code_map",
    filename="CODE_MAP.md",
    gated_writeback=True,
    metadata_filename="CODE_MAP_META.json",
)

# Cross-round distilled AscendC/debug knowledge produced by the knowledge-curator
# agent after evaluation. Same persistence mechanics as CODE_MAP: written back only
# when the attempt was adopted (gated), materialized into the next round's worktree
# so plan/codegen can read accumulated lessons.
KNOWLEDGE = MemoryKind(name="knowledge", filename="KNOWLEDGE.md", gated_writeback=True)


class MemoryStore:
    """Persists per-task memory under <base>/<task>/<task_id>/artifacts/memory."""

    def __init__(self, *, artifacts_dir: str | Path | None, task_name: str | None) -> None:
        self._artifacts_dir = artifacts_dir
        self._task_name = str(task_name or "") or None

    @classmethod
    def for_task(cls, task: object) -> "MemoryStore":
        artifacts_dir = getattr(task, "artifacts_dir", None)
        task_name = getattr(task, "definition_name", None) or getattr(task, "name", None)
        return cls(artifacts_dir=artifacts_dir, task_name=task_name)

    def _path(self, kind: MemoryKind) -> Path:
        base = get_ksearch_artifacts_dir(
            base_dir=self._artifacts_dir,
            task_name=self._task_name,
            include_run=False,
        )
        return base / "memory" / kind.name / kind.filename

    def _meta_path(self, kind: MemoryKind) -> Path | None:
        if not kind.metadata_filename:
            return None
        return self._path(kind).with_name(kind.metadata_filename)

    def load(self, kind: MemoryKind) -> str | None:
        p = self._path(kind)
        if not p.is_file():
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
        return text if text.strip() else None

    def load_meta(self, kind: MemoryKind) -> dict[str, Any] | None:
        p = self._meta_path(kind)
        if p is None or not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def save(self, kind: MemoryKind, text: str | None, *, meta: dict[str, Any] | None = None) -> None:
        if not text or not str(text).strip():
            return
        p = self._path(kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(text), encoding="utf-8")
        meta_path = self._meta_path(kind)
        if meta_path is not None and meta is not None:
            meta_path.write_text(json.dumps(dict(meta), indent=2, sort_keys=True), encoding="utf-8")

    def materialize(
        self,
        kind: MemoryKind,
        project_dir: str | Path,
        *,
        code_map_reuse_context: CodeMapReuseContext | None = None,
    ) -> bool:
        """Copy saved memory into the worktree so an agent can Read it. Returns True if written."""
        text = self.load(kind)
        if text is None:
            return False
        if kind == CODE_MAP and code_map_reuse_context is not None:
            if not should_reuse_code_map(self.load_meta(kind), code_map_reuse_context):
                return False
        dest = Path(project_dir).expanduser().resolve() / kind.filename
        dest.write_text(text, encoding="utf-8")
        return True

    def read_from_worktree(self, kind: MemoryKind, project_dir: str | Path) -> str | None:
        p = Path(project_dir).expanduser().resolve() / kind.filename
        if not p.is_file():
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
        return text if text.strip() else None


def save_code_map_if_adopted(
    *,
    task: object,
    code_map_text: str | None,
    adopted: bool,
    solution_id: str | None = None,
    parent_solution_id: str | None = None,
    candidate_id: str | None = None,
    action_node_id: str | None = None,
    strategy_id: str | None = None,
    branch_id: str | None = None,
    eval_status: str | None = None,
    speedup_vs_parent: float | None = None,
    created_round: int | None = None,
    created_attempt: int | None = None,
) -> None:
    """Write code_map back to artifacts only when the attempt was adopted (new best)."""
    if not adopted or not code_map_text or not str(code_map_text).strip():
        return
    meta = build_code_map_meta(
        solution_id=solution_id,
        parent_solution_id=parent_solution_id,
        candidate_id=candidate_id,
        action_node_id=action_node_id,
        strategy_id=strategy_id,
        branch_id=branch_id,
        adopted=True,
        eval_status=eval_status,
        speedup_vs_parent=speedup_vs_parent,
        created_round=created_round,
        created_attempt=created_attempt,
    )
    MemoryStore.for_task(task).save(CODE_MAP, code_map_text, meta=meta)


def save_knowledge_if_adopted(*, task: object, knowledge_text: str | None, adopted: bool) -> None:
    """Write curator-distilled knowledge back to artifacts only when the attempt was adopted.

    Mirrors save_code_map_if_adopted so distilled lessons accumulate across rounds
    without being polluted by transient failures.
    """
    if not adopted or not knowledge_text or not str(knowledge_text).strip():
        return
    MemoryStore.for_task(task).save(KNOWLEDGE, knowledge_text)
