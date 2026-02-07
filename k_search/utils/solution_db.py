"""A minimal solution database for associating decision-tree nodes with solutions.

Stores:
- solution_id (we use Solution.hash() as a deterministic ID)
- solution_name
- definition
- optional parent_solution_id
- code excerpt (or raw code)
- eval_result summary

Optionally appends to a JSONL file for persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from k_search.tasks.task_base import EvalResult
from k_search.tasks.task_base import Solution


@dataclass
class SolutionRecord:
    solution_id: str
    solution_name: str
    definition: str
    parent_solution_id: Optional[str] = None
    eval_result: Optional[dict] = None
    code: str = ""
    code_excerpt: str = ""


class SolutionDB:
    def __init__(self, *, jsonl_path: Optional[str | Path] = None, max_excerpt_chars: int = 4000):
        self._records: dict[str, SolutionRecord] = {}
        self._max_excerpt_chars = int(max_excerpt_chars)
        self._jsonl_path: Optional[Path] = Path(jsonl_path) if jsonl_path else None
        # Best-effort load for persistence across runs.
        if self._jsonl_path is not None and self._jsonl_path.exists():
            try:
                for line in self._jsonl_path.read_text(encoding="utf-8").splitlines():
                    s = (line or "").strip()
                    if not s:
                        continue
                    obj = json.loads(s)
                    if not isinstance(obj, dict):
                        continue
                    # Be tolerant to schema drift: only keep known fields.
                    rec = SolutionRecord(
                        solution_id=str(obj.get("solution_id", "") or ""),
                        solution_name=str(obj.get("solution_name", "") or ""),
                        definition=str(obj.get("definition", "") or ""),
                        parent_solution_id=(obj.get("parent_solution_id") if obj.get("parent_solution_id") is None else str(obj.get("parent_solution_id"))),
                        eval_result=(obj.get("eval_result") if isinstance(obj.get("eval_result"), dict) else None),
                        code=str(obj.get("code", "") or ""),
                        code_excerpt=str(obj.get("code_excerpt", "") or ""),
                    )
                    if rec.solution_id:
                        self._records[rec.solution_id] = rec
            except Exception:
                # Non-fatal: DB is an optimization aid, not correctness-critical.
                pass

    def add(
        self,
        *,
        solution: Solution,
        eval_result: Optional[EvalResult],
        code_text: str,
        parent_solution_id: Optional[str],
    ) -> SolutionRecord:
        solution_id = solution.hash()
        rec = SolutionRecord(
            solution_id=solution_id,
            solution_name=solution.name,
            definition=solution.definition,
            parent_solution_id=parent_solution_id,
            eval_result=(eval_result.__dict__ if eval_result is not None else None),
            code=(code_text or ""),
            code_excerpt=(code_text or "")[: self._max_excerpt_chars],
        )
        self._records[solution_id] = rec
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return rec

    def get(self, solution_id: str) -> Optional[SolutionRecord]:
        return self._records.get(solution_id)


