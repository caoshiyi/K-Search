from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from k_search.utils.paths import safe_path_component


@dataclass(frozen=True)
class StagePromptRecord:
    index: int
    stage: str
    agent: str
    path: str
    prompt_chars: int
    hygiene: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StagePromptSink:
    def __init__(self, candidate_dir: str | Path) -> None:
        self.candidate_dir = Path(candidate_dir).expanduser().resolve()
        self.stage_dir = self.candidate_dir / "stage_prompts"
        self._records: list[StagePromptRecord] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def write(self, *, index: int, stage: Any, prompt: str, hygiene: dict[str, Any]) -> StagePromptRecord:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        stage_name = str(getattr(stage, "name", "") or "stage")
        agent = str(getattr(stage, "agent", "") or stage_name)
        safe_stage = safe_path_component(stage_name, default="stage", max_len=64)
        rel_path = f"stage_prompts/stage_{int(index):02d}_{safe_stage}.md"
        target = self.candidate_dir / rel_path
        target.write_text(str(prompt or ""), encoding="utf-8")
        record = StagePromptRecord(
            index=int(index),
            stage=stage_name,
            agent=agent,
            path=rel_path,
            prompt_chars=len(str(prompt or "")),
            hygiene=dict(hygiene or {}),
        )
        self._records.append(record)
        return record
