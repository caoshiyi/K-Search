from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from k_search.kernel_generators.candidate_patch import CandidatePatch
from k_search.kernel_generators.project_snapshot import ProjectSnapshot
from k_search.utils.paths import get_ksearch_artifacts_dir


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def eval_result_to_dict(eval_result: Any) -> dict[str, Any]:
    if hasattr(eval_result, "to_dict") and callable(eval_result.to_dict):
        return dict(eval_result.to_dict(include_log_excerpt=True, max_log_chars=8000))
    data = _jsonable(eval_result)
    return data if isinstance(data, dict) else {"value": data}


def write_agentic_candidate_artifacts(
    *,
    artifacts_dir: str | Path | None,
    task_name: str,
    run_id: str,  # New parameter: run_id for organizing runs
    round_num: int,
    attempt_idx: int,
    prompt: str,
    transcript: str,
    changed_paths: list[str],
    diff_text: str,
    eval_result: Any,
    project_snapshot: ProjectSnapshot,
    parent_candidate_id: str | None,
    base_ref: str,
    project_rel_path: str,
    action_node_id: str | None,
    model_name: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[CandidatePatch, dict[str, str]]:
    root = get_ksearch_artifacts_dir(base_dir=artifacts_dir, task_name=task_name, run_id=run_id)
    candidate_id = f"round_{int(round_num):04d}_attempt_{int(attempt_idx):02d}"
    out_dir = root / "candidates" / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "prompt_path": out_dir / "prompt.md",
        "transcript_path": out_dir / "transcript.md",
        "changed_paths_path": out_dir / "changed_paths.txt",
        "diff_path": out_dir / "diff.patch",
        "eval_path": out_dir / "eval.json",
        "snapshot_manifest_path": out_dir / "snapshot.json",
        "manifest_path": out_dir / "manifest.json",
    }
    paths["prompt_path"].write_text(str(prompt or ""), encoding="utf-8")
    paths["transcript_path"].write_text(str(transcript or ""), encoding="utf-8")
    paths["changed_paths_path"].write_text("\n".join(changed_paths) + ("\n" if changed_paths else ""), encoding="utf-8")
    paths["diff_path"].write_text(str(diff_text or ""), encoding="utf-8")
    eval_dict = eval_result_to_dict(eval_result)
    paths["eval_path"].write_text(json.dumps(eval_dict, indent=2, sort_keys=True), encoding="utf-8")
    paths["snapshot_manifest_path"].write_text(
        json.dumps(project_snapshot.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    candidate = CandidatePatch(
        candidate_id=candidate_id,
        parent_candidate_id=parent_candidate_id,
        base_ref=str(base_ref or ""),
        project_rel_path=str(project_rel_path or "."),
        changed_paths=list(changed_paths or []),
        diff_text=str(diff_text or ""),
        prompt_path=str(paths["prompt_path"]),
        transcript_path=str(paths["transcript_path"]),
        eval_path=str(paths["eval_path"]),
        manifest_path=str(paths["manifest_path"]),
        snapshot_id=project_snapshot.snapshot_id,
        snapshot_manifest_path=str(paths["snapshot_manifest_path"]),
        round_num=int(round_num),
        action_node_id=action_node_id,
        model_name=str(model_name or ""),
        eval_result=eval_dict,
    )
    manifest = {
        **asdict(candidate),
        "diff_path": str(paths["diff_path"]),
        "changed_paths_path": str(paths["changed_paths_path"]),
        "snapshot_archive_path": project_snapshot.archive_path,
        **(metadata or {}),
    }
    paths["manifest_path"].write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return candidate, {key: str(path) for key, path in paths.items()}
