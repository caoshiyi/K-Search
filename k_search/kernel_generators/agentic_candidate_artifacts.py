from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from k_search.kernel_generators.candidate_patch import CandidatePatch
from k_search.kernel_generators.project_snapshot import ProjectSnapshot
from k_search.utils.paths import get_attempt_dir, get_ksearch_run_dir


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


def get_agentic_candidate_artifact_dir(
    *,
    artifacts_dir: str | Path | None,
    task_name: str,
    run_id: str,
    round_num: int,
    attempt_idx: int,
    action_node_id: str | None = None,
) -> Path:
    return get_attempt_dir(
        base_dir=artifacts_dir,
        task_name=task_name,
        run_id=run_id,
        round_num=round_num,
        attempt_idx=attempt_idx,
        action_node_id=action_node_id,
    )


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_snapshot_archive(
    project_snapshot: ProjectSnapshot, snapshot_dir: Path
) -> str | None:
    src_raw = project_snapshot.archive_path or project_snapshot.project_root
    if not src_raw:
        return None
    src = Path(src_raw).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        return None
    archive_base = snapshot_dir / "project"
    archive_path = snapshot_dir / "project.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(archive_base), "gztar", root_dir=src)
    return str(archive_path)


def _update_artifact_index(
    *,
    artifacts_dir: str | Path | None,
    task_name: str,
    run_id: str,
    attempt_dir: Path,
    manifest_path: Path,
    round_num: int,
    attempt_idx: int,
    action_node_id: str | None,
) -> None:
    try:
        run_dir = get_ksearch_run_dir(
            base_dir=artifacts_dir, task_name=task_name, run_id=run_id
        )
        index_path = run_dir / "artifact_index.json"
        if index_path.is_file():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
        attempts = [a for a in data.get("attempts", []) if isinstance(a, dict)]
        rel_attempt = _rel(attempt_dir, run_dir)
        attempts = [a for a in attempts if a.get("attempt_dir") != rel_attempt]
        attempts.append(
            {
                "round_num": int(round_num),
                "attempt_idx": int(attempt_idx),
                "action_node_id": action_node_id,
                "attempt_dir": rel_attempt,
                "manifest_path": _rel(manifest_path, run_dir),
            }
        )
        data.update(
            {
                "schema_version": 1,
                "summary_path": "summary.md",
                "events_path": "events.jsonl",
                "world_model_dir": "world_model",
                "checkpoints_dir": "checkpoints",
                "attempts": sorted(
                    attempts,
                    key=lambda a: (
                        a.get("round_num") or 0,
                        a.get("attempt_idx") or 0,
                        str(a.get("action_node_id") or ""),
                    ),
                ),
            }
        )
        index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass


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
    handoff_files: dict[str, str] | None = None,
    stage_prompt_records: list[dict[str, Any]] | None = None,
) -> tuple[CandidatePatch, dict[str, str]]:
    candidate_id = f"round_{int(round_num):04d}_attempt_{int(attempt_idx):02d}"
    out_dir = get_agentic_candidate_artifact_dir(
        artifacts_dir=artifacts_dir,
        task_name=task_name,
        run_id=run_id,
        round_num=round_num,
        attempt_idx=attempt_idx,
        action_node_id=action_node_id,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "prompt_path": out_dir / "prompt.md",
        "transcript_path": out_dir / "transcript.md",
        "changed_paths_path": out_dir / "changed_paths.txt",
        "diff_path": out_dir / "diff.patch",
        "eval_path": out_dir / "eval.json",
        "snapshot_manifest_path": out_dir / "snapshot" / "snapshot.json",
        "snapshot_archive_path": out_dir / "snapshot" / "project.tar.gz",
        "manifest_path": out_dir / "manifest.json",
    }
    paths["prompt_path"].write_text(str(prompt or ""), encoding="utf-8")
    paths["transcript_path"].write_text(str(transcript or ""), encoding="utf-8")
    paths["changed_paths_path"].write_text(
        "\n".join(changed_paths) + ("\n" if changed_paths else ""), encoding="utf-8"
    )
    paths["diff_path"].write_text(str(diff_text or ""), encoding="utf-8")
    eval_dict = eval_result_to_dict(eval_result)
    paths["eval_path"].write_text(
        json.dumps(eval_dict, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["snapshot_manifest_path"].parent.mkdir(parents=True, exist_ok=True)
    archive_path = _write_snapshot_archive(
        project_snapshot, paths["snapshot_manifest_path"].parent
    )
    snapshot_dict = project_snapshot.to_dict()
    if archive_path:
        snapshot_dict["archive_path"] = _rel(Path(archive_path), out_dir)
    paths["snapshot_manifest_path"].write_text(
        json.dumps(snapshot_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    handoff_paths: dict[str, str] = {}
    for name, text in sorted((handoff_files or {}).items()):
        safe_name = Path(str(name)).name
        if not safe_name:
            continue
        p = out_dir / "handoff" / safe_name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(text or ""), encoding="utf-8")
        handoff_paths[safe_name] = _rel(p, out_dir)

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
        "diff_path": _rel(paths["diff_path"], out_dir),
        "changed_paths_path": _rel(paths["changed_paths_path"], out_dir),
        "snapshot_archive_path": (
            _rel(paths["snapshot_archive_path"], out_dir)
            if paths["snapshot_archive_path"].exists()
            else None
        ),
        **(metadata or {}),
    }
    if handoff_paths:
        manifest["native_handoff_paths"] = handoff_paths
    if stage_prompt_records is not None:
        manifest["stage_prompt_paths"] = list(stage_prompt_records)
    paths["manifest_path"].write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    _update_artifact_index(
        artifacts_dir=artifacts_dir,
        task_name=task_name,
        run_id=run_id,
        attempt_dir=out_dir,
        manifest_path=paths["manifest_path"],
        round_num=round_num,
        attempt_idx=attempt_idx,
        action_node_id=action_node_id,
    )
    return candidate, {key: str(path) for key, path in paths.items()}


def write_agentic_failed_attempt_manifest(
    *,
    artifacts_dir: str | Path | None,
    task_name: str,
    run_id: str,
    round_num: int,
    attempt_idx: int,
    stage: str,
    error_type: str,
    error_message: str,
    prompt: str = "",
    model_name: str = "",
    action_node_id: str | None = None,
    parent_candidate_id: str | None = None,
    stage_prompt_records: list[dict[str, Any]] | None = None,
    telemetry_paths: dict[str, str | None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = f"round_{int(round_num):04d}_attempt_{int(attempt_idx):02d}"
    out_dir = get_agentic_candidate_artifact_dir(
        artifacts_dir=artifacts_dir,
        task_name=task_name,
        run_id=run_id,
        round_num=round_num,
        attempt_idx=attempt_idx,
        action_node_id=action_node_id,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = out_dir / "prompt.md"
    manifest_path = out_dir / "manifest.json"
    if prompt:
        prompt_path.write_text(str(prompt), encoding="utf-8")

    manifest: dict[str, Any] = {
        "candidate_id": candidate_id,
        "status": "failed",
        "run_id": run_id,
        "task_name": task_name,
        "round_num": int(round_num),
        "attempt_idx": int(attempt_idx),
        "action_node_id": action_node_id,
        "parent_candidate_id": parent_candidate_id,
        "model_name": str(model_name or ""),
        "prompt_path": str(prompt_path) if prompt else None,
        "failure": {
            "stage": str(stage or ""),
            "error_type": str(error_type or "Error"),
            "error_message": str(error_message or ""),
        },
        "stage_prompt_paths": list(stage_prompt_records or []),
        "telemetry": dict(telemetry_paths or {}),
    }
    if metadata:
        manifest.update(dict(metadata))
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    _update_artifact_index(
        artifacts_dir=artifacts_dir,
        task_name=task_name,
        run_id=run_id,
        attempt_dir=out_dir,
        manifest_path=manifest_path,
        round_num=round_num,
        attempt_idx=attempt_idx,
        action_node_id=action_node_id,
    )
    return manifest
