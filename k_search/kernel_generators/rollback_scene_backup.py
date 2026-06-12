from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from k_search.utils.paths import safe_path_component


class SceneBackupFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class SceneBackupResult:
    scene_dir: Path
    manifest_path: Path


def backup_failure_scene(
    *,
    project_dir: str | Path,
    artifacts_dir: str | Path,
    run_id: str,
    round_num: int,
    attempt_idx: int,
    candidate_id: str | None,
    stage: str,
    agent: str,
    stage_retry_index: int,
    rollback_reason: str,
    checkpoint_manifest_path: str | Path | None,
    telemetry_recorder: Any | None = None,
    runtime_state: dict[str, Any] | None = None,
) -> SceneBackupResult:
    try:
        project_root = Path(project_dir).expanduser().resolve()
        root = Path(artifacts_dir).expanduser().resolve() / "failures"
        root.mkdir(parents=True, exist_ok=True)
        scene_dir = _unique_scene_dir(
            root
            / (
                f"round_{int(round_num):04d}_attempt_{int(attempt_idx):02d}_"
                f"{safe_path_component(stage, default='stage')}_retry_{int(stage_retry_index):02d}"
            )
        )
        raw_dir = scene_dir / "raw"
        raw_dir.mkdir(parents=True)

        project_backup = raw_dir / "project_before_rollback"
        _copy_project_tree(project_root, project_backup)

        telemetry_paths = _copy_telemetry_artifacts(
            raw_dir / "telemetry", telemetry_recorder
        )
        attempt_artifact_path = _copy_attempt_artifacts(
            raw_dir / "attempt_artifacts",
            runtime_state or {},
            forbidden_roots={scene_dir, project_root},
        )
        checkpoint_backup = _copy_checkpoint(
            raw_dir / "checkpoint", checkpoint_manifest_path
        )

        manifest = {
            "schema_version": 1,
            "run_id": str(run_id or ""),
            "round_num": int(round_num),
            "attempt_idx": int(attempt_idx),
            "candidate_id": candidate_id,
            "stage": str(stage or ""),
            "agent": str(agent or ""),
            "stage_retry_index": int(stage_retry_index),
            "rollback_reason": str(rollback_reason or ""),
            "checkpoint_id": _checkpoint_id(checkpoint_manifest_path),
            "backup_paths": {
                "project_before_rollback": "raw/project_before_rollback",
                "attempt_artifacts": (
                    "raw/attempt_artifacts" if attempt_artifact_path else None
                ),
                "telemetry": "raw/telemetry" if telemetry_paths else None,
                "checkpoint": "raw/checkpoint" if checkpoint_backup else None,
            },
        }
        manifest_path = scene_dir / "scene_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return SceneBackupResult(scene_dir=scene_dir, manifest_path=manifest_path)
    except Exception as exc:
        raise SceneBackupFailed(f"failed to backup failure scene: {exc}") from exc


def _unique_scene_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to allocate failure scene directory under {path.parent}")


def _copy_project_tree(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name == ".git"}

    shutil.copytree(src, dst, symlinks=True, ignore=ignore)


def _copy_telemetry_artifacts(dst: Path, telemetry_recorder: Any | None) -> list[str]:
    artifacts = getattr(telemetry_recorder, "artifacts", None)
    if artifacts is None:
        return []
    copied: list[str] = []
    for attr in ("trace_path", "timeline_path", "cost_path"):
        value = getattr(artifacts, attr, None)
        if not value:
            continue
        src = Path(str(value)).expanduser()
        if not src.is_file():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        target = dst / src.name
        shutil.copy2(src, target)
        copied.append(str(target))
    return copied


def _copy_attempt_artifacts(
    dst: Path,
    runtime_state: dict[str, Any],
    *,
    forbidden_roots: set[Path],
) -> Path | None:
    raw_candidates = [
        runtime_state.get("attempt_artifacts_dir"),
        runtime_state.get("attempt_artifact_dir"),
        runtime_state.get("artifact_dir"),
    ]
    for raw in raw_candidates:
        if not raw:
            continue
        src = Path(str(raw)).expanduser().resolve()
        if not src.exists() or any(_path_is_under(src, root) for root in forbidden_roots):
            continue
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
        return dst
    return None


def _copy_checkpoint(dst: Path, checkpoint_manifest_path: str | Path | None) -> Path | None:
    if not checkpoint_manifest_path:
        return None
    manifest_path = Path(checkpoint_manifest_path).expanduser()
    if manifest_path.is_dir():
        checkpoint_dir = manifest_path
    else:
        checkpoint_dir = manifest_path.parent
    if not checkpoint_dir.is_dir():
        return None
    shutil.copytree(checkpoint_dir, dst, symlinks=True)
    return dst


def _checkpoint_id(checkpoint_manifest_path: str | Path | None) -> str | None:
    if not checkpoint_manifest_path:
        return None
    path = Path(checkpoint_manifest_path).expanduser()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        return path.name or None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return manifest_path.parent.name
    return str(data.get("checkpoint_id") or manifest_path.parent.name)


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
    except Exception:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents
