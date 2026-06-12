from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from k_search.kernel_generators.checkpoint_index import (
    append_or_update_checkpoint_index,
    sync_pruned_entries,
)
from k_search.kernel_generators.project_snapshot import ProjectSnapshot
from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
)
from k_search.utils.paths import get_run_checkpoints_dir, get_run_world_model_dir

CheckpointEvery = Literal["cycle", "attempt"]
ResumeMode = Literal["same-run", "new-run"]
ResumePolicy = Literal["latest", "stable"]


@dataclass
class CheckpointConfig:
    enabled: bool = False
    every: CheckpointEvery = "cycle"
    checkpoint_dir: str | Path | None = None
    keep: int = 5
    resume_from: str | None = None
    resume_mode: ResumeMode = "new-run"
    resume_policy: ResumePolicy = "latest"
    include_project_snapshot_payload: bool = True
    enable_claude_file_checkpointing: bool = False
    resume_claude_session: bool = False
    retry_failed_attempt: bool = False


@dataclass
class CheckpointRef:
    checkpoint_id: str
    checkpoint_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass
class RestoredCheckpoint:
    checkpoint_id: str
    checkpoint_version: str
    checkpoint_kind: str
    manifest: dict[str, Any]
    runtime_state: dict[str, Any]
    world_model_path: Path
    solution_db_path: Path | None
    best_solution: Solution | None
    best_eval: EvalResult | None
    best_score: float
    current_solution: Solution | None
    start_round: int
    resume_in_cycle: bool


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _eval_to_dict(eval_result: Any | None) -> dict[str, Any] | None:
    if eval_result is None:
        return None
    if hasattr(eval_result, "to_dict"):
        try:
            return eval_result.to_dict(include_log_excerpt=True)
        except TypeError:
            return eval_result.to_dict()
    if hasattr(eval_result, "__dict__"):
        return dict(eval_result.__dict__)
    return None


def _solution_id(solution: Any | None) -> str | None:
    if solution is None:
        return None
    try:
        return str(solution.hash())
    except Exception:
        return None


def _solution_payload(
    *,
    solution: Any | None,
    eval_result: Any | None,
    score: float | None = None,
) -> dict[str, Any] | None:
    if solution is None:
        return None
    if hasattr(solution, "to_dict"):
        data = solution.to_dict()
    elif hasattr(solution, "__dict__"):
        data = dict(solution.__dict__)
    else:
        data = {"value": str(solution)}
    return {
        "schema_version": 1,
        "solution_id": _solution_id(solution),
        "solution": data,
        "eval": _eval_to_dict(eval_result),
        "score": score,
    }


def _score_or_default(value: Any, default: float = -1.0) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _solution_from_payload(payload: dict[str, Any] | None) -> Solution | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("solution")
    if not isinstance(raw, dict):
        return None
    spec_raw = raw.get("spec") if isinstance(raw.get("spec"), dict) else {}
    language = str(spec_raw.get("language") or "python")
    try:
        spec_language: SupportedLanguages | str = SupportedLanguages(language)
    except Exception:
        spec_language = language
    sources_raw = raw.get("sources") if isinstance(raw.get("sources"), list) else []
    return Solution(
        name=str(raw.get("name") or "restored_solution"),
        definition=str(raw.get("definition") or ""),
        author=str(raw.get("author") or "checkpoint"),
        description=(
            raw.get("description")
            if raw.get("description") is None
            else str(raw.get("description"))
        ),
        spec=BuildSpec(
            language=spec_language,  # type: ignore[arg-type]
            target_hardware=[str(x) for x in (spec_raw.get("target_hardware") or [])],
            entry_point=str(spec_raw.get("entry_point") or ""),
            dependencies=[str(x) for x in (spec_raw.get("dependencies") or [])],
        ),
        sources=[
            SourceFile(
                path=str(item.get("path") or ""), content=str(item.get("content") or "")
            )
            for item in sources_raw
            if isinstance(item, dict)
        ],
    )


def _eval_from_payload(payload: dict[str, Any] | None) -> EvalResult | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("eval")
    if not isinstance(raw, dict):
        return None
    return EvalResult(
        status=str(raw.get("status") or ""),
        latency_ms=(
            float(raw["latency_ms"])
            if isinstance(raw.get("latency_ms"), (int, float))
            else None
        ),
        reference_latency_ms=(
            float(raw["reference_latency_ms"])
            if isinstance(raw.get("reference_latency_ms"), (int, float))
            else None
        ),
        mean_vs_baseline_factor=(
            float(raw["mean_vs_baseline_factor"])
            if isinstance(raw.get("mean_vs_baseline_factor"), (int, float))
            else None
        ),
        speedup_factor=(
            float(raw["speedup_factor"])
            if isinstance(raw.get("speedup_factor"), (int, float))
            else None
        ),
        log_excerpt=str(raw.get("log_excerpt") or ""),
        metrics=(raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}),
    )


class CheckpointManager:
    def __init__(
        self,
        *,
        artifacts_dir: str | Path | None,
        task_name: str,
        task_id: str | None,
        run_id: str,
        config: CheckpointConfig,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.task_name = str(task_name or "")
        self.task_id = str(task_id or "") if task_id else None
        self.run_id = str(run_id or "")
        self.config = config
        if config.checkpoint_dir:
            self.checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve()
        else:
            self.checkpoint_root = get_run_checkpoints_dir(
                base_dir=artifacts_dir,
                task_name=self.task_name,
                task_id=self.task_id,
                run_id=self.run_id,
            )

    def save_cycle_checkpoint(self, **kwargs: Any) -> Path:
        checkpoint_id = self._next_checkpoint_id(
            kind="cycle", round_index=int(kwargs.get("round_index") or 0)
        )
        tmp_dir = self.checkpoint_root / f".{checkpoint_id}.{uuid.uuid4().hex}.tmp"
        final_dir = self.checkpoint_root / checkpoint_id
        if final_dir.exists():
            raise FileExistsError(f"checkpoint already exists: {final_dir}")
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            manifest = self._write_cycle_payload(
                tmp_dir=tmp_dir, checkpoint_id=checkpoint_id, **kwargs
            )
            manifest_path = tmp_dir / "manifest.json"
            _write_json(manifest_path, manifest)
            os.replace(tmp_dir, final_dir)
            final_manifest_path = final_dir / "manifest.json"
            self._write_latest(checkpoint_id=checkpoint_id)
            append_or_update_checkpoint_index(self.checkpoint_root, final_manifest_path)
            self.prune_old_checkpoints()
            return final_manifest_path
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def save_attempt_checkpoint(self, **kwargs: Any) -> Path:
        checkpoint_id = self._next_checkpoint_id(
            kind="attempt", round_index=int(kwargs.get("round_index") or 0)
        )
        tmp_dir = self.checkpoint_root / f".{checkpoint_id}.{uuid.uuid4().hex}.tmp"
        final_dir = self.checkpoint_root / checkpoint_id
        if final_dir.exists():
            raise FileExistsError(f"checkpoint already exists: {final_dir}")
        tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            manifest = self._write_attempt_payload(
                tmp_dir=tmp_dir, checkpoint_id=checkpoint_id, **kwargs
            )
            manifest_path = tmp_dir / "manifest.json"
            _write_json(manifest_path, manifest)
            os.replace(tmp_dir, final_dir)
            final_manifest_path = final_dir / "manifest.json"
            self._write_latest(checkpoint_id=checkpoint_id)
            append_or_update_checkpoint_index(self.checkpoint_root, final_manifest_path)
            self.prune_old_checkpoints()
            return final_manifest_path
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def resolve(self, ref: str, *, policy: ResumePolicy = "latest") -> CheckpointRef:
        raw = str(ref or "").strip()
        if not raw or raw == "latest":
            latest_path = self.checkpoint_root / "latest.json"
            if not latest_path.is_file():
                raise FileNotFoundError(
                    f"checkpoint latest.json not found: {latest_path}"
                )
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            raw = str(latest.get("latest_checkpoint_path") or "")
            if not raw:
                raw = str(latest.get("latest_checkpoint_id") or "")

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.checkpoint_root / candidate
        if candidate.is_dir():
            manifest_path = candidate / "manifest.json"
        elif candidate.name == "manifest.json":
            manifest_path = candidate
        else:
            manifest_path = self.checkpoint_root / raw / "manifest.json"
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"checkpoint manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_id = str(manifest.get("checkpoint_id") or manifest_path.parent.name)
        return CheckpointRef(
            checkpoint_id=checkpoint_id,
            checkpoint_dir=manifest_path.parent,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    def restore_to_run(
        self, ref: CheckpointRef, *, target_run_id: str
    ) -> RestoredCheckpoint:
        manifest = dict(ref.manifest or {})
        if int(manifest.get("schema_version") or 0) != 1:
            raise ValueError("unsupported checkpoint schema_version")
        checkpoint_kind = str(manifest.get("checkpoint_kind") or "")
        if checkpoint_kind not in {"cycle_boundary", "attempt_boundary"}:
            raise ValueError(
                f"unsupported checkpoint_kind for checkpoint restore: {checkpoint_kind}"
            )
        task_meta = (
            manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
        )
        llm_meta = manifest.get("llm") if isinstance(manifest.get("llm"), dict) else {}
        if str(task_meta.get("task_source") or "").strip() != "ascendc":
            raise ValueError("checkpoint currently supports only task_source=ascendc")
        if str(llm_meta.get("provider") or "").strip() != "claude-agent":
            raise ValueError(
                "checkpoint currently supports only llm provider claude-agent"
            )
        if str(llm_meta.get("language") or "").strip() != "ascendc":
            raise ValueError("checkpoint currently supports only language=ascendc")

        paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
        runtime_state = self._read_relative_json(ref, paths.get("runtime_state"))
        target_world_model = get_run_world_model_dir(
            base_dir=self.artifacts_dir,
            task_name=self.task_name,
            task_id=self.task_id,
            run_id=target_run_id,
        )

        source_wm = self._relative_path(ref, paths.get("world_model"))
        if source_wm is None or not source_wm.is_file():
            raise FileNotFoundError("checkpoint world_model file is missing")
        world_model_path = target_world_model / "world_model.json"
        _copy_file(source_wm, world_model_path)

        solution_db_path: Path | None = None
        source_solution_db = self._relative_path(ref, paths.get("solution_db"))
        if source_solution_db is not None and source_solution_db.is_file():
            solution_db_path = target_world_model / "solution_db.jsonl"
            _copy_file(source_solution_db, solution_db_path)

        best_payload = self._read_relative_json(
            ref, paths.get("best_solution"), missing_ok=True
        )
        current_payload = self._read_relative_json(
            ref, paths.get("current_solution"), missing_ok=True
        )
        best_score = (
            manifest.get("search", {}).get("best_score")
            if isinstance(manifest.get("search"), dict)
            else None
        )
        if not isinstance(best_score, (int, float)):
            best_score = (
                best_payload.get("score") if isinstance(best_payload, dict) else -1.0
            )
        runtime_next_round = (
            runtime_state.get("next_round") if isinstance(runtime_state, dict) else None
        )
        return RestoredCheckpoint(
            checkpoint_id=ref.checkpoint_id,
            checkpoint_version=str(manifest.get("checkpoint_version") or "v1"),
            checkpoint_kind=checkpoint_kind,
            manifest=manifest,
            runtime_state=runtime_state,
            world_model_path=world_model_path,
            solution_db_path=solution_db_path,
            best_solution=_solution_from_payload(best_payload),
            best_eval=_eval_from_payload(best_payload),
            best_score=(
                float(best_score) if isinstance(best_score, (int, float)) else -1.0
            ),
            current_solution=_solution_from_payload(current_payload),
            start_round=int(runtime_next_round or 1),
            resume_in_cycle=checkpoint_kind == "attempt_boundary",
        )

    def prune_old_checkpoints(self) -> None:
        try:
            keep = int(self.config.keep)
        except Exception:
            keep = 0
        if keep <= 0 or not self.checkpoint_root.exists():
            return
        checkpoints = sorted(
            [
                p
                for p in self.checkpoint_root.iterdir()
                if p.is_dir() and p.name.startswith("ckpt_")
            ]
        )
        for path in checkpoints[:-keep]:
            shutil.rmtree(path, ignore_errors=True)

    def _write_cycle_payload(
        self, *, tmp_dir: Path, checkpoint_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        created_at = _utc_now()
        world_model_path = tmp_dir / "world_model" / "world_model.json"
        world_model_path.parent.mkdir(parents=True, exist_ok=True)
        world_model_path.write_text(
            str(kwargs.get("world_model_json") or ""), encoding="utf-8"
        )

        source_solution_db = kwargs.get("solution_db_path")
        if source_solution_db:
            _copy_file(
                Path(source_solution_db), tmp_dir / "world_model" / "solution_db.jsonl"
            )

        best_solution = kwargs.get("best_solution")
        best_eval = kwargs.get("best_eval")
        current_solution = kwargs.get("current_solution")
        current_eval = kwargs.get("current_eval")
        best_payload = _solution_payload(
            solution=best_solution,
            eval_result=best_eval,
            score=kwargs.get("best_score"),
        )
        current_payload = _solution_payload(
            solution=current_solution, eval_result=current_eval
        )
        if best_payload is not None:
            _write_json(tmp_dir / "solutions" / "best_solution.json", best_payload)
        if current_payload is not None:
            _write_json(
                tmp_dir / "solutions" / "current_solution.json", current_payload
            )
        if kwargs.get("last_solution") is not None:
            _write_json(
                tmp_dir / "solutions" / "last_solution.json",
                _solution_payload(
                    solution=kwargs.get("last_solution"),
                    eval_result=kwargs.get("last_eval"),
                ),
            )
        if kwargs.get("cycle_best_solution") is not None:
            _write_json(
                tmp_dir / "solutions" / "cycle_best_solution.json",
                _solution_payload(
                    solution=kwargs.get("cycle_best_solution"),
                    eval_result=kwargs.get("cycle_best_eval"),
                    score=kwargs.get("cycle_best_score"),
                ),
            )

        candidate_manifest_path = kwargs.get("candidate_manifest_path")
        if candidate_manifest_path:
            _copy_file(
                Path(candidate_manifest_path), tmp_dir / "candidate" / "manifest.json"
            )
        if current_eval is not None:
            _write_json(
                tmp_dir / "candidate" / "eval.json", _eval_to_dict(current_eval) or {}
            )
        diff_text = str(
            kwargs.get("candidate_diff") or kwargs.get("diff_summary") or ""
        )
        if diff_text:
            diff_path = tmp_dir / "candidate" / "diff.patch"
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text(diff_text, encoding="utf-8")

        project_snapshot = kwargs.get("project_snapshot")
        if isinstance(project_snapshot, ProjectSnapshot):
            _write_json(
                tmp_dir / "project" / "snapshot_manifest.json",
                project_snapshot.to_dict(),
            )
            if (
                self.config.include_project_snapshot_payload
                and project_snapshot.archive_path
            ):
                src = Path(project_snapshot.archive_path)
                dst = tmp_dir / "project" / "snapshot"
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)

        claude_session = kwargs.get("claude_session")
        if isinstance(claude_session, dict):
            _write_json(tmp_dir / "claude" / "session.json", claude_session)

        runtime_state = {
            "schema_version": 1,
            "checkpoint_version": "v1",
            "state_kind": "cycle_boundary",
            "resume_action": "select_next_action",
            "next_round": int(kwargs.get("next_round") or 1),
            "last_completed_action_node_id": str(kwargs.get("action_node_id") or ""),
            "last_completed_cycle_start_round": int(
                kwargs.get("cycle_start_round") or kwargs.get("round_index") or 1
            ),
            "last_completed_cycle_end_round": int(kwargs.get("round_index") or 1),
            "best_score": _score_or_default(kwargs.get("best_score")),
            "best_solution_id": _solution_id(best_solution),
            "current_solution_id": _solution_id(current_solution),
        }
        _write_json(tmp_dir / "runtime_state.json", runtime_state)

        args_payload = {
            "max_opt_rounds": kwargs.get("max_opt_rounds"),
            "wm_stagnation_window": kwargs.get("wm_stagnation_window"),
            "wm_max_difficulty": kwargs.get("wm_max_difficulty"),
        }
        _write_json(tmp_dir / "args.json", args_payload)
        env_payload = {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("KSEARCH_") and "KEY" not in key and "TOKEN" not in key
        }
        _write_json(tmp_dir / "env.json", env_payload)

        paths = {
            "world_model": "world_model/world_model.json",
            "solution_db": "world_model/solution_db.jsonl",
            "best_solution": "solutions/best_solution.json",
            "current_solution": "solutions/current_solution.json",
            "candidate_manifest": "candidate/manifest.json",
            "project_snapshot_manifest": "project/snapshot_manifest.json",
            "runtime_state": "runtime_state.json",
            "args": "args.json",
            "env": "env.json",
            "claude_session": "claude/session.json",
        }
        integrity: dict[str, str] = {}
        for path in sorted(p for p in tmp_dir.rglob("*") if p.is_file()):
            rel = str(path.relative_to(tmp_dir)).replace("\\", "/")
            if rel == "manifest.json":
                continue
            integrity[rel] = _sha256_file(path)

        task = kwargs.get("task")
        task_source = str(
            getattr(task, "task_source", "") or kwargs.get("task_source") or "ascendc"
        )
        task_path = str(getattr(task, "task_path", "") or kwargs.get("task_path") or "")
        best_eval_dict = _eval_to_dict(best_eval) or {}
        return {
            "schema_version": 1,
            "checkpoint_id": checkpoint_id,
            "checkpoint_version": "v1",
            "checkpoint_kind": "cycle_boundary",
            "created_at": created_at,
            "task": {
                "task_source": task_source,
                "task_name": self.task_name,
                "task_id": self.task_id,
                "definition": str(getattr(task, "name", "") or self.task_name),
                "task_path": task_path,
            },
            "run": {
                "run_id": self.run_id,
                "parent_run_id": kwargs.get("parent_run_id"),
                "parent_checkpoint_id": kwargs.get("parent_checkpoint_id"),
                "resume_mode": kwargs.get("resume_mode") or "new-run",
            },
            "llm": {
                "provider": str(kwargs.get("llm_provider") or ""),
                "model_name": str(kwargs.get("model_name") or ""),
                "language": str(kwargs.get("language") or ""),
                "target_gpu": str(kwargs.get("target_gpu") or ""),
            },
            "search": {
                "round_index": int(kwargs.get("round_index") or 0),
                "attempt_idx": None,
                "cycle_start_round": int(
                    kwargs.get("cycle_start_round") or kwargs.get("round_index") or 1
                ),
                "action_node_id": str(kwargs.get("action_node_id") or ""),
                "active_leaf_id": str(kwargs.get("action_node_id") or ""),
                "max_opt_rounds": kwargs.get("max_opt_rounds"),
                "wm_stagnation_window": kwargs.get("wm_stagnation_window"),
                "wm_max_difficulty": kwargs.get("wm_max_difficulty"),
                "best_score": _score_or_default(kwargs.get("best_score")),
                "best_solution_id": _solution_id(best_solution),
                "current_solution_id": _solution_id(current_solution),
            },
            "paths": paths,
            "eval": {
                "status": best_eval_dict.get("status"),
                "latency_ms": best_eval_dict.get("latency_ms"),
                "reference_latency_ms": best_eval_dict.get("reference_latency_ms"),
                "score": (
                    best_eval.score()
                    if isinstance(best_eval, EvalResult)
                    else kwargs.get("best_score")
                ),
                "score_name": (
                    best_eval.metrics.get("score_name")
                    if isinstance(best_eval, EvalResult)
                    and isinstance(best_eval.metrics, dict)
                    else None
                ),
            },
            "integrity": {"files": integrity},
        }

    def _write_attempt_payload(
        self, *, tmp_dir: Path, checkpoint_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        manifest = self._write_cycle_payload(
            tmp_dir=tmp_dir, checkpoint_id=checkpoint_id, **kwargs
        )
        attempt_idx = int(kwargs.get("attempt_idx") or 0)
        next_attempt_idx = int(kwargs.get("next_attempt_idx") or (attempt_idx + 1))
        round_index = int(kwargs.get("round_index") or 0)
        runtime_state = {
            "schema_version": 1,
            "checkpoint_version": "v2",
            "state_kind": "attempt_boundary",
            "resume_action": "continue_current_action",
            "next_round": int(kwargs.get("next_round") or (round_index + 1)),
            "next_attempt_idx": next_attempt_idx,
            "last_completed_attempt_idx": attempt_idx,
            "last_completed_round": round_index,
            "last_completed_action_node_id": str(kwargs.get("action_node_id") or ""),
            "cycle_start_round": int(
                kwargs.get("cycle_start_round") or round_index or 1
            ),
            "best_score": _score_or_default(kwargs.get("best_score")),
            "best_solution_id": _solution_id(kwargs.get("best_solution")),
            "current_solution_id": _solution_id(kwargs.get("current_solution")),
            "last_solution_id": _solution_id(kwargs.get("last_solution")),
        }
        _write_json(tmp_dir / "runtime_state.json", runtime_state)
        manifest["checkpoint_version"] = "v2"
        manifest["checkpoint_kind"] = "attempt_boundary"
        if isinstance(manifest.get("search"), dict):
            manifest["search"]["attempt_idx"] = attempt_idx
            manifest["search"]["next_attempt_idx"] = next_attempt_idx
        integrity: dict[str, str] = {}
        for path in sorted(p for p in tmp_dir.rglob("*") if p.is_file()):
            rel = str(path.relative_to(tmp_dir)).replace("\\", "/")
            if rel == "manifest.json":
                continue
            integrity[rel] = _sha256_file(path)
        manifest["integrity"] = {"files": integrity}
        return manifest

    def _next_checkpoint_id(self, *, kind: str, round_index: int) -> str:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        max_idx = 0
        for path in self.checkpoint_root.iterdir():
            if not path.is_dir() or not path.name.startswith("ckpt_"):
                continue
            parts = path.name.split("_", 2)
            if len(parts) < 2:
                continue
            try:
                max_idx = max(max_idx, int(parts[1]))
            except ValueError:
                continue
        return f"ckpt_{max_idx + 1:06d}_{kind}_r{int(round_index):04d}"

    def _write_latest(self, *, checkpoint_id: str) -> None:
        payload = {
            "schema_version": 1,
            "latest_checkpoint_id": checkpoint_id,
            "latest_checkpoint_path": f"{checkpoint_id}/manifest.json",
            "updated_at": _utc_now(),
        }
        tmp = self.checkpoint_root / f".latest.{uuid.uuid4().hex}.tmp"
        _write_json(tmp, payload)
        os.replace(tmp, self.checkpoint_root / "latest.json")

    def _relative_path(self, ref: CheckpointRef, rel: Any) -> Path | None:
        if not isinstance(rel, str) or not rel.strip():
            return None
        path = Path(rel)
        if not path.is_absolute():
            path = ref.checkpoint_dir / path
        return path

    def _read_relative_json(
        self,
        ref: CheckpointRef,
        rel: Any,
        *,
        missing_ok: bool = False,
    ) -> dict[str, Any]:
        path = self._relative_path(ref, rel)
        if path is None or not path.is_file():
            if missing_ok:
                return {}
            raise FileNotFoundError(f"checkpoint JSON file missing: {rel}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"checkpoint JSON file is not an object: {path}")
        return data
