from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from k_search.kernel_generators.checkpoint_index import (
    append_or_update_checkpoint_index,
)
from k_search.kernel_generators.project_snapshot import (
    FileMeta,
    ProjectSnapshot,
    create_project_snapshot,
    materialize_project_snapshot,
)
from k_search.utils.paths import safe_path_component

StageStatus = Literal["pending", "running", "completed", "failed", "skipped", "invalid"]
ResumeStagePolicy = Literal["next-pending"]
FileStateSource = Literal["project-snapshot"]


@dataclass(frozen=True)
class StageCheckpointConfig:
    enabled: bool = False
    resume_from: str | None = None
    save_stage_start: bool = True
    save_stage_completed: bool = True
    resume_claude_session: bool = False
    claude_session_required: bool = False
    subagent_resume: bool = False
    require_subagent_agent_id: bool = False
    enable_claude_file_checkpointing: bool = False
    session_store_kind: str = "none"
    session_store_config: str | None = None
    resume_stage_policy: ResumeStagePolicy = "next-pending"
    file_state_source: FileStateSource = "project-snapshot"


@dataclass(frozen=True)
class StageRecord:
    index: int
    name: str
    agent: str
    status: StageStatus
    required_files: list[str]
    produced_files: list[str]
    missing_files: list[str]
    stage_prompt_path: str | None = None
    stage_result_path: str | None = None
    pre_snapshot_manifest_path: str | None = None
    post_snapshot_manifest_path: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    file_checkpoint_uuid: str | None = None
    result_status: str | None = None


@dataclass(frozen=True)
class StageRuntimeState:
    flow_name: str
    flow_version: int
    round_num: int
    attempt_idx: int
    stages: list[StageRecord]
    last_completed_stage_index: int | None
    next_stage_index: int | None
    next_stage_name: str | None
    next_stage_agent: str | None


@dataclass(frozen=True)
class RestoredStageCheckpoint:
    checkpoint_id: str
    manifest: dict[str, Any]
    runtime_state: dict[str, Any]
    stage_state: dict[str, Any]
    restored_project_dir: Path
    next_stage_index: int | None
    next_stage_name: str | None
    all_stages_completed: bool
    session_id: str | None


def validate_stage_checkpoint_config(
    config: StageCheckpointConfig,
    *,
    task_source: str,
    language: str,
    llm_provider: str,
    world_model: bool,
) -> None:
    if not config.enabled:
        return
    if str(task_source or "").strip().lower() != "ascendc":
        raise ValueError("--checkpoint-v3 requires --task-source ascendc")
    if str(language or "").strip().lower() != "ascendc":
        raise ValueError("--checkpoint-v3 requires --language ascendc")
    if str(llm_provider or "").strip().lower() != "claude-agent":
        raise ValueError("--checkpoint-v3 requires --llm-provider claude-agent")
    if not bool(world_model):
        raise ValueError("--checkpoint-v3 requires --world-model")
    if config.session_store_kind != "none" and config.enable_claude_file_checkpointing:
        raise ValueError(
            "SessionStore cannot be combined with Claude file checkpointing"
        )
    if config.subagent_resume and not config.resume_claude_session:
        raise ValueError(
            "--checkpoint-subagent-resume requires --checkpoint-resume-claude-session"
        )
    if config.claude_session_required and not config.resume_claude_session:
        raise ValueError(
            "--checkpoint-claude-session-required requires --checkpoint-resume-claude-session"
        )
    if config.enable_claude_file_checkpointing and not config.resume_claude_session:
        raise ValueError(
            "--checkpoint-enable-claude-file-checkpointing requires --checkpoint-resume-claude-session"
        )
    if config.resume_stage_policy != "next-pending":
        raise ValueError(
            "--checkpoint-resume-stage-policy currently supports only next-pending"
        )
    if config.file_state_source != "project-snapshot":
        raise ValueError(
            "--checkpoint-file-state-source currently supports only project-snapshot"
        )


class StageCheckpointManager:
    def __init__(
        self,
        *,
        artifacts_dir: str | Path | None,
        task_name: str,
        task_id: str,
        run_id: str,
        config: StageCheckpointConfig,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir or ".ksearch").expanduser().resolve()
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"
        self.task_name = str(task_name or "__unknown__")
        self.task_id = str(task_id or "task")
        self.run_id = str(run_id or "run")
        self.config = config

    def save_stage_start(
        self,
        *,
        task: Any,
        project_dir: str | Path,
        flow: Any,
        stage: Any,
        stage_index: int,
        round_num: int,
        attempt_idx: int,
        prompt: str,
        session: Any | None,
        runtime_state: dict[str, Any],
    ) -> Path:
        if not self.config.enabled or not self.config.save_stage_start:
            raise RuntimeError("stage_start checkpointing is disabled")
        return self._save_stage_checkpoint(
            checkpoint_kind="stage_start",
            task=task,
            project_dir=project_dir,
            flow=flow,
            stage=stage,
            stage_index=stage_index,
            round_num=round_num,
            attempt_idx=attempt_idx,
            prompt=prompt,
            result=None,
            session=session,
            runtime_state=runtime_state,
        )

    def save_stage_completed(
        self,
        *,
        task: Any,
        project_dir: str | Path,
        flow: Any,
        stage: Any,
        stage_index: int,
        round_num: int,
        attempt_idx: int,
        prompt: str,
        result: Any,
        session: Any | None,
        telemetry_recorder: Any | None,
        runtime_state: dict[str, Any],
    ) -> Path:
        if not self.config.enabled or not self.config.save_stage_completed:
            raise RuntimeError("stage_completed checkpointing is disabled")
        return self._save_stage_checkpoint(
            checkpoint_kind="stage_completed",
            task=task,
            project_dir=project_dir,
            flow=flow,
            stage=stage,
            stage_index=stage_index,
            round_num=round_num,
            attempt_idx=attempt_idx,
            prompt=prompt,
            result=result,
            session=session,
            runtime_state=runtime_state,
        )

    def resolve(self, ref: str) -> Path:
        value = str(ref or "").strip()
        if not value:
            raise ValueError("checkpoint ref is required")
        if value == "latest":
            latest_path = self.checkpoints_dir / "latest.json"
            latest = _read_json(latest_path)
            path = self.checkpoints_dir / str(
                latest.get("latest_checkpoint_path") or ""
            )
            if not path.is_file():
                raise FileNotFoundError(f"latest checkpoint manifest not found: {path}")
            return path
        p = Path(value).expanduser()
        if p.exists():
            p = p.resolve()
            if p.is_dir():
                p = p / "manifest.json"
            if not p.is_file():
                raise FileNotFoundError(f"checkpoint manifest not found: {p}")
            return p
        p = self.checkpoints_dir / value / "manifest.json"
        if p.is_file():
            return p
        raise FileNotFoundError(f"checkpoint ref not found: {ref}")

    def restore(self, ref: str, *, target_run_id: str) -> RestoredStageCheckpoint:
        manifest_path = self.resolve(ref)
        checkpoint_root = manifest_path.parent
        manifest = _read_json(manifest_path)
        runtime_state = _read_json(
            checkpoint_root / str(manifest["paths"]["runtime_state"])
        )
        stage_state = _read_json(
            checkpoint_root / str(manifest["paths"]["stage_state"])
        )
        checkpoint_id = str(manifest.get("checkpoint_id") or checkpoint_root.name)
        snapshot_ref = _snapshot_path_for_restore(stage_state, manifest)
        if not snapshot_ref:
            raise FileNotFoundError(
                f"checkpoint has no project snapshot path: {checkpoint_id}"
            )
        snapshot_path = checkpoint_root / snapshot_ref
        snapshot = load_project_snapshot(snapshot_path)
        restore_dir = (
            self.artifacts_dir
            / "checkpoint_restores"
            / safe_path_component(target_run_id, default="resume")
            / safe_path_component(checkpoint_id, default="checkpoint")
        )
        if restore_dir.exists():
            shutil.rmtree(restore_dir)
        materialize_project_snapshot(snapshot, restore_dir)
        _restore_handoff_files(checkpoint_root=checkpoint_root, destination=restore_dir)
        next_index, next_name, all_completed = _restore_position(stage_state)
        session_id = _session_id_from_checkpoint(checkpoint_root)
        return RestoredStageCheckpoint(
            checkpoint_id=checkpoint_id,
            manifest=manifest,
            runtime_state=runtime_state,
            stage_state=stage_state,
            restored_project_dir=restore_dir,
            next_stage_index=next_index,
            next_stage_name=next_name,
            all_stages_completed=all_completed,
            session_id=session_id,
        )

    def _save_stage_checkpoint(
        self,
        *,
        checkpoint_kind: Literal["stage_start", "stage_completed"],
        task: Any,
        project_dir: str | Path,
        flow: Any,
        stage: Any,
        stage_index: int,
        round_num: int,
        attempt_idx: int,
        prompt: str,
        result: Any | None,
        session: Any | None,
        runtime_state: dict[str, Any],
    ) -> Path:
        project_root = Path(project_dir).expanduser().resolve()
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        parent_checkpoint_id = _latest_checkpoint_id(self.checkpoints_dir)
        sequence = _next_checkpoint_sequence(self.checkpoints_dir)
        suffix = "start" if checkpoint_kind == "stage_start" else "completed"
        stage_slug = safe_path_component(
            getattr(stage, "name", "stage"), default="stage"
        )
        checkpoint_id = (
            f"ckpt_{sequence:06d}_stage_"
            f"r{int(round_num):04d}_a{int(attempt_idx):02d}_{stage_slug}_{suffix}"
        )
        tmp_dir = self.checkpoints_dir / f".tmp_{checkpoint_id}_{os.getpid()}"
        final_dir = self.checkpoints_dir / checkpoint_id
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        prompt_rel = (
            Path("stage")
            / "stage_prompts"
            / f"stage_{int(stage_index):02d}_{stage_slug}.md"
        )
        _write_text(tmp_dir / prompt_rel, str(prompt or ""))
        result_rel: Path | None = None
        if result is not None:
            result_rel = (
                Path("stage")
                / "stage_results"
                / f"stage_{int(stage_index):02d}_{stage_slug}.json"
            )
            _write_json(tmp_dir / result_rel, _result_payload(result))

        snapshot_kind = (
            "pre_snapshot" if checkpoint_kind == "stage_start" else "post_snapshot"
        )
        snapshot_rel = (
            Path("project")
            / "stages"
            / f"stage_{int(stage_index):02d}"
            / snapshot_kind
            / "snapshot.json"
        )
        _write_project_snapshot(
            project_dir=project_root,
            snapshot_manifest_path=tmp_dir / snapshot_rel,
            snapshot_id=f"{checkpoint_id}_{snapshot_kind}",
            parent_snapshot_id=None,
            created_by_round=round_num,
            eval_result=None,
        )
        _write_handoff_files(root=tmp_dir, project_root=project_root, flow=flow)

        session_id = _session_id(result=result, session=session)
        agent_id = _agent_id_from_result(result)
        file_checkpoint_uuid = (
            str(getattr(result, "file_checkpoint_uuid", "") or "").strip() or None
            if result is not None
            else None
        )
        stages = _build_stage_records(
            flow=flow,
            project_root=project_root,
            current_stage=stage,
            current_stage_index=int(stage_index),
            checkpoint_kind=checkpoint_kind,
            prompt_rel=str(prompt_rel).replace("\\", "/"),
            result_rel=(str(result_rel).replace("\\", "/") if result_rel else None),
            snapshot_rel=str(snapshot_rel).replace("\\", "/"),
            session_id=session_id,
            agent_id=agent_id,
            file_checkpoint_uuid=file_checkpoint_uuid,
        )
        runtime = _runtime_payload(
            runtime_state=runtime_state,
            checkpoint_kind=checkpoint_kind,
            flow=flow,
            stage=stage,
            stage_index=int(stage_index),
            round_num=round_num,
            attempt_idx=attempt_idx,
        )
        stage_state = _stage_state_payload(
            checkpoint_kind=checkpoint_kind,
            flow=flow,
            stage=stage,
            stage_index=int(stage_index),
            round_num=round_num,
            attempt_idx=attempt_idx,
            stages=stages,
        )
        _write_json(tmp_dir / "runtime_state.json", runtime)
        _write_json(tmp_dir / "stage_state.json", stage_state)
        _write_claude_metadata(
            root=tmp_dir,
            session_id=session_id,
            result=result,
            stages=stages,
            config=self.config,
        )

        paths = {
            "runtime_state": "runtime_state.json",
            "stage_state": "stage_state.json",
            "post_stage_project_snapshot": (
                str(snapshot_rel).replace("\\", "/")
                if checkpoint_kind == "stage_completed"
                else None
            ),
            "pre_stage_project_snapshot": (
                str(snapshot_rel).replace("\\", "/")
                if checkpoint_kind == "stage_start"
                else None
            ),
            "claude_session": "claude/session.json",
            "subagents": "claude/subagents.json",
            "file_checkpoints": "claude/file_checkpoints.json",
        }
        manifest = {
            "schema_version": 1,
            "checkpoint_version": "v3",
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_id": checkpoint_id,
            "created_at": _utc_now(),
            "task": {
                "task_source": "ascendc",
                "task_name": str(getattr(task, "name", "") or self.task_name),
                "definition": str(
                    getattr(task, "definition_name", "")
                    or getattr(task, "name", "")
                    or self.task_name
                ),
                "task_path": str(getattr(task, "task_path", "") or project_root),
            },
            "run": {
                "run_id": self.run_id,
                "task_id": self.task_id,
                "parent_checkpoint_id": parent_checkpoint_id,
                "resume_mode": "new-run",
            },
            "position": stage_state["position"],
            "paths": paths,
            "capabilities": {
                "session_resume_available": bool(session_id),
                "subagent_resume_available": bool(agent_id),
                "file_rewind_available": bool(file_checkpoint_uuid),
                "session_store_available": self.config.session_store_kind != "none",
                "fresh_session_fallback_available": True,
            },
        }
        manifest["integrity"] = {"files": _hash_tree_files(tmp_dir)}
        _write_json(tmp_dir / "manifest.json", manifest)

        tmp_dir.rename(final_dir)
        _write_latest(self.checkpoints_dir, checkpoint_id=checkpoint_id)
        final_manifest_path = final_dir / "manifest.json"
        append_or_update_checkpoint_index(self.checkpoints_dir, final_manifest_path)
        return final_manifest_path


def load_project_snapshot(path: str | Path) -> ProjectSnapshot:
    snapshot_path = Path(path).expanduser().resolve()
    data = _read_json(snapshot_path)
    manifest = {
        str(rel): FileMeta(**meta)
        for rel, meta in (data.get("manifest") or {}).items()
        if isinstance(meta, dict)
    }
    archive_path = data.get("archive_path")
    if (
        isinstance(archive_path, str)
        and archive_path.strip()
        and not Path(archive_path).is_absolute()
    ):
        archive_path = str((snapshot_path.parent / archive_path).resolve())
    return ProjectSnapshot(
        snapshot_id=str(data.get("snapshot_id") or snapshot_path.parent.name),
        parent_snapshot_id=data.get("parent_snapshot_id"),
        base_commit=data.get("base_commit"),
        project_root=str(data.get("project_root") or ""),
        manifest=manifest,
        archive_path=(str(archive_path) if archive_path else None),
        diff_from_parent=data.get("diff_from_parent"),
        created_by_round=int(data.get("created_by_round") or 0),
        eval_result=(
            data.get("eval_result")
            if isinstance(data.get("eval_result"), dict)
            else None
        ),
    )


def _write_project_snapshot(
    *,
    project_dir: Path,
    snapshot_manifest_path: Path,
    snapshot_id: str,
    parent_snapshot_id: str | None,
    created_by_round: int,
    eval_result: dict[str, Any] | None,
) -> None:
    snapshot_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_dir = snapshot_manifest_path.parent / "archive"
    snapshot = create_project_snapshot(
        project_dir=project_dir,
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent_snapshot_id,
        base_commit=None,
        created_by_round=int(created_by_round),
        eval_result=eval_result,
        archive_dir=archive_dir,
    )
    snapshot = replace(snapshot, archive_path=f"archive/{snapshot_id}")
    _write_json(snapshot_manifest_path, snapshot.to_dict())


def _write_handoff_files(*, root: Path, project_root: Path, flow: Any) -> None:
    rel_paths: set[str] = set()
    for stage in tuple(getattr(flow, "stages", ()) or ()):
        for rel in tuple(getattr(stage, "required_files", ()) or ()):
            rel_s = str(rel or "").strip()
            if rel_s:
                rel_paths.add(rel_s)
    handoff_root = root / "stage" / "handoff"
    for rel in sorted(rel_paths):
        src = project_root / rel
        if not src.is_file():
            continue
        dst = handoff_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _restore_handoff_files(*, checkpoint_root: Path, destination: Path) -> None:
    handoff_root = checkpoint_root / "stage" / "handoff"
    if not handoff_root.is_dir():
        return
    for src in sorted(handoff_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(handoff_root)
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _build_stage_records(
    *,
    flow: Any,
    project_root: Path,
    current_stage: Any,
    current_stage_index: int,
    checkpoint_kind: str,
    prompt_rel: str,
    result_rel: str | None,
    snapshot_rel: str,
    session_id: str | None,
    agent_id: str | None,
    file_checkpoint_uuid: str | None,
) -> list[StageRecord]:
    records: list[StageRecord] = []
    for index, flow_stage in enumerate(
        tuple(getattr(flow, "stages", ()) or ()), start=1
    ):
        required = [
            str(path) for path in tuple(getattr(flow_stage, "required_files", ()) or ())
        ]
        produced = [path for path in required if (project_root / path).is_file()]
        missing = [path for path in required if path not in produced]
        is_current = _stage_matches(flow_stage, current_stage) and index == int(
            current_stage_index
        )
        if index < int(current_stage_index):
            status: StageStatus = "completed" if not missing else "invalid"
        elif is_current:
            status = "running" if checkpoint_kind == "stage_start" else "completed"
        else:
            status = "pending"
            produced = []
            missing = []
        records.append(
            StageRecord(
                index=index,
                name=str(getattr(flow_stage, "name", "") or ""),
                agent=str(getattr(flow_stage, "agent", "") or ""),
                status=status,
                required_files=required,
                produced_files=produced,
                missing_files=missing,
                stage_prompt_path=prompt_rel if is_current else None,
                stage_result_path=result_rel if is_current else None,
                pre_snapshot_manifest_path=(
                    snapshot_rel
                    if is_current and checkpoint_kind == "stage_start"
                    else None
                ),
                post_snapshot_manifest_path=(
                    snapshot_rel
                    if is_current and checkpoint_kind == "stage_completed"
                    else None
                ),
                session_id=session_id if is_current else None,
                agent_id=agent_id if is_current else None,
                file_checkpoint_uuid=file_checkpoint_uuid if is_current else None,
                result_status=(
                    "success"
                    if is_current and checkpoint_kind == "stage_completed"
                    else None
                ),
            )
        )
    return records


def _stage_state_payload(
    *,
    checkpoint_kind: str,
    flow: Any,
    stage: Any,
    stage_index: int,
    round_num: int,
    attempt_idx: int,
    stages: list[StageRecord],
) -> dict[str, Any]:
    next_index, next_name, next_agent = _next_stage_after(
        checkpoint_kind, flow, stage_index
    )
    last_completed = (
        stage_index
        if checkpoint_kind == "stage_completed"
        else _last_completed_before(stages, stage_index)
    )
    return {
        "schema_version": 1,
        "checkpoint_version": "v3",
        "state_kind": "stage_boundary",
        "attempt_id": f"round_{int(round_num):04d}_attempt_{int(attempt_idx):02d}",
        "flow_name": str(getattr(flow, "name", "") or ""),
        "flow_version": int(getattr(flow, "version", 1) or 1),
        "resume_policy": "next_pending",
        "position": {
            "round_num": int(round_num),
            "attempt_idx": int(attempt_idx),
            "flow_name": str(getattr(flow, "name", "") or ""),
            "flow_version": int(getattr(flow, "version", 1) or 1),
            "last_completed_stage_index": last_completed,
            "last_completed_stage_name": _record_name(stages, last_completed),
            "last_completed_stage_agent": _record_agent(stages, last_completed),
            "next_stage_index": next_index,
            "next_stage_name": next_name,
            "next_stage_agent": next_agent,
            "resume_action": (
                "rerun_current_stage"
                if checkpoint_kind == "stage_start"
                else (
                    "enter_eval_finalize"
                    if next_index is None
                    else "continue_next_pending_stage"
                )
            ),
        },
        "stages": [asdict(record) for record in stages],
    }


def _runtime_payload(
    *,
    runtime_state: dict[str, Any],
    checkpoint_kind: str,
    flow: Any,
    stage: Any,
    stage_index: int,
    round_num: int,
    attempt_idx: int,
) -> dict[str, Any]:
    payload = dict(runtime_state or {})
    payload.setdefault("schema_version", 1)
    payload["checkpoint_version"] = "v3"
    payload["state_kind"] = "stage_boundary"
    payload["resume_action"] = (
        "rerun_current_stage"
        if checkpoint_kind == "stage_start"
        else "continue_next_pending_stage"
    )
    payload.setdefault("position", {})
    if isinstance(payload["position"], dict):
        payload["position"].update(
            {"round_num": int(round_num), "attempt_idx": int(attempt_idx)}
        )
    payload.setdefault("attempt", {})
    if isinstance(payload["attempt"], dict):
        payload["attempt"].update(
            {
                "flow_name": str(getattr(flow, "name", "") or ""),
                "active_stage": str(getattr(stage, "name", "") or ""),
                "active_stage_index": int(stage_index),
            }
        )
    return payload


def _write_claude_metadata(
    *,
    root: Path,
    session_id: str | None,
    result: Any | None,
    stages: list[StageRecord],
    config: StageCheckpointConfig,
) -> None:
    claude_dir = root / "claude"
    session_payload = {
        "schema_version": 1,
        "enabled": bool(session_id),
        "session_id": session_id,
        "cwd": None,
        "resume_mode": "explicit_session_id" if session_id else "fresh_session",
        "resume_required": bool(config.claude_session_required),
        "default_restore_behavior": "fallback_to_fresh_session",
        "session_store": {
            "enabled": config.session_store_kind != "none",
            "kind": config.session_store_kind,
            "supports_list_subkeys": False,
        },
        "file_checkpointing": {
            "enabled": bool(config.enable_claude_file_checkpointing),
            "last_uuid": (
                str(getattr(result, "file_checkpoint_uuid", "") or "") or None
                if result is not None
                else None
            ),
        },
    }
    subagents: dict[str, Any] = {}
    for record in stages:
        subagents[record.agent] = {
            "agent_name": record.agent,
            "agent_id": record.agent_id,
            "stage_name": record.name,
            "stage_index": record.index,
            "status": record.status,
            "handoff_files": list(record.required_files),
            "can_resume_followup": bool(record.agent_id),
        }
    file_checkpoint_uuid = (
        str(getattr(result, "file_checkpoint_uuid", "") or "").strip() or None
        if result is not None
        else None
    )
    _write_json(claude_dir / "session.json", session_payload)
    _write_json(
        claude_dir / "subagents.json",
        {"schema_version": 1, "session_id": session_id, "subagents": subagents},
    )
    _write_json(
        claude_dir / "file_checkpoints.json",
        {
            "schema_version": 1,
            "enabled": bool(config.enable_claude_file_checkpointing),
            "session_id": session_id,
            "checkpoints": (
                [
                    {
                        "user_message_uuid": file_checkpoint_uuid,
                        "meaning": "stage_prompt",
                        "captured_at": _utc_now(),
                    }
                ]
                if file_checkpoint_uuid
                else []
            ),
        },
    )
    _write_json(
        claude_dir / "transcript_index.json",
        {"schema_version": 1, "session_id": session_id, "entries": []},
    )


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "text": str(getattr(result, "text", "") or ""),
        "transcript": str(getattr(result, "transcript", "") or ""),
        "session_id": str(getattr(result, "session_id", "") or "") or None,
        "file_checkpoint_uuid": str(getattr(result, "file_checkpoint_uuid", "") or "")
        or None,
        "user_message_uuids": list(getattr(result, "user_message_uuids", None) or []),
        "subagent_agent_ids": list(getattr(result, "subagent_agent_ids", None) or []),
        "subagent_invocations": list(
            getattr(result, "subagent_invocations", None) or []
        ),
    }


def _snapshot_path_for_restore(
    stage_state: dict[str, Any], manifest: dict[str, Any]
) -> str | None:
    stages = list(stage_state.get("stages") or [])
    running = [
        item
        for item in stages
        if isinstance(item, dict)
        and str(item.get("status") or "") in {"running", "failed"}
    ]
    if running:
        path = running[0].get("pre_snapshot_manifest_path")
        return (
            str(path)
            if path
            else manifest.get("paths", {}).get("pre_stage_project_snapshot")
        )
    completed = [
        item
        for item in stages
        if isinstance(item, dict) and str(item.get("status") or "") == "completed"
    ]
    if completed:
        completed.sort(key=lambda item: int(item.get("index") or 0))
        path = completed[-1].get("post_snapshot_manifest_path")
        return (
            str(path)
            if path
            else manifest.get("paths", {}).get("post_stage_project_snapshot")
        )
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    return paths.get("pre_stage_project_snapshot") or paths.get(
        "post_stage_project_snapshot"
    )


def _restore_position(
    stage_state: dict[str, Any],
) -> tuple[int | None, str | None, bool]:
    stages = [
        item for item in list(stage_state.get("stages") or []) if isinstance(item, dict)
    ]
    stages.sort(key=lambda item: int(item.get("index") or 0))
    for item in stages:
        status = str(item.get("status") or "pending")
        if status in {"running", "failed", "invalid", "pending"}:
            return int(item.get("index") or 0), str(item.get("name") or ""), False
    return None, None, True


def _session_id_from_checkpoint(root: Path) -> str | None:
    try:
        data = _read_json(root / "claude" / "session.json")
    except Exception:
        return None
    value = data.get("session_id")
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def _latest_checkpoint_id(checkpoints_dir: Path) -> str | None:
    latest = checkpoints_dir / "latest.json"
    if not latest.is_file():
        return None
    try:
        data = _read_json(latest)
        value = data.get("latest_checkpoint_id")
        return str(value).strip() if isinstance(value, str) and value.strip() else None
    except Exception:
        return None


def _next_checkpoint_sequence(checkpoints_dir: Path) -> int:
    max_seq = 0
    if checkpoints_dir.exists():
        for path in checkpoints_dir.iterdir():
            if not path.is_dir() or path.name.startswith(".tmp_"):
                continue
            match = re.match(r"ckpt_(\d+)_", path.name)
            if match:
                max_seq = max(max_seq, int(match.group(1)))
    return max_seq + 1


def _write_latest(checkpoints_dir: Path, *, checkpoint_id: str) -> None:
    latest = {
        "schema_version": 1,
        "latest_checkpoint_id": checkpoint_id,
        "latest_checkpoint_path": f"{checkpoint_id}/manifest.json",
        "updated_at": _utc_now(),
    }
    _write_json_atomic(checkpoints_dir / "latest.json", latest)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    _write_json(tmp, payload)
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _hash_tree_files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        out[rel] = "sha256:" + _sha256(path)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _session_id(*, result: Any | None, session: Any | None) -> str | None:
    for source in (result, session):
        value = getattr(source, "session_id", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _agent_id_from_result(result: Any | None) -> str | None:
    if result is None:
        return None
    ids = getattr(result, "subagent_agent_ids", None) or []
    if isinstance(ids, (list, tuple)) and ids:
        value = str(ids[0] or "").strip()
        if value:
            return value
    invocations = getattr(result, "subagent_invocations", None) or []
    if isinstance(invocations, (list, tuple)):
        for item in invocations:
            if isinstance(item, dict):
                value = str(item.get("agent_id") or item.get("agentId") or "").strip()
                if value:
                    return value
    return None


def _stage_matches(left: Any, right: Any) -> bool:
    return str(getattr(left, "name", "") or "") == str(
        getattr(right, "name", "") or ""
    ) and str(getattr(left, "agent", "") or "") == str(
        getattr(right, "agent", "") or ""
    )


def _next_stage_after(
    checkpoint_kind: str, flow: Any, stage_index: int
) -> tuple[int | None, str | None, str | None]:
    stages = tuple(getattr(flow, "stages", ()) or ())
    if checkpoint_kind == "stage_start":
        idx = int(stage_index)
    else:
        idx = int(stage_index) + 1
    if idx < 1 or idx > len(stages):
        return None, None, None
    stage = stages[idx - 1]
    return (
        idx,
        str(getattr(stage, "name", "") or ""),
        str(getattr(stage, "agent", "") or ""),
    )


def _last_completed_before(stages: list[StageRecord], stage_index: int) -> int | None:
    completed = [
        record.index
        for record in stages
        if record.index < int(stage_index) and record.status == "completed"
    ]
    return max(completed) if completed else None


def _record_name(stages: list[StageRecord], index: int | None) -> str | None:
    if index is None:
        return None
    for record in stages:
        if record.index == int(index):
            return record.name
    return None


def _record_agent(stages: list[StageRecord], index: int | None) -> str | None:
    if index is None:
        return None
    for record in stages:
        if record.index == int(index):
            return record.agent
    return None
