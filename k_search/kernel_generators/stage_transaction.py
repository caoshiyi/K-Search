from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from k_search.kernel_generators.checkpoint_v3 import load_project_snapshot
from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.kernel_generators.project_snapshot import create_project_snapshot
from k_search.kernel_generators.rollback_scene_backup import backup_failure_scene
from k_search.kernel_generators.runtime_artifacts import is_native_runtime_path
from k_search.telemetry.events import TelemetryEvent

logger = logging.getLogger(__name__)

SUBAGENT_TOOL_NAMES = {"Agent", "Task"}
SUBAGENT_NAME_KEYS = (
    "subagent_type",
    "agent",
    "name",
    "subagent",
    "agent_name",
    "type",
)
NON_MUTATING_STAGE_AGENTS = {
    "code-reader",
    "designer",
    "reviewer",
    "improvement-assessor",
    "knowledge-curator",
}
MUTATING_STAGE_AGENTS = {"codegen", "bug-fixer"}


@dataclass
class StageContext:
    project_root: Path
    flow: Any
    stage: Any
    stage_index: int
    round_num: int
    attempt_idx: int
    stage_retry_index: int
    run_id: str
    task_name: str
    action_node_id: str | None
    candidate_id: str | None
    checkpoint_manifest_path: Path
    telemetry_recorder: Any | None
    session: Any | None
    runtime_state: dict[str, Any]
    event_start: int
    require_agent_tool_use: bool = False


@dataclass(frozen=True)
class StageViolation:
    code: str
    severity: str
    stage: str
    agent: str
    message: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class RecoveryResult:
    recovered: bool
    recovery_code: str
    source_violation_code: str
    changed_paths: list[str]
    cleanup_paths: list[str]
    evidence: dict[str, Any]


class StageAction(Enum):
    COMMIT = "commit"
    KEEP_AND_CONTINUE = "keep_and_continue"
    RECOVERED_AND_CONTINUE = "recovered_and_continue"
    ROLLBACK_AND_RETRY = "rollback_and_retry"
    ROLLBACK_AND_FAIL = "rollback_and_fail"
    FAIL_WITHOUT_ROLLBACK = "fail_without_rollback"


@dataclass(frozen=True)
class StageRunOutcome:
    result: ClaudeProjectEditResult
    session: Any | None


class StageTransactionFailed(RuntimeError):
    def __init__(self, message: str, *, violations: list[StageViolation]) -> None:
        super().__init__(message)
        self.violations = list(violations)


class DetectorRegistry:
    def __init__(self, detectors: list[Any] | None = None) -> None:
        self.detectors = detectors or [
            RequiredOutputDetector(),
            HandoffContentDetector(),
            PathEscapeDetector(),
            SubagentInvocationDetector(),
            StageScopeDetector(),
            CandidateDiffDetector(),
        ]

    def detect(self, ctx: StageContext, result: ClaudeProjectEditResult | None) -> list[StageViolation]:
        violations: list[StageViolation] = []
        for detector in self.detectors:
            violations.extend(detector.detect(ctx, result))
        return violations


class RequiredOutputDetector:
    def detect(self, ctx: StageContext, result: ClaudeProjectEditResult | None) -> list[StageViolation]:
        del result
        missing = [
            str(path)
            for path in tuple(getattr(ctx.stage, "required_files", ()) or ())
            if not (ctx.project_root / str(path)).is_file()
        ]
        if not missing:
            return []
        return [
            _violation(
                ctx,
                code="MISSING_REQUIRED_OUTPUT",
                message=(
                    f"subagent stage {getattr(ctx.stage, 'name', '')!r} did not "
                    f"produce required file(s): {', '.join(missing)}"
                ),
                evidence={"missing_files": missing},
            )
        ]


class HandoffContentDetector:
    def detect(self, ctx: StageContext, result: ClaudeProjectEditResult | None) -> list[StageViolation]:
        del result
        violations: list[StageViolation] = []
        for rel in tuple(getattr(ctx.stage, "required_files", ()) or ()):
            path = ctx.project_root / str(rel)
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            issue = _handoff_content_issue(str(rel), text)
            if issue is None:
                continue
            violations.append(
                _violation(
                    ctx,
                    code="INVALID_HANDOFF",
                    message=f"{rel} violates native handoff contract: {issue}",
                    evidence={"path": str(rel), "issue": issue},
                )
            )
        return violations


class PathEscapeDetector:
    def detect(self, ctx: StageContext, result: ClaudeProjectEditResult | None) -> list[StageViolation]:
        del result
        missing = [
            str(path)
            for path in tuple(getattr(ctx.stage, "required_files", ()) or ())
            if not (ctx.project_root / str(path)).is_file()
        ]
        if not missing:
            return []
        external = _required_file_writes_outside_project(
            project_root=ctx.project_root,
            missing_files=missing,
            telemetry_recorder=ctx.telemetry_recorder,
            event_start=ctx.event_start,
        )
        if not external:
            return []
        return [
            _violation(
                ctx,
                code="PATH_ESCAPE",
                message=(
                    "subagent wrote required file(s) outside candidate project root: "
                    + "; ".join(f"{rel} -> {path}" for rel, path in external)
                ),
                evidence={
                    "external_writes": [
                        {"required_file": rel, "source_path": str(path)}
                        for rel, path in external
                    ]
                },
            )
        ]


class SubagentInvocationDetector:
    def detect(self, ctx: StageContext, result: ClaudeProjectEditResult | None) -> list[StageViolation]:
        del result
        if not ctx.require_agent_tool_use or not bool(getattr(ctx.telemetry_recorder, "enabled", False)):
            return []
        events = _iter_stage_telemetry_events(
            telemetry_recorder=ctx.telemetry_recorder,
            event_start=ctx.event_start,
        )
        agent_calls = [
            event
            for event in events
            if _event_get(event, "event_type") == "tool_use"
            and _event_get(event, "tool_name") in SUBAGENT_TOOL_NAMES
        ]
        observed_calls = [
            {
                "tool_name": _event_get(event, "tool_name"),
                "subagent": _subagent_name_from_tool_input(_event_get(event, "tool_input")),
                "tool_input": _event_get(event, "tool_input"),
            }
            for event in agent_calls
        ]
        matching = [
            item
            for item in observed_calls
            if _subagent_name_matches(item.get("subagent"), str(getattr(ctx.stage, "agent", "")))
        ]
        nonmatching = [
            item
            for item in observed_calls
            if not _subagent_name_matches(item.get("subagent"), str(getattr(ctx.stage, "agent", "")))
        ]
        if matching and not nonmatching:
            return []
        return [
            _violation(
                ctx,
                code="SUBAGENT_INVOCATION_VIOLATION",
                message=(
                    "subagent stage invocation validation failed: "
                    f"stage={getattr(ctx.stage, 'name', '')!r}, "
                    f"expected_agent={getattr(ctx.stage, 'agent', '')!r}, "
                    f"matching_calls={len(matching)}, total_subagent_calls={len(agent_calls)}"
                ),
                evidence={"observed_calls": observed_calls},
            )
        ]


class StageScopeDetector:
    def detect(self, ctx: StageContext, result: ClaudeProjectEditResult | None) -> list[StageViolation]:
        del result
        agent = str(getattr(ctx.stage, "agent", "") or "").strip()
        stage_name = str(getattr(ctx.stage, "name", "") or "").strip()
        if agent not in NON_MUTATING_STAGE_AGENTS and stage_name not in NON_MUTATING_STAGE_AGENTS:
            return []
        try:
            before = _load_pre_stage_snapshot(ctx.checkpoint_manifest_path)
            after = create_project_snapshot(
                project_dir=ctx.project_root,
                snapshot_id="stage_scope_after",
                parent_snapshot_id=None,
                base_commit=None,
                created_by_round=int(ctx.round_num),
                eval_result=None,
            )
        except Exception:
            logger.debug("failed to build stage scope snapshot", exc_info=True)
            return []
        changed = [
            path
            for path in _changed_snapshot_paths(before.manifest, after.manifest)
            if not is_native_runtime_path(path)
        ]
        if not changed:
            return []
        return [
            _violation(
                ctx,
                code="SCOPE_VIOLATION",
                message=(
                    f"non-mutating stage {stage_name or agent!r} changed candidate source paths: "
                    + ", ".join(changed)
                ),
                evidence={"changed_paths": changed},
            )
        ]


class CandidateDiffDetector:
    def detect(self, ctx: StageContext, result: ClaudeProjectEditResult | None) -> list[StageViolation]:
        del result
        agent = str(getattr(ctx.stage, "agent", "") or "").strip()
        stage_name = str(getattr(ctx.stage, "name", "") or "").strip()
        if agent not in MUTATING_STAGE_AGENTS and stage_name not in MUTATING_STAGE_AGENTS:
            return []
        try:
            before = _load_pre_stage_snapshot(ctx.checkpoint_manifest_path)
            after = create_project_snapshot(
                project_dir=ctx.project_root,
                snapshot_id="candidate_diff_after",
                parent_snapshot_id=None,
                base_commit=None,
                created_by_round=int(ctx.round_num),
                eval_result=None,
            )
        except Exception:
            logger.debug("failed to build candidate diff snapshot", exc_info=True)
            return []
        changed = [
            path
            for path in _changed_snapshot_paths(before.manifest, after.manifest)
            if not is_native_runtime_path(path)
        ]
        if changed:
            return []
        return [
            _violation(
                ctx,
                code="DIFF_POLICY_VIOLATION",
                message=(
                    f"implementation stage {stage_name or agent!r} did not change "
                    "any candidate source file"
                ),
                evidence={"changed_paths": []},
            )
        ]


class RecoveryRegistry:
    def __init__(self, handlers: list[Any] | None = None) -> None:
        self.handlers = handlers or [HandoffPathRecoveryHandler()]

    def try_recover(
        self, ctx: StageContext, violations: list[StageViolation]
    ) -> list[RecoveryResult]:
        results: list[RecoveryResult] = []
        for handler in self.handlers:
            results.extend(handler.try_recover(ctx, violations))
        return results


class HandoffPathRecoveryHandler:
    def try_recover(
        self, ctx: StageContext, violations: list[StageViolation]
    ) -> list[RecoveryResult]:
        missing: list[str] = []
        for violation in violations:
            if violation.code == "MISSING_REQUIRED_OUTPUT":
                missing.extend(str(path) for path in violation.evidence.get("missing_files", []))
        missing = sorted(set(path for path in missing if path))
        if not missing:
            return []
        recovered: list[RecoveryResult] = []
        external = _required_file_writes_outside_project(
            project_root=ctx.project_root,
            missing_files=missing,
            telemetry_recorder=ctx.telemetry_recorder,
            event_start=ctx.event_start,
        )
        by_rel: dict[str, list[Path]] = {}
        for rel, source in external:
            by_rel.setdefault(rel, []).append(source)
        for rel in missing:
            candidates = [path for path in by_rel.get(rel, []) if path.is_file()]
            if len(candidates) != 1:
                continue
            source = candidates[0]
            text = source.read_text(encoding="utf-8", errors="replace")
            issue = _handoff_content_issue(rel, text)
            if issue is not None:
                continue
            target = ctx.project_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            _emit_recovery_event(
                telemetry_recorder=ctx.telemetry_recorder,
                stage=ctx.stage,
                rel=rel,
                source=source,
                target=target,
            )
            recovered.append(
                RecoveryResult(
                    recovered=True,
                    recovery_code="HANDOFF_PATH_RECOVERED",
                    source_violation_code="MISSING_REQUIRED_OUTPUT",
                    changed_paths=[rel],
                    cleanup_paths=[],
                    evidence={"source_path": str(source), "target_path": str(target)},
                )
            )
        return recovered


class RollbackPolicy:
    def decide(
        self,
        ctx: StageContext,
        violations: list[StageViolation],
        recovery_results: list[RecoveryResult],
    ) -> StageAction:
        del ctx
        if not violations:
            if any(result.recovered for result in recovery_results):
                return StageAction.RECOVERED_AND_CONTINUE
            return StageAction.COMMIT
        codes = {violation.code for violation in violations}
        if "SCENE_BACKUP_FAILED" in codes or "ROLLBACK_FAILED" in codes:
            return StageAction.FAIL_WITHOUT_ROLLBACK
        if "PATH_ESCAPE" in codes or "SECRET_LEAK" in codes or "GIT_STATE_CORRUPTED" in codes:
            return StageAction.ROLLBACK_AND_FAIL
        return StageAction.ROLLBACK_AND_RETRY


class RetryBudgetPolicy:
    DEFAULT_RETRY_BUDGET = {
        "PROVIDER_TRANSIENT_NO_EDIT": 3,
        "MISSING_REQUIRED_OUTPUT": 1,
        "INVALID_HANDOFF": 1,
        "TOOL_PROTOCOL_ERROR": 1,
        "SUBAGENT_TIMEOUT": 1,
        "MAX_TURNS": 1,
        "EMPTY_MODEL_RESULT": 1,
        "SOURCE_POLLUTION": 1,
        "SCOPE_VIOLATION": 1,
        "DIFF_POLICY_VIOLATION": 1,
        "PATH_ESCAPE": 0,
        "SECRET_LEAK": 0,
        "GIT_STATE_CORRUPTED": 0,
        "ROLLBACK_FAILED": 0,
        "SCENE_BACKUP_FAILED": 0,
    }

    def max_retries(self, violation_code: str) -> int:
        return int(self.DEFAULT_RETRY_BUDGET.get(str(violation_code), 0))

    def can_retry(self, violations: list[StageViolation], *, stage_retry_index: int) -> bool:
        code = _primary_violation_code(violations)
        return int(stage_retry_index) < self.max_retries(code)


class StageTransactionExecutor:
    def __init__(
        self,
        *,
        editor_client: Any,
        project_dir: str | Path,
        telemetry_recorder: Any | None,
        stage_checkpoint_manager: Any,
        checkpoint_task: Any | None,
        round_num: int,
        attempt_idx: int,
        runtime_state: dict[str, Any] | None,
    ) -> None:
        self.editor_client = editor_client
        self.project_root = Path(project_dir).expanduser().resolve()
        self.telemetry_recorder = telemetry_recorder
        self.stage_checkpoint_manager = stage_checkpoint_manager
        self.checkpoint_task = checkpoint_task
        self.round_num = int(round_num)
        self.attempt_idx = int(attempt_idx)
        self.runtime_state = dict(runtime_state or {})
        self.detectors = DetectorRegistry()
        self.recovery = RecoveryRegistry()
        self.rollback_policy = RollbackPolicy()
        self.retry_budget = RetryBudgetPolicy()

    def run_stage(
        self,
        *,
        session: Any,
        flow: Any,
        stage: Any,
        stage_index: int,
        prompt: str,
    ) -> StageRunOutcome:
        stage_retry_index = 0
        rollback_reason: str | None = None
        while True:
            checkpoint_manifest_path = self.stage_checkpoint_manager.save_stage_start(
                task=self.checkpoint_task,
                project_dir=self.project_root,
                flow=flow,
                stage=stage,
                stage_index=stage_index,
                round_num=self.round_num,
                attempt_idx=self.attempt_idx,
                prompt=prompt,
                session=session,
                runtime_state=self._runtime_state_for_stage(
                    stage_retry_index=stage_retry_index,
                    rollback_reason=rollback_reason,
                ),
            )
            event_start = _event_count(self.telemetry_recorder)
            ctx = self._context(
                flow=flow,
                stage=stage,
                stage_index=stage_index,
                stage_retry_index=stage_retry_index,
                checkpoint_manifest_path=Path(checkpoint_manifest_path),
                session=session,
                event_start=event_start,
            )
            result: ClaudeProjectEditResult | None = None
            recovery_results: list[RecoveryResult] = []
            try:
                result = self.editor_client.send_prompt(
                    session,
                    prompt=prompt,
                    telemetry_recorder=self.telemetry_recorder,
                )
                violations = self.detectors.detect(ctx, result)
                recovery_results = self.recovery.try_recover(ctx, violations)
                if recovery_results:
                    violations = self.detectors.detect(ctx, result)
                action = self.rollback_policy.decide(ctx, violations, recovery_results)
            except Exception as exc:
                violations = [_classify_exception(ctx, exc)]
                action = self.rollback_policy.decide(ctx, violations, recovery_results)

            if action in {StageAction.COMMIT, StageAction.RECOVERED_AND_CONTINUE, StageAction.KEEP_AND_CONTINUE}:
                assert result is not None
                self.stage_checkpoint_manager.save_stage_completed(
                    task=self.checkpoint_task,
                    project_dir=self.project_root,
                    flow=flow,
                    stage=stage,
                    stage_index=stage_index,
                    round_num=self.round_num,
                    attempt_idx=self.attempt_idx,
                    prompt=prompt,
                    result=result,
                    session=session,
                    telemetry_recorder=self.telemetry_recorder,
                    runtime_state=self._runtime_state_for_stage(
                        stage_retry_index=stage_retry_index,
                        rollback_reason=rollback_reason,
                    ),
                )
                result.session = session
                return StageRunOutcome(result=result, session=session)

            primary_code = _primary_violation_code(violations)
            rollback_reason = primary_code
            should_retry = (
                action == StageAction.ROLLBACK_AND_RETRY
                and self.retry_budget.can_retry(
                    violations,
                    stage_retry_index=stage_retry_index,
                )
            )
            try:
                self._backup_and_restore(
                    ctx=ctx,
                    violations=violations,
                    rollback_reason=rollback_reason,
                )
            except Exception:
                self._close_session(session)
                raise
            if not should_retry:
                self._close_session(session)
                raise StageTransactionFailed(
                    _failure_message(stage, violations, retry_exhausted=(action == StageAction.ROLLBACK_AND_RETRY)),
                    violations=violations,
                )
            self._close_session(session)
            session = self._open_fresh_session()
            stage_retry_index += 1

    def _context(
        self,
        *,
        flow: Any,
        stage: Any,
        stage_index: int,
        stage_retry_index: int,
        checkpoint_manifest_path: Path,
        session: Any,
        event_start: int,
    ) -> StageContext:
        action = self.runtime_state.get("action") if isinstance(self.runtime_state.get("action"), dict) else {}
        attempt = self.runtime_state.get("attempt") if isinstance(self.runtime_state.get("attempt"), dict) else {}
        return StageContext(
            project_root=self.project_root,
            flow=flow,
            stage=stage,
            stage_index=int(stage_index),
            round_num=self.round_num,
            attempt_idx=self.attempt_idx,
            stage_retry_index=int(stage_retry_index),
            run_id=str(getattr(self.stage_checkpoint_manager, "run_id", "") or ""),
            task_name=str(getattr(self.stage_checkpoint_manager, "task_name", "") or ""),
            action_node_id=_optional_str(action.get("action_node_id")),
            candidate_id=_optional_str(
                attempt.get("candidate_id")
                or action.get("candidate_id")
                or action.get("parent_candidate_id")
            ),
            checkpoint_manifest_path=checkpoint_manifest_path,
            telemetry_recorder=self.telemetry_recorder,
            session=session,
            runtime_state=self._runtime_state_for_stage(
                stage_retry_index=stage_retry_index,
                rollback_reason=None,
            ),
            event_start=int(event_start),
            require_agent_tool_use=bool(getattr(self.editor_client, "require_agent_tool_use", False)),
        )

    def _runtime_state_for_stage(
        self, *, stage_retry_index: int, rollback_reason: str | None
    ) -> dict[str, Any]:
        state = dict(self.runtime_state)
        rollback = dict(state.get("rollback") or {})
        rollback["stage_retry_index"] = int(stage_retry_index)
        rollback["rollback_count"] = int(stage_retry_index)
        if rollback_reason:
            rollback["rollback_reason"] = rollback_reason
        state["rollback"] = rollback
        return state

    def _backup_and_restore(
        self,
        *,
        ctx: StageContext,
        violations: list[StageViolation],
        rollback_reason: str,
    ) -> None:
        backup_failure_scene(
            project_dir=self.project_root,
            artifacts_dir=getattr(self.stage_checkpoint_manager, "artifacts_dir"),
            run_id=ctx.run_id,
            round_num=ctx.round_num,
            attempt_idx=ctx.attempt_idx,
            candidate_id=ctx.candidate_id,
            stage=str(getattr(ctx.stage, "name", "") or ""),
            agent=str(getattr(ctx.stage, "agent", "") or ""),
            stage_retry_index=ctx.stage_retry_index,
            rollback_reason=rollback_reason,
            checkpoint_manifest_path=ctx.checkpoint_manifest_path,
            telemetry_recorder=self.telemetry_recorder,
            runtime_state=ctx.runtime_state,
        )
        try:
            self.stage_checkpoint_manager.restore_stage_start_to_project(
                ctx.checkpoint_manifest_path,
                self.project_root,
            )
        except Exception as exc:
            rollback_failed = _violation(
                ctx,
                code="ROLLBACK_FAILED",
                message=f"failed to restore stage_start checkpoint: {exc}",
                evidence={"source_violations": [violation.code for violation in violations]},
            )
            raise StageTransactionFailed(str(rollback_failed.message), violations=[rollback_failed]) from exc

    def _open_fresh_session(self) -> Any:
        try:
            return self.editor_client.open_session(
                project_dir=self.project_root,
                telemetry_recorder=self.telemetry_recorder,
            )
        except TypeError:
            return self.editor_client.open_session(project_dir=self.project_root)

    def _close_session(self, session: Any) -> None:
        close = getattr(self.editor_client, "close_session", None)
        if not callable(close) or session is None:
            return
        if bool(getattr(session, "_closed", False)):
            return
        close(session)


def _violation(
    ctx: StageContext,
    *,
    code: str,
    message: str,
    evidence: dict[str, Any],
    severity: str = "error",
) -> StageViolation:
    return StageViolation(
        code=code,
        severity=severity,
        stage=str(getattr(ctx.stage, "name", "") or ""),
        agent=str(getattr(ctx.stage, "agent", "") or ""),
        message=message,
        evidence=evidence,
    )


def _classify_exception(ctx: StageContext, exc: Exception) -> StageViolation:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in text:
        code = "SUBAGENT_TIMEOUT"
    elif "max_turn" in text or "max turns" in text:
        code = "MAX_TURNS"
    elif "empty" in text and "result" in text:
        code = "EMPTY_MODEL_RESULT"
    else:
        code = "TOOL_PROTOCOL_ERROR"
    return _violation(
        ctx,
        code=code,
        message=str(exc),
        evidence={"exception_type": type(exc).__name__},
    )


def _failure_message(
    stage: Any, violations: list[StageViolation], *, retry_exhausted: bool
) -> str:
    prefix = (
        f"subagent stage {getattr(stage, 'name', '')!r} failed after rollback retry budget was exhausted"
        if retry_exhausted
        else f"subagent stage {getattr(stage, 'name', '')!r} failed and was rolled back"
    )
    details = "; ".join(f"{violation.code}: {violation.message}" for violation in violations)
    return f"{prefix}: {details}"


def _primary_violation_code(violations: list[StageViolation]) -> str:
    priority = (
        "SCENE_BACKUP_FAILED",
        "ROLLBACK_FAILED",
        "PATH_ESCAPE",
        "SECRET_LEAK",
        "GIT_STATE_CORRUPTED",
        "SUBAGENT_TIMEOUT",
        "MAX_TURNS",
        "TOOL_PROTOCOL_ERROR",
        "EMPTY_MODEL_RESULT",
        "SUBAGENT_INVOCATION_VIOLATION",
        "SCOPE_VIOLATION",
        "DIFF_POLICY_VIOLATION",
        "INVALID_HANDOFF",
        "MISSING_REQUIRED_OUTPUT",
    )
    codes = {str(violation.code) for violation in violations}
    for code in priority:
        if code in codes:
            return code
    return str(violations[0].code if violations else "UNKNOWN")


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _event_count(telemetry_recorder: Any | None) -> int:
    return len(getattr(telemetry_recorder, "events", []) or [])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pre_stage_snapshot(checkpoint_manifest_path: Path) -> Any:
    manifest = _read_json(checkpoint_manifest_path)
    snapshot_ref = str(
        (manifest.get("paths") or {}).get("pre_stage_project_snapshot") or ""
    ).strip()
    if not snapshot_ref:
        raise FileNotFoundError(
            f"checkpoint has no pre_stage_project_snapshot: {checkpoint_manifest_path}"
        )
    return load_project_snapshot(checkpoint_manifest_path.parent / snapshot_ref)


def _changed_snapshot_paths(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    paths = set(before) | set(after)
    changed: list[str] = []
    for path in sorted(paths):
        lhs = before.get(path)
        rhs = after.get(path)
        if lhs is None or rhs is None:
            changed.append(path)
            continue
        if getattr(lhs, "sha256", None) != getattr(rhs, "sha256", None):
            changed.append(path)
            continue
        if getattr(lhs, "mode", None) != getattr(rhs, "mode", None):
            changed.append(path)
    return changed


def _handoff_content_issue(rel: str, text: str) -> str | None:
    stripped = str(text or "").strip()
    if not stripped:
        return "empty file"
    name = Path(rel).name
    if name == "REVIEW_NOTES.md":
        status = (_field_value(stripped, "status") or "").strip().lower()
        eval_ready = (_field_value(stripped, "eval_ready") or "").strip().lower()
        if not status:
            return "missing status"
        if eval_ready not in {"true", "false"}:
            return "missing or invalid eval_ready"
        if eval_ready == "true" and status not in {"ok", "fixed"}:
            return "eval_ready true requires status ok or fixed"
    if name == "IMPROVEMENT_ASSESSMENT.md":
        status = (_field_value(stripped, "status") or "").strip().lower().replace("-", "_")
        if status not in {"improve", "no_op", "needs_design_update", "blocked"}:
            return "missing or unsupported improvement assessment status"
    return None


def _field_value(text: str, field: str) -> str | None:
    lines = str(text or "").splitlines()
    target = field.strip().lower()
    for index, line in enumerate(lines):
        stripped = line.strip().strip("#").strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(target + ":"):
            return stripped.split(":", 1)[1].strip().strip("\"'")
        if lower == target and index + 1 < len(lines):
            return lines[index + 1].strip().strip("\"'")
    return None


def _required_file_writes_outside_project(
    *,
    project_root: Path,
    missing_files: list[str],
    telemetry_recorder: Any | None,
    event_start: int,
) -> list[tuple[str, Path]]:
    root = project_root.expanduser().resolve(strict=False)
    missing_by_name = {Path(rel).name: rel for rel in missing_files}
    out: list[tuple[str, Path]] = []
    seen: set[tuple[str, str]] = set()
    for event in _iter_stage_telemetry_events(
        telemetry_recorder=telemetry_recorder,
        event_start=event_start,
    ):
        if _event_get(event, "event_type") != "tool_use" or _event_get(event, "tool_name") not in {"Write", "Edit"}:
            continue
        tool_input = _event_get(event, "tool_input")
        if not isinstance(tool_input, dict):
            continue
        raw_path = tool_input.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        observed = _observed_tool_path(root, raw_path)
        rel = missing_by_name.get(observed.name)
        if rel is None or _path_is_under_root(observed, root):
            continue
        key = (rel, str(observed))
        if key in seen:
            continue
        seen.add(key)
        out.append((rel, observed))
    return out


def _iter_stage_telemetry_events(*, telemetry_recorder: Any | None, event_start: int) -> list[Any]:
    if telemetry_recorder is None:
        return []
    events = list(getattr(telemetry_recorder, "events", []) or [])
    if events:
        return events[int(event_start) :]

    trace_path = getattr(getattr(telemetry_recorder, "artifacts", None), "trace_path", None)
    if not trace_path:
        return []
    path = Path(str(trace_path))
    if not path.is_file():
        return []
    parsed: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                parsed.append(item)
    except Exception:
        return []
    return parsed[int(event_start) :]


def _event_get(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _observed_tool_path(root: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip()).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _path_is_under_root(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _subagent_name_from_tool_input(tool_input: Any) -> str | None:
    def norm(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        if text.startswith("@agent-"):
            text = text[len("@agent-") :]
        if text.endswith(" (agent)"):
            text = text[: -len(" (agent)")]
        return text.strip() or None

    if not isinstance(tool_input, dict):
        return None
    for key in SUBAGENT_NAME_KEYS:
        value = norm(tool_input.get(key))
        if value:
            return value
    for nested_key in ("input", "arguments", "params"):
        nested = tool_input.get(nested_key)
        if isinstance(nested, dict):
            for key in SUBAGENT_NAME_KEYS:
                value = norm(nested.get(key))
                if value:
                    return value
    return None


def _subagent_name_matches(observed: str | None, expected: str) -> bool:
    if not observed:
        return False
    obs = str(observed).strip()
    exp = str(expected).strip()
    return obs == exp or obs.endswith(":" + exp) or obs.endswith("/" + exp)


def _emit_recovery_event(
    *,
    telemetry_recorder: Any | None,
    stage: Any,
    rel: str,
    source: Path,
    target: Path,
) -> None:
    emit = getattr(telemetry_recorder, "emit", None)
    if not callable(emit):
        return
    try:
        emit(
            TelemetryEvent(
                event_type="subagent_handoff_recovered",
                context={
                    "stage": str(getattr(stage, "name", "") or ""),
                    "agent": str(getattr(stage, "agent", "") or ""),
                    "required_file": rel,
                    "source_path": str(source),
                    "target_path": str(target),
                },
            )
        )
    except Exception:
        logger.debug("failed to emit subagent handoff recovery telemetry", exc_info=True)
