from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from k_search.meta_harness.contracts import FailureSignature, RunState
from k_search.meta_harness.process_log import append_jsonl, now_ms, write_process_exit, write_process_start


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class RunStateWriter:
    def __init__(self, *, run_root: Path, run_id: str, task_id: str | None, task_name: str, task_source: str = "ascendc", language: str = "ascendc", llm_provider: str = "claude-agent") -> None:
        self.run_root = Path(run_root)
        self.logs_dir = self.run_root / "logs"
        self.path = self.run_root / "run_state.json"
        self.state = RunState(run_id=run_id, task_id=task_id, task_name=task_name, task_source=task_source, language=language, llm_provider=llm_provider, pid=os.getpid(), last_heartbeat_ts_ms=now_ms())
        self._started = False

    def mark_running(self, *, stage: str | None = None, argv: list[str] | None = None) -> None:
        self.state.status = "running"
        if stage:
            self.state.last_stage = stage
        self.state.pid = os.getpid()
        self.state.last_heartbeat_ts_ms = now_ms()
        self._flush()
        if not self._started:
            self._started = True
            for name in ("stdout.log", "stderr.log", "resource.jsonl", "harness_decisions.jsonl", "harness_actions.jsonl", "logging_errors.jsonl", "failure_index.jsonl"):
                (self.logs_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (self.logs_dir / name).touch(exist_ok=True)
            write_process_start(self.logs_dir, run_id=self.state.run_id, argv=argv or sys.argv, cwd=os.getcwd())

    def update_stage(self, *, stage: str, round_index: int | None = None, attempt_index: int | None = None, action_node_id: str | None = None, candidate_id: str | None = None) -> None:
        self.state.last_stage = stage
        if round_index is not None:
            self.state.last_round_index = int(round_index)
        if attempt_index is not None:
            self.state.last_attempt_index = int(attempt_index)
        if action_node_id is not None:
            self.state.last_action_node_id = str(action_node_id)
        if candidate_id is not None:
            self.state.last_candidate_id = str(candidate_id)
        self.heartbeat(stage=stage)

    def heartbeat(self, *, stage: str | None = None, elapsed_stage_ms: int | None = None, last_output_ts_ms: int | None = None) -> None:
        ts = now_ms()
        self.state.last_heartbeat_ts_ms = ts
        if stage:
            self.state.last_stage = stage
        payload = {
            "schema_version": 1,
            "event_type": "heartbeat",
            "ts_ms": ts,
            "run_id": self.state.run_id,
            "stage": self.state.last_stage,
            "round_index": self.state.last_round_index,
            "attempt_index": self.state.last_attempt_index,
            "action_node_id": self.state.last_action_node_id,
            "candidate_id": self.state.last_candidate_id,
            "elapsed_stage_ms": elapsed_stage_ms,
            "last_output_ts_ms": last_output_ts_ms,
        }
        append_jsonl(self.logs_dir / "process.jsonl", payload)
        self._flush()

    def set_best(self, *, candidate_id: str | None = None, score: float | None = None) -> None:
        if candidate_id is not None:
            self.state.best_candidate_id = candidate_id
        if score is not None:
            self.state.best_score = float(score)
        self._flush()

    def set_latest_checkpoint(self, *, checkpoint_id: str | None, manifest_path: str | Path | None) -> None:
        self.state.latest_checkpoint_id = str(checkpoint_id) if checkpoint_id else None
        self.state.latest_checkpoint_manifest = str(manifest_path) if manifest_path else None
        self._flush()

    def update_telemetry_health(self, sink: str, status: str) -> None:
        self.state.telemetry_health[str(sink)] = str(status)
        self._flush()

    def mark_failed(self, failure: FailureSignature | dict[str, Any]) -> None:
        data = failure.to_dict() if hasattr(failure, "to_dict") else dict(failure)
        self.state.status = "failed"
        self.state.latest_failure_signature_path = data.get("failure_signature_path") or data.get("path") or self.state.latest_failure_signature_path
        self.state.failure_class = data.get("failure_class")
        self.state.recommended_harness_action = data.get("recommended_action") or data.get("recommended_harness_action") or "restart_from_latest_checkpoint"
        self.state.last_heartbeat_ts_ms = now_ms()
        self._flush()
        write_process_exit(self.logs_dir, run_id=self.state.run_id, exit_code=1, failure_class=self.state.failure_class, retryable=data.get("retryable"), recommended_harness_action=self.state.recommended_harness_action)

    def mark_crashed(self, failure: FailureSignature | dict[str, Any]) -> None:
        data = failure.to_dict() if hasattr(failure, "to_dict") else dict(failure)
        self.state.status = "crashed"
        self.state.latest_failure_signature_path = data.get("failure_signature_path") or data.get("path") or self.state.latest_failure_signature_path
        self.state.failure_class = data.get("failure_class") or "PROCESS_CRASH"
        self.state.recommended_harness_action = data.get("recommended_action") or "restart_from_latest_checkpoint"
        self.state.last_heartbeat_ts_ms = now_ms()
        self._flush()
        write_process_exit(self.logs_dir, run_id=self.state.run_id, exit_code=1, failure_class=self.state.failure_class, retryable=data.get("retryable", True), recommended_harness_action=self.state.recommended_harness_action)

    def mark_completed(self, *, best_candidate_id: str | None = None, best_score: float | None = None) -> None:
        self.state.status = "completed"
        self.state.recommended_harness_action = "continue"
        if best_candidate_id is not None:
            self.state.best_candidate_id = best_candidate_id
        if best_score is not None:
            self.state.best_score = float(best_score)
        self.state.last_heartbeat_ts_ms = now_ms()
        self._flush()
        write_process_exit(self.logs_dir, run_id=self.state.run_id, exit_code=0)

    def _flush(self) -> None:
        _write_json_atomic(self.path, self.state.to_dict())
