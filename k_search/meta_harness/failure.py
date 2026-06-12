from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from k_search.meta_harness.contracts import FailureSignature
from k_search.meta_harness.process_log import append_jsonl

_RECOMMENDED = {
    "CLAUDE_AUTH_ERROR": (False, "stop_require_credentials"),
    "CLAUDE_TIMEOUT": (True, "retry_same_checkpoint"),
    "CLAUDE_TOOL_PROTOCOL_ERROR": (True, "retry_codegen_with_recovery_prompt"),
    "CLAUDE_PERMISSION_ERROR": (True, "inspect_policy_or_retry"),
    "AGENTIC_CODEGEN_FAILED": (True, "retry_current_action"),
    "GENERATED_PROJECT_INVALID": (True, "retry_codegen"),
    "ASCENDC_COMPILE_FAILED": (True, "repair_candidate_compile"),
    "ASCENDC_CORRECTNESS_FAILED": (True, "repair_candidate_correctness"),
    "ASCENDC_BENCHMARK_FAILED": (True, "repair_candidate_benchmark"),
    "ASCENDC_EVAL_TIMEOUT": (True, "repair_or_reduce_search"),
    "NO_EXECUTABLE_ACTION": (False, "stop_or_refresh_strategy"),
    "CHECKPOINT_CORRUPTION": (False, "rollback_checkpoint"),
    "CHECKPOINT_SAVE_FAILED": (True, "restart_from_latest_checkpoint"),
    "PROCESS_CRASH": (True, "restart_from_latest_checkpoint"),
    "PROCESS_OOM_OR_KILLED": (True, "restart_with_lower_budget"),
}


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _norm(text: str, limit: int = 1200) -> str:
    text = re.sub(r"/[^\s:]+", "<PATH>", str(text or ""))
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _excerpt(text: str, limit: int = 8000) -> str:
    s = str(text or "")
    return s if len(s) <= limit else s[:limit] + "\n...<truncated>..."


def make_failure_signature(*, failure_class: str, message: str = "", stage: str | None = None, round_index: int | None = None, attempt_index: int | None = None, action_node_id: str | None = None, candidate_id: str | None = None, evidence_paths: list[str] | None = None, failure_subclass: str | None = None, responsible_component: str | None = None, confidence: float = 0.8) -> FailureSignature:
    retryable, action = _RECOMMENDED.get(failure_class, (True, "restart_from_latest_checkpoint"))
    normalized = _norm(message)
    log_hash = _sha(normalized or message or failure_class)
    return FailureSignature(
        signature_id=f"{failure_class}:{log_hash}",
        failure_class=failure_class,
        failure_subclass=failure_subclass,
        responsible_component=responsible_component or ("generated_candidate" if failure_class.startswith("ASCENDC_") else "k_search"),
        stage=stage,
        round_index=round_index,
        attempt_index=attempt_index,
        action_node_id=action_node_id,
        candidate_id=candidate_id,
        normalized_message=normalized,
        raw_error_excerpt=_excerpt(message),
        log_hash=log_hash,
        retryable=retryable,
        recommended_action=action,
        confidence=float(confidence),
        evidence_paths=list(evidence_paths or []),
    )


def classify_ascendc_eval_failure(eval_result: Any, stage: str | None = None, paths: list[str] | None = None, **context: Any) -> FailureSignature:
    status = str(getattr(eval_result, "status", "") or "").lower()
    metrics = getattr(eval_result, "metrics", {}) if isinstance(getattr(eval_result, "metrics", {}), dict) else {}
    fc = str(metrics.get("failure_class") or "")
    if not fc:
        if status == "compile_failed":
            fc = "ASCENDC_COMPILE_FAILED"
        elif status == "benchmark_failed":
            fc = "ASCENDC_BENCHMARK_FAILED"
        elif status == "timeout":
            fc = "ASCENDC_EVAL_TIMEOUT"
        else:
            fc = "ASCENDC_CORRECTNESS_FAILED"
    return make_failure_signature(
        failure_class=fc,
        message=str(getattr(eval_result, "log_excerpt", "") or ""),
        stage=stage or str(metrics.get("failure_stage") or status or "evaluate_solution"),
        evidence_paths=list(paths or []),
        round_index=context.get("round_index"),
        attempt_index=context.get("attempt_index"),
        action_node_id=context.get("action_node_id"),
        candidate_id=context.get("candidate_id"),
        confidence=0.86,
    )


def classify_claude_exception(exc: BaseException, trace_path: str | None = None, context: dict[str, Any] | None = None) -> FailureSignature:
    ctx = dict(context or {})
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    if "auth" in low or "api key" in low or "credential" in low:
        fc = "CLAUDE_AUTH_ERROR"
    elif "timeout" in low or "timed out" in low:
        fc = "CLAUDE_TIMEOUT"
    elif "no such tool" in low or "unknown tool" in low or "tool_use" in low:
        fc = "CLAUDE_TOOL_PROTOCOL_ERROR"
    elif "permission" in low or "denied" in low:
        fc = "CLAUDE_PERMISSION_ERROR"
    else:
        fc = "AGENTIC_CODEGEN_FAILED"
    evidence = [p for p in [trace_path, ctx.get("tool_timeline_path"), ctx.get("candidate_manifest_path")] if p]
    return make_failure_signature(failure_class=fc, message=msg, stage=ctx.get("stage"), evidence_paths=evidence, round_index=ctx.get("round_index"), attempt_index=ctx.get("attempt_index"), action_node_id=ctx.get("action_node_id"), candidate_id=ctx.get("candidate_id"), confidence=0.8)


def classify_process_exit(exit_code: int | None, signal: str | None = None, logs: str | None = None, **context: Any) -> FailureSignature:
    msg = logs or f"process exited with exit_code={exit_code} signal={signal}"
    fc = "PROCESS_OOM_OR_KILLED" if exit_code == 137 or str(signal or "") == "SIGKILL" or "oom" in msg.lower() else "PROCESS_CRASH"
    return make_failure_signature(failure_class=fc, message=msg, stage=context.get("stage"), evidence_paths=list(context.get("evidence_paths") or []), round_index=context.get("round_index"), attempt_index=context.get("attempt_index"), action_node_id=context.get("action_node_id"), candidate_id=context.get("candidate_id"), confidence=0.75)


def write_failure_artifacts(run_root: Path, failure: FailureSignature, *, candidate_dir: Path | None = None) -> dict[str, str]:
    logs_dir = Path(run_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = failure.to_dict()
    latest_path = logs_dir / "latest_failure.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_jsonl(logs_dir / "failure_index.jsonl", payload)
    out = {"latest_failure_path": str(latest_path)}
    if candidate_dir is not None:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        cand_path = candidate_dir / "failure_signature.json"
        cand_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        out["candidate_failure_signature_path"] = str(cand_path)
        eval_dir = candidate_dir / "eval"
        if eval_dir.is_dir():
            eval_path = eval_dir / "failure_signature.json"
            eval_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            out["eval_failure_signature_path"] = str(eval_path)
    return out
