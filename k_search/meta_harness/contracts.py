from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FailureClass = Literal[
    "CLAUDE_AUTH_ERROR",
    "CLAUDE_TIMEOUT",
    "CLAUDE_TOOL_PROTOCOL_ERROR",
    "CLAUDE_PERMISSION_ERROR",
    "AGENTIC_CODEGEN_FAILED",
    "GENERATED_PROJECT_INVALID",
    "ASCENDC_COMPILE_FAILED",
    "ASCENDC_CORRECTNESS_FAILED",
    "ASCENDC_BENCHMARK_FAILED",
    "ASCENDC_EVAL_TIMEOUT",
    "NO_EXECUTABLE_ACTION",
    "CHECKPOINT_CORRUPTION",
    "CHECKPOINT_SAVE_FAILED",
    "PROCESS_CRASH",
    "PROCESS_OOM_OR_KILLED",
]

RunStatus = Literal["running", "completed", "failed", "crashed"]
HarnessAction = Literal[
    "continue",
    "stop_require_credentials",
    "retry_same_checkpoint",
    "retry_codegen_with_recovery_prompt",
    "inspect_policy_or_retry",
    "retry_current_action",
    "retry_codegen",
    "repair_candidate_compile",
    "repair_candidate_correctness",
    "repair_candidate_benchmark",
    "repair_or_reduce_search",
    "stop_or_refresh_strategy",
    "rollback_checkpoint",
    "restart_from_latest_checkpoint",
    "restart_with_lower_budget",
]

@dataclass
class MetaHarnessArtifactRef:
    path: str
    kind: str = "file"
    description: str | None = None

@dataclass
class FailureSignature:
    schema_version: int = 1
    signature_id: str = ""
    failure_class: str = "PROCESS_CRASH"
    failure_subclass: str | None = None
    responsible_component: str = "k_search"
    stage: str | None = None
    round_index: int | None = None
    attempt_index: int | None = None
    action_node_id: str | None = None
    candidate_id: str | None = None
    normalized_message: str = ""
    raw_error_excerpt: str = ""
    log_hash: str = ""
    retryable: bool = True
    recommended_action: str = "restart_from_latest_checkpoint"
    confidence: float = 0.5
    evidence_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class RunState:
    schema_version: int = 1
    run_id: str = ""
    task_id: str | None = None
    task_name: str = ""
    task_source: str = "ascendc"
    language: str = "ascendc"
    llm_provider: str = "claude-agent"
    status: str = "running"
    pid: int | None = None
    last_heartbeat_ts_ms: int | None = None
    last_stage: str | None = None
    last_round_index: int | None = None
    last_attempt_index: int | None = None
    last_action_node_id: str | None = None
    last_candidate_id: str | None = None
    best_candidate_id: str | None = None
    best_score: float | None = None
    latest_checkpoint_id: str | None = None
    latest_checkpoint_manifest: str | None = None
    latest_failure_signature_path: str | None = None
    failure_class: str | None = None
    recommended_harness_action: str = "continue"
    telemetry_health: dict[str, str] = field(default_factory=lambda: {
        "narrative": "ok",
        "agent_trace": "ok",
        "candidate_artifacts": "ok",
        "checkpoint": "ok",
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class HarnessDecision:
    schema_version: int = 1
    event_type: str = "harness_decision"
    ts_ms: int | None = None
    observed_run_id: str = ""
    observed_status: str = ""
    failure_class: str | None = None
    selected_action: str = "continue"
    checkpoint_id: str | None = None
    reason: str = ""
    confidence: float = 0.0
