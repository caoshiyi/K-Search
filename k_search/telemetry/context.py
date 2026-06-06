from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from k_search.utils.paths import get_run_id, get_run_logs_dir, resolve_output_base, safe_path_component


@dataclass(frozen=True)
class TelemetryContext:
    run_id: str | None = None
    task_name: str | None = None
    definition: str | None = None
    flow: str | None = None
    stage: str | None = None
    round_index: int | None = None
    attempt_index: int | None = None
    action_node_id: str | None = None
    action_title: str | None = None
    model_name: str | None = None
    provider: str | None = None
    target_gpu: str | None = None
    language: str | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class TelemetryArtifacts:
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None


def is_telemetry_enabled() -> bool:
    raw = os.getenv("KSEARCH_TELEMETRY", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def telemetry_root() -> Path:
    raw = os.getenv("KSEARCH_TELEMETRY_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    # Default telemetry is rooted at the shared output base; build_attempt_dir()
    # places it under the run-local logs directory.
    return resolve_output_base()


def default_run_id() -> str:
    # Single source of truth shared with llm logs and the narrative log.
    return get_run_id()


def _round_component(value: int | None) -> str:
    if value is None:
        return "round_global"
    return f"round_{int(value):04d}"


def _attempt_component(value: int | None) -> str:
    if value is None:
        return "attempt_unknown"
    return f"attempt_{int(value):04d}"


def build_attempt_dir(context: TelemetryContext, *, root: Path | None = None) -> Path:
    task_name = safe_path_component(context.task_name or context.definition, default="__unknown__")
    run_id = safe_path_component(context.run_id or default_run_id(), default="run")
    action = "action_" + safe_path_component(context.action_node_id, default="unknown")
    if root is None and not os.getenv("KSEARCH_TELEMETRY_DIR", "").strip():
        base = get_run_logs_dir(task_name=task_name, run_id=run_id) / "telemetry"
        return (
            base
            / _round_component(context.round_index)
            / action
            / _attempt_component(context.attempt_index)
        )
    return (
        (root or telemetry_root())
        / task_name
        / run_id
        / "telemetry"
        / _round_component(context.round_index)
        / action
        / _attempt_component(context.attempt_index)
    )
