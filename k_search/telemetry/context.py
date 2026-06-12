from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from k_search.utils.paths import (
    get_attempt_dir,
    get_run_id,
    resolve_output_base,
    safe_path_component,
)


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
    task_name = safe_path_component(
        context.task_name or context.definition, default="__unknown__"
    )
    run_id = safe_path_component(context.run_id or default_run_id(), default="run")
    if root is None and not os.getenv("KSEARCH_TELEMETRY_DIR", "").strip():
        return get_attempt_dir(
            task_name=task_name,
            run_id=run_id,
            round_num=int(context.round_index or 0),
            attempt_idx=int(context.attempt_index or 0),
            action_node_id=context.action_node_id,
        )
    # Explicit telemetry roots keep a run-like attempt-centric layout under the supplied root.
    return (
        (root or telemetry_root())
        / task_name
        / run_id
        / "attempts"
        / get_attempt_dir(
            base_dir=".",
            task_name="_",
            task_id="_",
            run_id="_",
            round_num=int(context.round_index or 0),
            attempt_idx=int(context.attempt_index or 0),
            action_node_id=context.action_node_id,
        ).name
    )
