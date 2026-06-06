from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


PathLike = Union[str, Path]
_generated_task_id: Optional[str] = None


def safe_path_component(value: Any, *, default: str, max_len: int = 96) -> str:
    """Sanitize an arbitrary value into a single safe path component.

    Single source of truth; telemetry/context.py re-exports this.
    """
    text = str(value if value is not None else "").strip() or default
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text).strip(".")
    return (safe or default)[:max_len]


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_task_id() -> str:
    """Task-level identifier shared by all runs inside one task invocation.

    Priority: KSEARCH_TASK_ID > KSEARCH_TASK_START > local wall-clock timestamp.
    Local time (not UTC) so directory names match what the user sees on the clock.
    """
    global _generated_task_id
    for name in ("KSEARCH_TASK_ID", "KSEARCH_TASK_START"):
        raw = os.getenv(name, "").strip()
        if raw:
            return safe_path_component(raw, default="task")
    if _generated_task_id is None:
        _generated_task_id = safe_path_component(_timestamp_id(), default="task")
    return _generated_task_id


def get_run_id() -> str:
    """Unified run identifier shared by llm logs, telemetry and the narrative log.

    Priority: KSEARCH_RUN_ID > KSEARCH_RUN_START > local wall-clock timestamp.
    Local time (not UTC) so directory names match what the user sees on the clock.
    """
    for name in ("KSEARCH_RUN_ID", "KSEARCH_RUN_START"):
        raw = os.getenv(name, "").strip()
        if raw:
            return safe_path_component(raw, default="run")
    return _timestamp_id()


def resolve_output_base(base_dir: Optional[PathLike] = None) -> Path:
    """Resolve the single output base dir shared by logs and artifacts.

    Priority: explicit base_dir arg > env KSEARCH_ARTIFACTS_DIR > default `.ksearch`.
    """
    if base_dir:
        root = Path(base_dir)
    else:
        env = os.getenv("KSEARCH_ARTIFACTS_DIR", "").strip()
        root = Path(env) if env else (Path.cwd() / ".ksearch")
    return root.expanduser().resolve()


def get_ksearch_task_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Path:
    """Task root: `<base>/<task>/<task_id>`."""
    base = resolve_output_base(base_dir)
    task = safe_path_component(task_name, default="__unknown__")
    tid = safe_path_component(task_id or get_task_id(), default="task")
    return base / task / tid


def get_ksearch_task_artifacts_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Path:
    """Task-scoped artifacts root: `<base>/<task>/<task_id>/artifacts`."""
    return get_ksearch_task_dir(base_dir=base_dir, task_name=task_name, task_id=task_id) / "artifacts"


def get_ksearch_artifacts_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[str] = None,
    include_run: bool = True,
) -> Path:
    """
    Default k-search artifacts directory (independent of flashinfer-bench dataset paths).

    Run-scoped artifacts live under the shared run root:
    `<base>/<task>/<task_id>/runs/<run_id>/artifacts`.

    When include_run=False, returns the task-level artifacts root:
    `<base>/<task>/<task_id>/artifacts`.
    """
    if include_run:
        return get_ksearch_run_dir(base_dir=base_dir, task_name=task_name, task_id=task_id, run_id=run_id) / "artifacts"
    return get_ksearch_task_artifacts_dir(base_dir=base_dir, task_name=task_name, task_id=task_id)


def get_ksearch_run_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Shared run root: `<base>/<task>/<task_id>/runs/<run_id>`.

    Artifacts, logs, and worktrees are siblings beneath this directory.
    """
    rid = safe_path_component(run_id or get_run_id(), default="run")
    return get_ksearch_task_dir(base_dir=base_dir, task_name=task_name, task_id=task_id) / "runs" / rid


def get_run_logs_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[str] = None,
    sub: Optional[str] = None,
) -> Path:
    """Unified run-scoped logs directory: `<base>/<task>/<task_id>/runs/<run_id>/logs[/<sub>]`.

    `sub` is typically "llm" or "telemetry". When omitted, returns the logs root
    (where summary.md / events.jsonl / run_meta.json live).
    """
    path = get_ksearch_run_dir(base_dir=base_dir, task_name=task_name, task_id=task_id, run_id=run_id) / "logs"
    if sub:
        path = path / safe_path_component(sub, default="sub")
    return path


def get_ksearch_worktrees_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Run-scoped worktree parent: `<base>/<task>/<task_id>/runs/<run_id>/worktrees`."""
    return get_ksearch_run_dir(base_dir=base_dir, task_name=task_name, task_id=task_id, run_id=run_id) / "worktrees"
