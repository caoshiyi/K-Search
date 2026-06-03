from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


PathLike = Union[str, Path]


def safe_path_component(value: Any, *, default: str, max_len: int = 96) -> str:
    """Sanitize an arbitrary value into a single safe path component.

    Single source of truth; telemetry/context.py re-exports this.
    """
    text = str(value if value is not None else "").strip() or default
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text).strip(".")
    return (safe or default)[:max_len]


def get_run_id() -> str:
    """Unified run identifier shared by llm logs, telemetry and the narrative log.

    Priority: KSEARCH_RUN_ID > KSEARCH_RUN_START > local wall-clock timestamp.
    Local time (not UTC) so directory names match what the user sees on the clock.
    """
    for name in ("KSEARCH_RUN_ID", "KSEARCH_RUN_START"):
        raw = os.getenv(name, "").strip()
        if raw:
            return safe_path_component(raw, default="run")
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def get_ksearch_artifacts_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    run_id: Optional[str] = None,
    include_run: bool = True,
) -> Path:
    """
    Default k-search artifacts directory (independent of flashinfer-bench dataset paths).

    Artifacts (solutions, eval, candidates, snapshots, world_model, memory) live at
    `<base>/<task>`. Base resolution is shared with logs via resolve_output_base().

    When include_run=True and run_id is provided, artifacts are scoped under
    `<base>/<task>/runs/<run_id>/` for run-level isolation.
    """
    root = resolve_output_base(base_dir)
    if task_name:
        root = root / safe_path_component(task_name, default="__unknown__")
    if include_run:
        rid = run_id or get_run_id()
        root = root / "runs" / safe_path_component(rid, default="run")
    return root


def get_run_logs_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    run_id: Optional[str] = None,
    sub: Optional[str] = None,
) -> Path:
    """Unified run-scoped logs directory: `<base>/logs/<task>/<run_id>[/<sub>]`.

    `sub` is typically "llm" or "telemetry". When omitted, returns the run root
    (where summary.md / events.jsonl / run_meta.json live).
    """
    base = resolve_output_base(base_dir)
    rid = run_id or get_run_id()
    path = (
        base
        / "logs"
        / safe_path_component(task_name, default="__unknown__")
        / safe_path_component(rid, default="run")
    )
    if sub:
        path = path / safe_path_component(sub, default="sub")
    return path
