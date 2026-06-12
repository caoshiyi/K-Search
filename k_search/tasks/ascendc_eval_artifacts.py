from __future__ import annotations

import json
import os
import time
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8", errors="replace")


def active_eval_artifact_dir() -> Path | None:
    raw = os.getenv("KSEARCH_META_EVAL_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def write_command_artifacts(candidate_eval_dir: Path, *, stage: str, cmd: str | None, cwd: str | Path, proc: CompletedProcess[str] | None, duration_ms: int | None = None, timeout_seconds: int | None = None) -> dict[str, str]:
    candidate_eval_dir = Path(candidate_eval_dir)
    candidate_eval_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": None if proc is None else proc.returncode,
        "duration_ms": duration_ms,
        "timeout_seconds": timeout_seconds,
        "ts_ms": int(time.time() * 1000),
        "skipped": proc is None,
    }
    command_path = candidate_eval_dir / f"{stage}.command.json"
    command_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    stdout_path = candidate_eval_dir / f"{stage}.stdout.log"
    stderr_path = candidate_eval_dir / f"{stage}.stderr.log"
    _write(stdout_path, "" if proc is None else (proc.stdout or ""))
    _write(stderr_path, "" if proc is None else (proc.stderr or ""))
    return {"command": str(command_path), "stdout": str(stdout_path), "stderr": str(stderr_path)}


def write_eval_result_artifact(candidate_eval_dir: Path, eval_result: Any) -> str:
    if hasattr(eval_result, "to_dict"):
        payload = eval_result.to_dict(include_log_excerpt=True, max_log_chars=8000)
    elif hasattr(eval_result, "__dict__"):
        payload = dict(eval_result.__dict__)
    else:
        payload = {"value": str(eval_result)}
    path = Path(candidate_eval_dir) / "eval_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def write_combined_log(candidate_eval_dir: Path, logs: list[str]) -> str:
    path = Path(candidate_eval_dir) / "combined.log"
    _write(path, "\n\n".join(str(x) for x in logs))
    return str(path)
