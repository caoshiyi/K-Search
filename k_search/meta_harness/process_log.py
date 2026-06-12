from __future__ import annotations

import json
import os
import signal as signal_module
import time
from pathlib import Path
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_process_start(logs_dir: Path, *, run_id: str, argv: list[str] | None = None, cwd: str | None = None) -> None:
    append_jsonl(logs_dir / "process.jsonl", {
        "schema_version": 1,
        "event_type": "process_start",
        "ts_ms": now_ms(),
        "pid": os.getpid(),
        "argv": list(argv or []),
        "cwd": cwd or os.getcwd(),
        "run_id": run_id,
    })


def write_process_exit(logs_dir: Path, *, run_id: str, exit_code: int = 0, signal: str | None = None, failure_class: str | None = None, retryable: bool | None = None, recommended_harness_action: str | None = None) -> None:
    append_jsonl(logs_dir / "process.jsonl", {
        "schema_version": 1,
        "event_type": "process_exit",
        "ts_ms": now_ms(),
        "pid": os.getpid(),
        "exit_code": int(exit_code),
        "signal": signal,
        "failure_class": failure_class,
        "retryable": retryable,
        "recommended_harness_action": recommended_harness_action,
        "run_id": run_id,
    })


def signal_name(signum: int | None) -> str | None:
    if signum is None:
        return None
    try:
        return signal_module.Signals(signum).name
    except Exception:
        return f"SIG{signum}"
