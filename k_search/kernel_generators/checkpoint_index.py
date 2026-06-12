from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _nested(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _eval_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    search = _nested(manifest, "search")
    current = _nested(manifest, "current_solution") or _nested(
        manifest, "last_solution"
    )
    eval_data = _nested(current, "eval")
    best = _nested(manifest, "best_solution")
    best_eval = _nested(best, "eval")
    score = (
        current.get("score")
        if current.get("score") is not None
        else search.get("score")
    )
    best_score = (
        best.get("score") if best.get("score") is not None else search.get("best_score")
    )
    return {
        "eval_status": eval_data.get("status") or best_eval.get("status"),
        "latency_ms": eval_data.get("latency_ms") or best_eval.get("latency_ms"),
        "score": score,
        "best_score": best_score,
        "best_solution_id": best.get("solution_id") or search.get("best_solution_id"),
        "current_solution_id": current.get("solution_id")
        or search.get("current_solution_id"),
    }


def entry_from_manifest(
    checkpoint_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(checkpoint_root).expanduser().resolve()
    mpath = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(mpath)
    checkpoint_id = str(manifest.get("checkpoint_id") or mpath.parent.name)
    position = _nested(manifest, "position")
    search = _nested(manifest, "search")
    paths = _nested(manifest, "paths")
    capabilities = _nested(manifest, "capabilities")
    runtime_state = _read_json(
        mpath.parent / str(paths.get("runtime_state") or "runtime_state.json")
    )
    stage_state = _read_json(
        mpath.parent / str(paths.get("stage_state") or "stage_state.json")
    )
    runtime_pos = _nested(runtime_state, "position")
    if not position:
        position = runtime_pos
    if not position and isinstance(stage_state.get("position"), dict):
        position = stage_state["position"]
    eval_summary = _eval_summary(manifest)
    rel_manifest = (
        str(mpath.relative_to(root)).replace("\\", "/")
        if mpath.is_relative_to(root)
        else str(mpath)
    )
    entry = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_version": str(manifest.get("checkpoint_version") or "v1"),
        "checkpoint_kind": str(manifest.get("checkpoint_kind") or ""),
        "created_at": manifest.get("created_at"),
        "manifest_path": rel_manifest,
        "round": position.get("round")
        or position.get("round_num")
        or search.get("round_index"),
        "attempt": position.get("attempt")
        or position.get("attempt_idx")
        or search.get("attempt_idx"),
        "action_node_id": position.get("action_node_id")
        or search.get("action_node_id"),
        "flow_name": position.get("flow_name") or position.get("flow"),
        "stage_index": position.get("stage_index"),
        "stage_name": position.get("stage_name"),
        "stage_agent": position.get("stage_agent"),
        "resume_action": runtime_state.get("resume_action")
        or position.get("resume_action"),
        "has_world_model": bool(
            paths.get("world_model") or (mpath.parent / "world_model").exists()
        ),
        "has_solution_db": bool(
            paths.get("solution_db")
            or (mpath.parent / "world_model" / "solution_db.jsonl").exists()
        ),
        "has_project_snapshot": bool(
            paths.get("project_snapshot")
            or paths.get("pre_stage_project_snapshot")
            or paths.get("post_stage_project_snapshot")
        ),
        "has_claude_session": bool(
            capabilities.get("session_resume_available") or paths.get("claude_session")
        ),
        "has_file_checkpoint": bool(
            capabilities.get("file_rewind_available") or paths.get("file_checkpoints")
        ),
        "pruned": False,
    }
    entry.update(eval_summary)
    return entry


def load_checkpoint_index(checkpoint_root: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_root).expanduser().resolve()
    data = _read_json(root / "checkpoint_index.json")
    if not data:
        data = {
            "schema_version": 1,
            "updated_at": None,
            "latest_checkpoint_id": None,
            "entries": [],
        }
    data.setdefault("entries", [])
    return data


def append_or_update_checkpoint_index(
    checkpoint_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(checkpoint_root).expanduser().resolve()
    entry = entry_from_manifest(root, manifest_path)
    index = load_checkpoint_index(root)
    entries = [
        e
        for e in index.get("entries", [])
        if isinstance(e, dict) and e.get("checkpoint_id") != entry["checkpoint_id"]
    ]
    entries.append(entry)
    entries.sort(key=lambda e: str(e.get("created_at") or ""))
    index.update(
        {
            "schema_version": 1,
            "updated_at": _utc_now(),
            "latest_checkpoint_id": entry["checkpoint_id"],
            "entries": entries,
        }
    )
    _write_json_atomic(root / "checkpoint_index.json", index)
    return index


def rebuild_checkpoint_index(checkpoint_root: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_root).expanduser().resolve()
    entries = []
    for manifest_path in sorted(root.glob("ckpt_*/manifest.json")):
        if manifest_path.is_file():
            entries.append(entry_from_manifest(root, manifest_path))
    latest = _read_json(root / "latest.json")
    latest_id = latest.get("latest_checkpoint_id") or (
        entries[-1].get("checkpoint_id") if entries else None
    )
    index = {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "latest_checkpoint_id": latest_id,
        "entries": entries,
    }
    _write_json_atomic(root / "checkpoint_index.json", index)
    return index


def sync_pruned_entries(checkpoint_root: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_root).expanduser().resolve()
    index = load_checkpoint_index(root)
    entries = []
    for entry in index.get("entries", []):
        if not isinstance(entry, dict):
            continue
        rel = entry.get("manifest_path")
        if isinstance(rel, str) and (root / rel).is_file():
            entries.append(entry)
    latest = _read_json(root / "latest.json")
    latest_id = latest.get("latest_checkpoint_id")
    if latest_id and not any(e.get("checkpoint_id") == latest_id for e in entries):
        latest_id = entries[-1].get("checkpoint_id") if entries else None
    index.update(
        {
            "updated_at": _utc_now(),
            "latest_checkpoint_id": latest_id,
            "entries": entries,
        }
    )
    _write_json_atomic(root / "checkpoint_index.json", index)
    return index


def resolve_checkpoint_by_query(
    checkpoint_root: str | Path, **query: Any
) -> dict[str, Any] | None:
    entries = [
        e
        for e in load_checkpoint_index(checkpoint_root).get("entries", [])
        if isinstance(e, dict)
    ]
    for key, expected in query.items():
        if expected is None:
            continue
        entries = [e for e in entries if e.get(key) == expected]
    return entries[-1] if entries else None
