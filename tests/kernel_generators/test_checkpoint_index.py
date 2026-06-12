from __future__ import annotations

import json

from k_search.kernel_generators.checkpoint_index import (
    append_or_update_checkpoint_index,
    resolve_checkpoint_by_query,
    sync_pruned_entries,
)


def test_checkpoint_index_appends_queries_and_syncs_pruned(tmp_path):
    root = tmp_path / "checkpoints"
    ckpt = root / "ckpt_000001_attempt_r0001"
    ckpt.mkdir(parents=True)
    (ckpt / "runtime_state.json").write_text(
        json.dumps({"resume_action": "continue_current_action"}), encoding="utf-8"
    )
    manifest = {
        "checkpoint_version": "v2",
        "checkpoint_kind": "attempt_boundary",
        "checkpoint_id": ckpt.name,
        "created_at": "2026-06-11T00:00:00Z",
        "search": {
            "round_index": 1,
            "attempt_idx": 2,
            "action_node_id": "s1",
            "best_score": 2.0,
        },
        "current_solution": {
            "solution_id": "cur",
            "eval": {"status": "passed", "latency_ms": 1.5},
            "score": 2.0,
        },
        "best_solution": {
            "solution_id": "best",
            "eval": {"status": "passed"},
            "score": 2.0,
        },
        "paths": {
            "runtime_state": "runtime_state.json",
            "world_model": "world_model/world_model.json",
            "solution_db": "world_model/solution_db.jsonl",
        },
    }
    (ckpt / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    index = append_or_update_checkpoint_index(root, ckpt / "manifest.json")
    assert index["latest_checkpoint_id"] == ckpt.name
    entry = resolve_checkpoint_by_query(root, eval_status="passed", round=1, attempt=2)
    assert entry is not None
    assert entry["manifest_path"] == f"{ckpt.name}/manifest.json"
    assert entry["score"] == 2.0

    (ckpt / "manifest.json").unlink()
    synced = sync_pruned_entries(root)
    assert synced["entries"] == []
