from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from k_search.kernel_generators.agentic_candidate_artifacts import (
    write_agentic_candidate_artifacts,
)
from k_search.kernel_generators.project_snapshot import create_project_snapshot


class DummyEval:
    def to_dict(self, include_log_excerpt=True, max_log_chars=8000):
        return {"status": "passed", "latency_ms": 1.25, "log_excerpt": "ok"}


def test_agentic_artifacts_are_attempt_centric_with_snapshot_tarball(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-1")
    project = tmp_path / "project"
    project.mkdir()
    (project / "kernel.cpp").write_text("int main(){}\n", encoding="utf-8")
    snapshot = create_project_snapshot(
        project_dir=project,
        snapshot_id="snap-1",
        parent_snapshot_id=None,
        base_commit="base",
        created_by_round=1,
        eval_result={"status": "passed"},
    )

    candidate, paths = write_agentic_candidate_artifacts(
        artifacts_dir=tmp_path / "out",
        task_name="flash/attn",
        run_id="run-1",
        round_num=1,
        attempt_idx=2,
        prompt="prompt",
        transcript="transcript",
        changed_paths=["kernel.cpp"],
        diff_text="diff --git a/kernel.cpp b/kernel.cpp\n",
        eval_result=DummyEval(),
        project_snapshot=snapshot,
        parent_candidate_id=None,
        base_ref="base",
        project_rel_path=".",
        action_node_id="s/1",
        model_name="claude",
        handoff_files={"CODE_MAP.md": "map"},
        stage_prompt_records=[{"path": "stage_prompts/stage_01_codegen.md"}],
    )

    attempt_dir = (
        tmp_path
        / "out"
        / "flash_attn"
        / "task-1"
        / "runs"
        / "run-1"
        / "attempts"
        / "r0001_a02_s_1"
    )
    assert Path(paths["manifest_path"]) == attempt_dir / "manifest.json"
    assert (attempt_dir / "prompt.md").read_text(encoding="utf-8") == "prompt"
    assert (attempt_dir / "eval.json").is_file()
    assert (attempt_dir / "diff.patch").is_file()
    assert (attempt_dir / "handoff" / "CODE_MAP.md").is_file()
    assert (attempt_dir / "snapshot" / "snapshot.json").is_file()
    assert (attempt_dir / "snapshot" / "project.tar.gz").is_file()

    manifest = json.loads((attempt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["diff_path"] == "diff.patch"
    assert manifest["changed_paths"] == ["kernel.cpp"]
    assert manifest["snapshot_archive_path"] == "snapshot/project.tar.gz"
    assert manifest["native_handoff_paths"] == {"CODE_MAP.md": "handoff/CODE_MAP.md"}
    artifact_index = json.loads(
        (attempt_dir.parents[1] / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert artifact_index["attempts"][0]["attempt_dir"] == "attempts/r0001_a02_s_1"
    assert (
        artifact_index["attempts"][0]["manifest_path"]
        == "attempts/r0001_a02_s_1/manifest.json"
    )
    assert candidate.manifest_path == str(attempt_dir / "manifest.json")
    assert not (
        tmp_path
        / "out"
        / "flash_attn"
        / "task-1"
        / "runs"
        / "run-1"
        / "artifacts"
        / "candidates"
    ).exists()
