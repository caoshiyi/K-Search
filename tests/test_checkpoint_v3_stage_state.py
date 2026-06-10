import json
from pathlib import Path
from types import SimpleNamespace

from k_search.kernel_generators.checkpoint_v3 import (
    StageCheckpointConfig,
    StageCheckpointManager,
)
from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.kernel_generators.subagent_orchestration import SubagentFlowConfig, SubagentStageConfig


def _flow_and_stage() -> tuple[SubagentFlowConfig, SubagentStageConfig]:
    stage = SubagentStageConfig(
        name="designer",
        agent="designer",
        instruction="Create ASCENDC_DESIGN.md.",
        required_files=("ASCENDC_DESIGN.md",),
    )
    return (
        SubagentFlowConfig(name="initial_codegen", description="test", stages=(stage,), version=3),
        stage,
    )


def test_stage_checkpoint_manager_saves_start_completed_latest_and_restores_project(tmp_path):
    project_dir = tmp_path / "project"
    (project_dir / "kernel").mkdir(parents=True)
    (project_dir / "kernel" / "foo.h").write_text("alpha\n", encoding="utf-8")
    flow, stage = _flow_and_stage()
    manager = StageCheckpointManager(
        artifacts_dir=tmp_path / "run_artifacts",
        task_name="vec_add",
        task_id="task-1",
        run_id="run-1",
        config=StageCheckpointConfig(enabled=True),
    )

    start_manifest = manager.save_stage_start(
        task=SimpleNamespace(name="vec_add", task_path=str(project_dir)),
        project_dir=project_dir,
        flow=flow,
        stage=stage,
        stage_index=1,
        round_num=7,
        attempt_idx=2,
        prompt="designer prompt",
        session=SimpleNamespace(session_id="session-1"),
        runtime_state={"resume_action": "run_stage"},
    )

    start_root = start_manifest.parent
    start_stage_state = json.loads((start_root / "stage_state.json").read_text(encoding="utf-8"))
    assert json.loads(start_manifest.read_text(encoding="utf-8"))["checkpoint_kind"] == "stage_start"
    assert start_stage_state["stages"][0]["status"] == "running"
    assert start_stage_state["stages"][0]["missing_files"] == ["ASCENDC_DESIGN.md"]
    assert (start_root / "stage" / "stage_prompts" / "stage_01_designer.md").read_text(encoding="utf-8") == "designer prompt"

    (project_dir / "ASCENDC_DESIGN.md").write_text("# design\n", encoding="utf-8")
    result = ClaudeProjectEditResult(
        text="designer done",
        transcript="transcript",
        prompt="designer prompt",
        prompt_chars=len("designer prompt"),
        prompt_lines=1,
        session_id="session-1",
        file_checkpoint_uuid="user-message-1",
        subagent_agent_ids=["agent-123"],
        subagent_invocations=[{"agent": "designer", "agent_id": "agent-123"}],
    )
    completed_manifest = manager.save_stage_completed(
        task=SimpleNamespace(name="vec_add", task_path=str(project_dir)),
        project_dir=project_dir,
        flow=flow,
        stage=stage,
        stage_index=1,
        round_num=7,
        attempt_idx=2,
        prompt="designer prompt",
        result=result,
        session=SimpleNamespace(session_id="session-1"),
        telemetry_recorder=None,
        runtime_state={"resume_action": "continue_next_pending_stage"},
    )

    checkpoints_dir = tmp_path / "run_artifacts" / "checkpoints"
    latest = json.loads((checkpoints_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["latest_checkpoint_path"] == f"{completed_manifest.parent.name}/manifest.json"

    completed_root = completed_manifest.parent
    manifest = json.loads(completed_manifest.read_text(encoding="utf-8"))
    stage_state = json.loads((completed_root / "stage_state.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_version"] == "v3"
    assert manifest["checkpoint_kind"] == "stage_completed"
    assert manifest["position"]["last_completed_stage_name"] == "designer"
    assert stage_state["stages"][0]["status"] == "completed"
    assert stage_state["stages"][0]["produced_files"] == ["ASCENDC_DESIGN.md"]
    assert stage_state["stages"][0]["agent_id"] == "agent-123"
    assert stage_state["stages"][0]["file_checkpoint_uuid"] == "user-message-1"
    assert json.loads((completed_root / "claude" / "subagents.json").read_text(encoding="utf-8"))["subagents"]["designer"]["agent_id"] == "agent-123"

    restored = manager.restore("latest", target_run_id="resume-run")
    assert restored.checkpoint_id == manifest["checkpoint_id"]
    assert restored.next_stage_index is None
    assert restored.all_stages_completed is True
    assert (restored.restored_project_dir / "ASCENDC_DESIGN.md").read_text(encoding="utf-8") == "# design\n"


def test_stage_checkpoint_restore_running_stage_uses_pre_stage_snapshot(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "kernel.h").write_text("before\n", encoding="utf-8")
    flow, stage = _flow_and_stage()
    manager = StageCheckpointManager(
        artifacts_dir=tmp_path / "run_artifacts",
        task_name="vec_add",
        task_id="task-1",
        run_id="run-1",
        config=StageCheckpointConfig(enabled=True),
    )

    start_manifest = manager.save_stage_start(
        task=SimpleNamespace(name="vec_add", task_path=str(project_dir)),
        project_dir=project_dir,
        flow=flow,
        stage=stage,
        stage_index=1,
        round_num=1,
        attempt_idx=1,
        prompt="designer prompt",
        session=None,
        runtime_state={},
    )
    (project_dir / "kernel.h").write_text("mutated after checkpoint\n", encoding="utf-8")

    restored = manager.restore(str(start_manifest), target_run_id="resume-run")

    assert restored.next_stage_index == 1
    assert restored.next_stage_name == "designer"
    assert restored.all_stages_completed is False
    assert (restored.restored_project_dir / "kernel.h").read_text(encoding="utf-8") == "before\n"
