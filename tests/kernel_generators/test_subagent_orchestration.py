import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.kernel_generators.subagent_orchestration import (
    SubagentFlowConfig,
    SubagentStageConfig,
    filter_stages_after_restore,
    load_default_subagent_flow,
    load_subagent_flows,
    load_subagent_flow,
    render_subagent_stage_prompt,
    run_configured_subagent_flow,
)
from k_search.telemetry.events import TelemetryEvent
from k_search.telemetry.recorder import TelemetryRecorder


def test_load_default_subagent_flow_uses_configured_stage_order():
    flow = load_default_subagent_flow()

    assert flow.name == "ascendc-native-codegen"
    assert [stage.name for stage in flow.stages] == ["code-reader", "designer", "codegen", "reviewer"]
    assert flow.stages[0].agent == "code-reader"
    assert flow.stages[0].run_when_missing_files == ("CODE_MAP.md",)
    assert flow.stages[1].required_files == ("ASCENDC_DESIGN.md",)
    assert flow.stages[2].required_files == (
        "CODE_MAP.md",
        "IMPLEMENTATION_EXECUTION_PLAN.md",
        "IMPLEMENTATION_HANDOFF.md",
    )
    assert flow.stages[3].required_files == ("REVIEW_NOTES.md",)


def test_readme_documents_configured_native_stage_order():
    repo_root = Path(__file__).resolve().parents[2]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "initial_codegen` runs `code-reader`, `designer`, `codegen`, and `reviewer`" in readme
    assert "initial_codegen` runs `code-reader`, `plan`, `codegen`, and `reviewer`" not in readme


def test_load_default_subagent_flows_includes_repair_and_improve_flows():
    flows = load_subagent_flows()

    assert flows.default_flow == "initial_codegen"
    assert set(flows.flows) == {
        "initial_codegen",
        "eval_failure_repair",
        "continue_improve_assessment",
        "continue_improve_codegen",
    }
    repair_flow = flows.get("eval_failure_repair")
    assert repair_flow.trigger == {
        "event": "python_eval_failed",
        "statuses": ["compile_failed", "failed", "benchmark_failed", "timeout"],
    }
    assert [stage.name for stage in repair_flow.stages] == ["bug-fixer", "reviewer"]
    assert [stage.agent for stage in repair_flow.stages] == ["bug-fixer", "reviewer"]
    assert repair_flow.stages[0].required_files == ("CODE_MAP.md",)
    assert repair_flow.stages[1].required_files == ("REVIEW_NOTES.md",)
    assessment_flow = flows.get("continue_improve_assessment")
    assert assessment_flow.trigger == {
        "event": "python_eval_passed_continue_action",
        "statuses": ["passed"],
    }
    assert [stage.name for stage in assessment_flow.stages] == ["improvement-assessor"]
    assert [stage.agent for stage in assessment_flow.stages] == ["improvement-assessor"]
    assert assessment_flow.stages[0].required_files == ("IMPROVEMENT_ASSESSMENT.md",)
    codegen_flow = flows.get("continue_improve_codegen")
    assert [stage.name for stage in codegen_flow.stages] == ["codegen", "reviewer"]
    assert [stage.agent for stage in codegen_flow.stages] == ["codegen", "reviewer"]
    assert codegen_flow.stages[0].required_files == (
        "CODE_MAP.md",
        "IMPLEMENTATION_EXECUTION_PLAN.md",
        "IMPLEMENTATION_HANDOFF.md",
    )
    assert codegen_flow.stages[1].required_files == ("REVIEW_NOTES.md",)


def test_load_subagent_flow_uses_env_config_path(tmp_path, monkeypatch):
    config_path = tmp_path / "custom_flow.json"
    config_path.write_text(
        """
        {
          "version": 1,
          "name": "custom-flow",
          "description": "Custom flow.",
          "stages": [
            {
              "name": "extra-review",
              "agent": "reviewer",
              "instruction": "Run an extra review.",
              "required_files": ["REVIEW_NOTES.md"]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("KSEARCH_SUBAGENT_FLOW_CONFIG", str(config_path))

    flow = load_subagent_flow()

    assert flow.name == "custom-flow"
    assert [stage.name for stage in flow.stages] == ["extra-review"]


def test_render_subagent_stage_prompt_names_exact_agent_and_required_outputs():
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="designer",
                agent="designer",
                instruction="Write the detailed design.",
                required_files=("ASCENDC_DESIGN.md",),
            ),
        ),
    )

    prompt = render_subagent_stage_prompt(
        flow=flow,
        stage=flow.stages[0],
        active_stage_index=1,
        active_stage_count=1,
        base_prompt="BASE ATTEMPT CONTEXT",
    )

    assert "Stage 1/1: designer" in prompt
    assert "Current agent: designer" in prompt
    assert "Use the designer subagent for this stage." in prompt
    assert "Do not invoke any other subagent during this stage." in prompt
    assert "ASCENDC_DESIGN.md" in prompt
    assert "Required reads:" in prompt
    assert ".ksearch/context/STRATEGY.md" in prompt
    assert "BASE ATTEMPT CONTEXT" in prompt
    assert "Flow:" not in prompt
    assert "Initial codegen flow agents" not in prompt
    assert "Eval-failure repair flow agents" not in prompt
    assert "bug-fixer" not in prompt


def test_render_subagent_stage_prompt_constrains_parent_agent_subagent_prompt():
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="designer",
                agent="designer",
                instruction="Write the detailed design.",
                required_files=("ASCENDC_DESIGN.md",),
            ),
        ),
    )

    prompt = render_subagent_stage_prompt(
        flow=flow,
        stage=flow.stages[0],
        active_stage_index=1,
        active_stage_count=1,
        base_prompt="BASE ATTEMPT CONTEXT",
    )

    assert "Project root contract:" in prompt
    assert 'Treat "." as the candidate project root.' in prompt
    assert "Parent-agent dispatch contract:" in prompt
    assert "When constructing the Agent prompt, preserve the project root contract verbatim." in prompt
    assert "Do not invent a parent directory, sibling task directory, archive directory, or historical run artifact as CWD." in prompt
    assert "Do not introduce task-layout assumptions not present in the candidate project." in prompt


def test_stage_prompt_hygiene_blocks_global_flow_policy(tmp_path):
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="designer",
                agent="designer",
                instruction="Create ASCENDC_DESIGN.md.",
                required_files=("ASCENDC_DESIGN.md",),
            ),
        ),
    )

    class SessionClient:
        def open_session(self, *, project_dir, telemetry_recorder=None):
            return SimpleNamespace(project_dir=Path(project_dir), _closed=False)

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            raise AssertionError("prompt with global policy should not be sent")

        def close_session(self, session):
            session._closed = True

    with pytest.raises(RuntimeError, match="global flow policy"):
        run_configured_subagent_flow(
            editor_client=SessionClient(),
            project_dir=tmp_path,
            base_prompt="Initial codegen flow agents: code-reader, designer, codegen, reviewer.",
            flow=flow,
        )


def test_run_configured_subagent_flow_uses_one_session_and_skips_reader_when_code_map_exists(tmp_path):
    (tmp_path / "CODE_MAP.md").write_text("# CODE_MAP\npreseeded\n", encoding="utf-8")
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="code-reader",
                agent="code-reader",
                instruction="Create CODE_MAP.md.",
                required_files=("CODE_MAP.md",),
                run_when_missing_files=("CODE_MAP.md",),
            ),
            SubagentStageConfig(
                name="designer",
                agent="designer",
                instruction="Create ASCENDC_DESIGN.md.",
                required_files=("ASCENDC_DESIGN.md",),
            ),
            SubagentStageConfig(
                name="reviewer",
                agent="reviewer",
                instruction="Create REVIEW_NOTES.md.",
                required_files=("REVIEW_NOTES.md",),
            ),
        ),
    )

    class SessionClient:
        def __init__(self):
            self.open_count = 0
            self.closed = False
            self.prompts: list[str] = []

        def open_session(self, *, project_dir, telemetry_recorder=None):
            self.open_count += 1
            return SimpleNamespace(project_dir=Path(project_dir), _closed=False)

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(session.project_dir)
            if "Stage 1/2: designer" in prompt:
                (root / "ASCENDC_DESIGN.md").write_text("# design\n" + "detail\n" * 20, encoding="utf-8")
                text = "designer done"
            elif "Stage 2/2: reviewer" in prompt:
                (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
                text = "review done"
            else:
                raise AssertionError(f"unexpected prompt: {prompt}")
            return ClaudeProjectEditResult(
                text=text,
                transcript=text,
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            self.closed = True
            session._closed = True

    client = SessionClient()

    result = run_configured_subagent_flow(
        editor_client=client,
        project_dir=tmp_path,
        base_prompt="BASE",
        flow=flow,
    )

    assert client.open_count == 1
    assert client.closed is True
    assert len(client.prompts) == 2
    assert "code-reader" not in "\n".join(client.prompts)
    assert "Use the designer subagent" in client.prompts[0]
    assert "Use the reviewer subagent" in client.prompts[1]
    assert result.text == "review done"
    assert result.transcript == "designer done\nreview done"


def test_run_configured_subagent_flow_fails_when_required_file_missing(tmp_path):
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="designer",
                agent="designer",
                instruction="Create ASCENDC_DESIGN.md.",
                required_files=("ASCENDC_DESIGN.md",),
            ),
        ),
    )

    class SessionClient:
        def open_session(self, *, project_dir, telemetry_recorder=None):
            return SimpleNamespace(project_dir=Path(project_dir), _closed=False)

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            return ClaudeProjectEditResult(
                text="missing design",
                transcript="missing design",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            session._closed = True

    with pytest.raises(RuntimeError, match="ASCENDC_DESIGN.md"):
        run_configured_subagent_flow(
            editor_client=SessionClient(),
            project_dir=tmp_path,
            base_prompt="BASE",
            flow=flow,
        )


def test_run_configured_subagent_flow_recovers_required_file_written_outside_project(tmp_path, caplog):
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="reviewer",
                agent="reviewer",
                instruction="Create REVIEW_NOTES.md.",
                required_files=("REVIEW_NOTES.md",),
            ),
        ),
    )
    outside = tmp_path.parent / f"{tmp_path.name}_wrong_recover" / "agent_workdir" / "flash_attention" / "REVIEW_NOTES.md"

    class SessionClient:
        def open_session(self, *, project_dir, telemetry_recorder=None):
            return SimpleNamespace(project_dir=Path(project_dir), _closed=False)

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            telemetry_recorder.emit(
                TelemetryEvent(
                    event_type="tool_use",
                    tool_name="Write",
                    tool_input={"file_path": str(outside), "content": "status: ok\n"},
                )
            )
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_text("status: ok\n", encoding="utf-8")
            return ClaudeProjectEditResult(
                text="review done",
                transcript="review done",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            session._closed = True

    recorder = TelemetryRecorder()

    result = run_configured_subagent_flow(
        editor_client=SessionClient(),
        project_dir=tmp_path,
        base_prompt="BASE",
        flow=flow,
        telemetry_recorder=recorder,
    )

    assert result.text == "review done"
    assert (tmp_path / "REVIEW_NOTES.md").read_text(encoding="utf-8") == "status: ok\n"
    assert "recovered subagent required file written outside project root" in caplog.text
    assert any(event.event_type == "subagent_handoff_recovered" for event in recorder.events)


def test_run_configured_subagent_flow_reports_missing_external_write_source(tmp_path):
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="reviewer",
                agent="reviewer",
                instruction="Create REVIEW_NOTES.md.",
                required_files=("REVIEW_NOTES.md",),
            ),
        ),
    )
    outside = tmp_path.parent / f"{tmp_path.name}_wrong_missing" / "agent_workdir" / "flash_attention" / "REVIEW_NOTES.md"

    class SessionClient:
        def open_session(self, *, project_dir, telemetry_recorder=None):
            return SimpleNamespace(project_dir=Path(project_dir), _closed=False)

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            telemetry_recorder.emit(
                TelemetryEvent(
                    event_type="tool_use",
                    tool_name="Write",
                    tool_input={"file_path": str(outside), "content": "status: ok\n"},
                )
            )
            return ClaudeProjectEditResult(
                text="review done",
                transcript="review done",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            session._closed = True

    with pytest.raises(RuntimeError) as exc:
        run_configured_subagent_flow(
            editor_client=SessionClient(),
            project_dir=tmp_path,
            base_prompt="BASE",
            flow=flow,
            telemetry_recorder=TelemetryRecorder(),
        )

    message = str(exc.value)
    assert "outside candidate project root" in message
    assert "REVIEW_NOTES.md" in message
    assert str(outside) in message


def test_run_configured_subagent_flow_recovers_external_write_from_trace_file(tmp_path):
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="reviewer",
                agent="reviewer",
                instruction="Create REVIEW_NOTES.md.",
                required_files=("REVIEW_NOTES.md",),
            ),
        ),
    )
    outside = tmp_path.parent / "wrong_run_trace" / "agent_workdir" / "flash_attention" / "REVIEW_NOTES.md"
    trace_path = tmp_path / "logs" / "agent_trace.jsonl"

    class SessionClient:
        def open_session(self, *, project_dir, telemetry_recorder=None):
            return SimpleNamespace(project_dir=Path(project_dir), _closed=False)

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_text("status: ok\n", encoding="utf-8")
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(
                    {
                        "event_type": "tool_use",
                        "tool_name": "Write",
                        "tool_input": {"file_path": str(outside), "content": "status: ok\n"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return ClaudeProjectEditResult(
                text="review done",
                transcript="review done",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            session._closed = True

    recorder = TelemetryRecorder(artifacts=SimpleNamespace(trace_path=str(trace_path)))

    run_configured_subagent_flow(
        editor_client=SessionClient(),
        project_dir=tmp_path,
        base_prompt="BASE",
        flow=flow,
        telemetry_recorder=recorder,
    )

    assert (tmp_path / "REVIEW_NOTES.md").read_text(encoding="utf-8") == "status: ok\n"


def test_run_configured_subagent_flow_saves_stage_checkpoints_in_order(tmp_path):
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="designer",
                agent="designer",
                instruction="Create ASCENDC_DESIGN.md.",
                required_files=("ASCENDC_DESIGN.md",),
            ),
            SubagentStageConfig(
                name="reviewer",
                agent="reviewer",
                instruction="Create REVIEW_NOTES.md.",
                required_files=("REVIEW_NOTES.md",),
            ),
        ),
    )

    class StageCheckpointRecorder:
        def __init__(self):
            self.calls = []

        def save_stage_start(self, **kwargs):
            self.calls.append(("start", kwargs["stage"].name, kwargs["stage_index"], kwargs["round_num"], kwargs["attempt_idx"]))
            return tmp_path / "start_manifest.json"

        def save_stage_completed(self, **kwargs):
            self.calls.append(("completed", kwargs["stage"].name, kwargs["stage_index"], kwargs["round_num"], kwargs["attempt_idx"]))
            return tmp_path / "completed_manifest.json"

    class SessionClient:
        def open_session(self, *, project_dir, telemetry_recorder=None):
            return SimpleNamespace(project_dir=Path(project_dir), _closed=False)

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            root = Path(session.project_dir)
            if "Stage 1/2: designer" in prompt:
                (root / "ASCENDC_DESIGN.md").write_text("# design\n", encoding="utf-8")
                text = "designer done"
            elif "Stage 2/2: reviewer" in prompt:
                (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
                text = "reviewer done"
            else:
                raise AssertionError(f"unexpected prompt: {prompt}")
            return ClaudeProjectEditResult(
                text=text,
                transcript=text,
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            session._closed = True

    recorder = StageCheckpointRecorder()

    run_configured_subagent_flow(
        editor_client=SessionClient(),
        project_dir=tmp_path,
        base_prompt="BASE",
        flow=flow,
        stage_checkpoint_manager=recorder,
        checkpoint_task=SimpleNamespace(name="task"),
        round_num=4,
        attempt_idx=2,
        runtime_state={"attempt": {"flow_name": "test-flow"}},
    )

    assert recorder.calls == [
        ("start", "designer", 1, 4, 2),
        ("completed", "designer", 1, 4, 2),
        ("start", "reviewer", 2, 4, 2),
        ("completed", "reviewer", 2, 4, 2),
    ]


def test_filter_stages_after_restore_skips_completed_and_starts_at_next_pending(tmp_path):
    (tmp_path / "CODE_MAP.md").write_text("# map\n", encoding="utf-8")
    (tmp_path / "ASCENDC_DESIGN.md").write_text("# design\n", encoding="utf-8")
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="code-reader",
                agent="code-reader",
                instruction="Create CODE_MAP.md.",
                required_files=("CODE_MAP.md",),
            ),
            SubagentStageConfig(
                name="designer",
                agent="designer",
                instruction="Create ASCENDC_DESIGN.md.",
                required_files=("ASCENDC_DESIGN.md",),
            ),
            SubagentStageConfig(
                name="codegen",
                agent="codegen",
                instruction="Create implementation.",
                required_files=("IMPLEMENTATION_HANDOFF.md",),
            ),
            SubagentStageConfig(
                name="reviewer",
                agent="reviewer",
                instruction="Review.",
                required_files=("REVIEW_NOTES.md",),
            ),
        ),
    )
    restored_state = {
        "stages": [
            {"index": 1, "name": "code-reader", "status": "completed", "required_files": ["CODE_MAP.md"]},
            {"index": 2, "name": "designer", "status": "completed", "required_files": ["ASCENDC_DESIGN.md"]},
            {"index": 3, "name": "codegen", "status": "pending", "required_files": ["IMPLEMENTATION_HANDOFF.md"]},
            {"index": 4, "name": "reviewer", "status": "pending", "required_files": ["REVIEW_NOTES.md"]},
        ]
    }

    active = filter_stages_after_restore(
        active_stages=list(flow.stages),
        restored_stage_state=restored_state,
        project_root=tmp_path,
    )

    assert [stage.name for stage in active] == ["codegen", "reviewer"]


def test_filter_stages_after_restore_reruns_running_stage(tmp_path):
    (tmp_path / "CODE_MAP.md").write_text("# map\n", encoding="utf-8")
    flow = SubagentFlowConfig(
        name="test-flow",
        description="Test flow.",
        stages=(
            SubagentStageConfig(
                name="code-reader",
                agent="code-reader",
                instruction="Create CODE_MAP.md.",
                required_files=("CODE_MAP.md",),
            ),
            SubagentStageConfig(
                name="designer",
                agent="designer",
                instruction="Create ASCENDC_DESIGN.md.",
                required_files=("ASCENDC_DESIGN.md",),
            ),
            SubagentStageConfig(
                name="codegen",
                agent="codegen",
                instruction="Create implementation.",
                required_files=("IMPLEMENTATION_HANDOFF.md",),
            ),
        ),
    )
    restored_state = {
        "stages": [
            {"index": 1, "name": "code-reader", "status": "completed", "required_files": ["CODE_MAP.md"]},
            {"index": 2, "name": "designer", "status": "running", "required_files": ["ASCENDC_DESIGN.md"]},
            {"index": 3, "name": "codegen", "status": "pending", "required_files": ["IMPLEMENTATION_HANDOFF.md"]},
        ]
    }

    active = filter_stages_after_restore(
        active_stages=list(flow.stages),
        restored_stage_state=restored_state,
        project_root=tmp_path,
    )

    assert [stage.name for stage in active] == ["designer", "codegen"]
