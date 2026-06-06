from pathlib import Path
from types import SimpleNamespace

import pytest

from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.kernel_generators.subagent_orchestration import (
    SubagentFlowConfig,
    SubagentStageConfig,
    load_default_subagent_flow,
    load_subagent_flows,
    load_subagent_flow,
    render_subagent_stage_prompt,
    run_configured_subagent_flow,
)


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


def test_load_default_subagent_flows_includes_eval_failure_repair_flow():
    flows = load_subagent_flows()

    assert flows.default_flow == "initial_codegen"
    assert set(flows.flows) == {"initial_codegen", "eval_failure_repair"}
    repair_flow = flows.get("eval_failure_repair")
    assert repair_flow.trigger == {
        "event": "python_eval_failed",
        "statuses": ["compile_failed", "failed", "benchmark_failed", "timeout"],
    }
    assert [stage.name for stage in repair_flow.stages] == ["bug-fixer", "reviewer"]
    assert [stage.agent for stage in repair_flow.stages] == ["bug-fixer", "reviewer"]
    assert repair_flow.stages[0].required_files == ("CODE_MAP.md",)
    assert repair_flow.stages[1].required_files == ("REVIEW_NOTES.md",)


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
    assert "Use the designer subagent for this stage." in prompt
    assert "Do not invoke any other subagent during this stage." in prompt
    assert "ASCENDC_DESIGN.md" in prompt
    assert "BASE ATTEMPT CONTEXT" in prompt


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
