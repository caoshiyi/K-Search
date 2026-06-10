import pytest

from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticPromptBuilder,
    _build_repair_prompt,
)
from k_search.kernel_generators.subagent_orchestration import (
    SubagentFlowConfig,
    SubagentStageConfig,
    _validate_stage_agent_is_in_flow,
    load_subagent_flows,
)
from k_search.tasks.task_base import EvalResult


def _request() -> AscendCAgenticCodegenRequest:
    return AscendCAgenticCodegenRequest(
        definition_text="spec",
        action_text="change kernel",
        trace_logs="compile failed",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
    )


def test_initial_prompt_does_not_globally_ban_bug_fixer():
    prompt = AscendCAgenticPromptBuilder(max_chars=20_000).build(
        _request(),
    )

    assert "must not be invoked in this release" not in prompt
    assert "eval_failure_repair" not in prompt
    assert "bug-fixer" not in prompt
    assert "Required native subagent flow" not in prompt
    assert "Initial codegen flow agents" not in prompt
    assert "Eval-failure repair flow agents" not in prompt


def test_repair_prompt_explicitly_allows_bug_fixer():
    eval_result = EvalResult(status="compile_failed", log_excerpt="missing semicolon", metrics={})

    prompt = _build_repair_prompt(eval_result, fix_round=1)

    assert "eval_failure_repair" in prompt
    assert "bug-fixer" in prompt
    assert "must not be invoked" not in prompt


def test_eval_failure_repair_flow_contains_bug_fixer_then_reviewer():
    repair = load_subagent_flows().get("eval_failure_repair")

    assert [stage.agent for stage in repair.stages] == ["bug-fixer", "reviewer"]


def test_continue_improve_flow_assesses_improvement_before_codegen_without_bug_fixer():
    improve = load_subagent_flows().get("continue_improve")

    assert [stage.agent for stage in improve.stages] == ["improvement-assessor", "codegen", "reviewer"]
    assert "bug-fixer" not in [stage.agent for stage in improve.stages]
    assert improve.stages[0].required_files == ("IMPROVEMENT_ASSESSMENT.md",)


def test_stage_agent_not_declared_in_flow_raises():
    flow = SubagentFlowConfig(
        name="repair",
        description="repair flow",
        stages=(
            SubagentStageConfig(
                name="reviewer",
                agent="reviewer",
                instruction="review",
            ),
        ),
    )
    stage = SubagentStageConfig(
        name="bug-fixer",
        agent="bug-fixer",
        instruction="fix",
    )

    with pytest.raises(RuntimeError, match="not declared in flow"):
        _validate_stage_agent_is_in_flow(flow, stage)
