from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticPromptBuilder,
    _candidate_changed_paths,
    _candidate_diff_text,
)
from k_search.kernel_generators.worktree_context import WorktreeContextPaths


def test_prompt_uses_context_paths_not_full_strategy_or_raw_log():
    paths = WorktreeContextPaths(
        strategy_md=".ksearch/context/STRATEGY.md",
        strategy_summary_md=".ksearch/context/STRATEGY_SUMMARY.md",
        eval_summary_json=".ksearch/context/EVAL_SUMMARY.json",
        eval_log_md=".ksearch/context/EVAL_LOG.md",
        manifest_json=".ksearch/context/CONTEXT_MANIFEST.json",
    )
    request = AscendCAgenticCodegenRequest(
        definition_text="spec",
        action_text=(
            "Chosen action\n\n"
            "Full natural-language strategy markdown:\n"
            "SECRET FULL STRATEGY BODY"
        ),
        trace_logs="[benchmark stderr]\nPermission mismatch\n/tmp/work/project",
        perf_summary="mean_latency_us=2396.596",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
        context_paths=paths,
        strategy_summary="Short strategy summary.",
        eval_summary={
            "eval_context_status": "performance_measured",
            "has_eval_log": False,
            "performance_available": True,
        },
    )

    prompt = AscendCAgenticPromptBuilder(max_chars=20_000).build(request)

    assert ".ksearch/context/STRATEGY.md" in prompt
    assert ".ksearch/context/EVAL_SUMMARY.json" in prompt
    assert "Short strategy summary." in prompt
    assert "performance_measured" in prompt
    assert "Full natural-language strategy markdown" not in prompt
    assert "SECRET FULL STRATEGY BODY" not in prompt
    assert "Recent failure or trace excerpt" not in prompt
    assert "[benchmark stderr]" not in prompt
    assert "Permission mismatch" not in prompt
    assert "/tmp/" not in prompt


def test_prompt_includes_selected_strategy_dependency_status_without_blocked_nodes():
    paths = WorktreeContextPaths(
        strategy_md=".ksearch/context/STRATEGY.md",
        strategy_summary_md=".ksearch/context/STRATEGY_SUMMARY.md",
        eval_summary_json=".ksearch/context/EVAL_SUMMARY.json",
        eval_log_md=".ksearch/context/EVAL_LOG.md",
        manifest_json=".ksearch/context/CONTEXT_MANIFEST.json",
    )
    request = AscendCAgenticCodegenRequest(
        definition_text="spec",
        action_text="Selected executable strategy.",
        trace_logs="",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=2,
        attempt_idx=1,
        mode="action",
        context_paths=paths,
        strategy_summary="Soft pipeline after two-level tiling.",
        strategy_context={
            "strategy_id": "fa_multibuffer_soft_pipeline",
            "requires": ["fa_qkv_two_level_l1_reuse"],
            "dependencies_satisfied": True,
            "parent_strategy_lineage": ["fa_qkv_two_level_l1_reuse"],
            "parent_solution_id": "round_0001_attempt_0001",
        },
        blocked_strategy_nodes=[
            {
                "node_id": "s2c1",
                "reason": "strategy_file_required_but_missing",
            }
        ],
    )

    prompt = AscendCAgenticPromptBuilder(max_chars=20_000).build(request)

    assert "Dependency check:" in prompt
    assert "- strategy_id: fa_multibuffer_soft_pipeline" in prompt
    assert "- requires: fa_qkv_two_level_l1_reuse" in prompt
    assert "- dependency_status: satisfied" in prompt
    assert "s2c1" not in prompt
    assert "strategy_file_required_but_missing" not in prompt


def test_ksearch_context_not_in_changed_paths_or_candidate_diff():
    changed = _candidate_changed_paths(
        [
            "kernel/operator.cpp",
            ".ksearch/context/STRATEGY.md",
            ".ksearch/context/EVAL_SUMMARY.json",
        ]
    )

    diff = _candidate_diff_text(
        "\n".join(
            [
                "--- /dev/null",
                "+++ b/.ksearch/context/STRATEGY.md",
                "@@",
                "+secret context",
                "--- /dev/null",
                "+++ b/kernel/operator.cpp",
                "@@",
                "+source change",
            ]
        )
    )

    assert changed == ["kernel/operator.cpp"]
    assert ".ksearch/context/STRATEGY.md" not in diff
    assert "secret context" not in diff
    assert "kernel/operator.cpp" in diff
    assert "source change" in diff
