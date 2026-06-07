from types import SimpleNamespace

from k_search.kernel_generators.eval_context import build_eval_context_for_llm
from k_search.tasks.task_base import EvalResult


def test_no_prior_eval_uses_reference_only_when_task_has_reference_latency():
    task = SimpleNamespace(reference_latency_ms=2.391)

    summary, log = build_eval_context_for_llm(eval_result=None, task=task)

    assert summary["eval_context_status"] == "reference_only"
    assert summary["has_prior_candidate_eval"] is False
    assert summary["has_eval_log"] is False
    assert summary["performance_available"] is False
    assert summary["baseline"]["original_baseline_latency_us"] is None
    assert summary["baseline"]["parent_latency_us"] is None
    assert summary["baseline"]["reference_latency_us"] == 2391.0
    assert "reference_latency_ms" not in summary["baseline"]
    assert "No previous candidate evaluation exists" in log


def test_compile_failed_context_contains_only_compile_error():
    result = EvalResult(
        status="compile_failed",
        log_excerpt=(
            "[build stderr]\n"
            "/tmp/work/project/kernel.cpp:10: error: expected ';'\n"
            "Permission mismatch\n"
            "[correctness stderr]\n"
            "correctness should not be shown\n"
            "[benchmark stdout]\n"
            "profiler.py: Start parsing profiling data\n"
            "mean=2396.596us\n"
        ),
    )

    summary, log = build_eval_context_for_llm(eval_result=result, task=SimpleNamespace())

    assert summary["eval_context_status"] == "compile_failed"
    assert summary["diagnostic_kind"] == "compile_error"
    assert summary["has_eval_log"] is True
    assert "Compile Error" in log
    assert "expected ';'" in log
    assert "Correctness Failure" not in log
    assert "benchmark" not in log.lower()
    assert "Permission mismatch" not in log
    assert "/tmp/" not in log


def test_correctness_failed_context_contains_only_failing_cases():
    result = EvalResult(
        status="failed",
        log_excerpt=(
            "[build stderr]\n"
            "build warning should not be shown\n"
            "[correctness stderr]\n"
            "case_id=7 shape=[1,16,128] max_abs_error=0.5 tolerance=0.01\n"
            "Permission mismatch\n"
            "[benchmark stderr]\n"
            "mean_latency_us=999\n"
        ),
    )

    summary, log = build_eval_context_for_llm(eval_result=result, task=SimpleNamespace())

    assert summary["eval_context_status"] == "correctness_failed"
    assert summary["performance_available"] is False
    assert summary["diagnostic_kind"] == "correctness_failure"
    assert "Correctness Failure" in log
    assert "case_id=7" in log
    assert "max_abs_error" in log
    assert "Compile Error" not in log
    assert "mean_latency_us" not in log
    assert "Permission mismatch" not in log


def test_passed_context_contains_performance_only():
    result = EvalResult(
        status="passed",
        latency_ms=2.396596,
        reference_latency_ms=2.391,
        mean_vs_baseline_factor=0.998,
        speedup_factor=1.001,
        metrics={
            "score": 0.998,
            "min_latency_ms": 2.3917,
            "max_latency_ms": 2.40204,
            "num_runs": 20,
            "parent_latency_ms": 1.59,
            "parent_solution_id": "sol_s1_adopted",
            "parent_strategy_id": "fa_qkv_two_level_l1_reuse",
            "speedup_vs_parent": 1.627,
        },
        log_excerpt=(
            "[benchmark stdout]\n"
            "profiler.py: Start parsing profiling data\n"
            "mean=2396.596us\n"
            "Permission mismatch\n"
        ),
    )

    summary, log = build_eval_context_for_llm(eval_result=result, task=SimpleNamespace())

    assert summary["eval_context_status"] == "performance_measured"
    assert summary["has_eval_log"] is False
    assert summary["correctness_passed"] is True
    assert summary["baseline"]["original_baseline_latency_us"] == 2391.0
    assert summary["baseline"]["parent_latency_us"] == 1590.0
    assert summary["baseline"]["parent_solution_id"] == "sol_s1_adopted"
    assert summary["baseline"]["parent_strategy_id"] == "fa_qkv_two_level_l1_reuse"
    assert summary["baseline"]["reference_latency_us"] is None
    assert summary["performance"]["mean_latency_us"] == 2396.596
    assert summary["performance"]["min_latency_us"] == 2391.7
    assert summary["performance"]["max_latency_us"] == 2402.04
    assert summary["performance"]["num_runs"] == 20
    assert summary["performance"]["speedup_vs_original_baseline"] == 0.998
    assert summary["performance"]["speedup_vs_parent"] == 1.627
    assert "No detailed failure log is needed" in log
    assert "profiler.py" not in log
    assert "Permission mismatch" not in log


def test_benchmark_failed_context_contains_only_benchmark_error():
    result = EvalResult(
        status="benchmark_failed",
        log_excerpt=(
            "[build stderr]\n"
            "build noise should not be shown\n"
            "[correctness stdout]\n"
            "PASS should not be shown\n"
            "[benchmark stderr]\n"
            "timeout after 30 seconds in bench command\n"
            "Permission mismatch\n"
        ),
    )

    summary, log = build_eval_context_for_llm(eval_result=result, task=SimpleNamespace())

    assert summary["eval_context_status"] == "performance_eval_failed"
    assert summary["diagnostic_kind"] == "performance_eval_failure"
    assert summary["has_eval_log"] is True
    assert "Performance Evaluation Failure" in log
    assert "timeout after 30 seconds" in log
    assert "Compile Error" not in log
    assert "Correctness Failure" not in log
    assert "Permission mismatch" not in log
