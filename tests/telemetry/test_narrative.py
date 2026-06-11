import json

from k_search.telemetry.narrative import RunNarrativeLogger
from k_search.tasks.task_base import EvalResult


def _read_events(run_dir):
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_run_start_writes_meta_and_summary(tmp_path):
    nar = RunNarrativeLogger(tmp_path, meta={"run_id": "r1", "task_name": "mqa"})
    nar.run_start()

    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "r1"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "K-Search Run Summary" in summary
    assert "mqa" in summary
    events = _read_events(tmp_path)
    assert events[0]["type"] == "run_start"


def test_full_event_sequence(tmp_path):
    nar = RunNarrativeLogger(tmp_path)
    nar.run_start({"run_id": "r1", "task_name": "mqa"})
    nar.world_model_init(node_count=4, actions=[{"node_id": "n1", "title": "vectorize", "difficulty_1_to_5": 3, "score_0_to_1": 0.7}])
    nar.action_selected(node_id="n1", title="vectorize", difficulty=3, score=0.7, round_num=1)
    nar.llm_codegen(
        round_num=1,
        attempt=1,
        mode="agentic/action",
        changed_paths=["kernel.cpp", "tiling.cpp"],
        diff="--- a\n+++ b\n",
        detail_paths=["llm/world_model/round_0001/x.md"],
    )
    ev = EvalResult(status="passed", latency_ms=0.987, reference_latency_ms=1.234, mean_vs_baseline_factor=1.25, metrics={"score": 1.25, "score_name": "vs_baseline"})
    nar.eval_result(round_num=1, eval_result=ev)
    nar.world_model_update(kind="attach+refine", round_num=1, detail="attach n1")
    nar.run_end(best_round=1, latency_ms=0.987, vs_baseline=1.25, total_rounds=1)

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "世界模型初始化" in summary
    assert "vectorize" in summary
    assert "kernel.cpp, tiling.cpp" in summary
    assert "编译: ✅" in summary
    assert "1.250x" in summary  # vs_baseline
    assert "世界模型更新" in summary
    assert "Run 结束" in summary

    types = [e["type"] for e in _read_events(tmp_path)]
    assert types == [
        "run_start",
        "world_model_init",
        "action_selected",
        "llm_codegen",
        "eval_result",
        "world_model_update",
        "run_end",
    ]
    assert _read_events(tmp_path)[1]["actions"][0]["node_id"] == "n1"


def test_blocked_actions_records_summary_and_event(tmp_path):
    nar = RunNarrativeLogger(tmp_path)

    nar.blocked_actions(
        round_num=3,
        actions=[
            {
                "node_id": "s2",
                "strategy_id": "two_level_l1",
                "reason": "missing_prerequisites",
                "missing": ["s1"],
            }
        ],
        no_executable=True,
    )

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "阻塞 action" in summary
    assert "missing_prerequisites" in summary
    assert "s1" in summary
    event = _read_events(tmp_path)[0]
    assert event["type"] == "blocked_actions"
    assert event["round"] == 3
    assert event["no_executable"] is True
    assert event["actions"][0]["node_id"] == "s2"


def test_eval_failure_records_compile_log(tmp_path):
    nar = RunNarrativeLogger(tmp_path)
    ev = EvalResult(status="compile_failed", log_excerpt="error: undefined symbol foo")
    nar.eval_result(round_num=2, eval_result=ev)

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "编译: ❌" in summary
    assert "编译日志" in summary
    assert "undefined symbol foo" in summary


def test_excerpt_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_SUMMARY_EXCERPT_CHARS", "20")
    nar = RunNarrativeLogger(tmp_path)
    nar.llm_codegen(round_num=1, prompt="P" * 500, response="R" * 500)

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "...<truncated>..." in summary
    # The full 500-char block must not be present.
    assert "P" * 500 not in summary


def test_logger_never_raises_on_bad_dir(tmp_path):
    # Point at a path whose parent is a file -> mkdir fails, but no exception escapes.
    bad_parent = tmp_path / "afile"
    bad_parent.write_text("x", encoding="utf-8")
    nar = RunNarrativeLogger(bad_parent / "sub")
    nar.run_start({"run_id": "r"})
    nar.eval_result(round_num=1, eval_result=EvalResult(status="passed", latency_ms=1.0))
    # No assertion needed: the test passes if nothing raised.


def test_run_failure_updates_meta_summary_and_events(tmp_path):
    nar = RunNarrativeLogger(tmp_path, meta={"run_id": "r1", "task_name": "flash_attention", "status": "running"})
    nar.run_start()

    nar.run_failure(
        reason="retryable_tool_protocol_error",
        error_type="UnknownToolError",
        error_message="No such tool available: file_path",
        retryable=True,
        stage="action",
        detail="Recovery: file_path is an argument key, not a tool.",
    )

    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert meta["failed_reason"] == "retryable_tool_protocol_error"
    assert meta["last_stage"] == "action"
    assert meta["retryable"] is True
    assert meta["last_error"]["type"] == "UnknownToolError"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Run 失败" in summary
    assert "retryable_tool_protocol_error" in summary
    assert "file_path is an argument key" in summary
    events = _read_events(tmp_path)
    assert events[-1]["type"] == "run_failure"
    assert events[-1]["retryable"] is True
