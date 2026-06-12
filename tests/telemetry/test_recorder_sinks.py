import json
from pathlib import Path

from k_search.telemetry.context import (
    TelemetryContext,
    build_attempt_dir,
    default_run_id,
    is_telemetry_enabled,
    safe_path_component,
    telemetry_root,
)
from k_search.telemetry.events import TelemetryEvent
from k_search.telemetry.recorder import (
    TelemetryRecorder,
    build_file_recorder,
    noop_recorder,
)
from k_search.telemetry.sinks import CostJsonSink, JsonlSink, MarkdownTimelineSink


def test_build_attempt_dir_uses_sanitized_attempt_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.setenv("KSEARCH_RUN_ID", "run:alpha")
    context = TelemetryContext(
        task_name="task/name",
        round_index=7,
        attempt_index=2,
        action_node_id="n/12",
    )

    path = build_attempt_dir(context)

    assert path == tmp_path / "task_name" / "run_alpha" / "attempts" / "r0007_a02_n_12"


def test_telemetry_root_defaults_to_unified_logs_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_TELEMETRY_DIR", raising=False)
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert telemetry_root() == tmp_path / ".ksearch"


def test_build_attempt_dir_defaults_under_run_logs(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_TELEMETRY_DIR", raising=False)
    monkeypatch.setenv("KSEARCH_ARTIFACTS_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("KSEARCH_TASK_ID", "task:alpha")
    monkeypatch.setenv("KSEARCH_RUN_ID", "run:alpha")
    context = TelemetryContext(
        task_name="task/name",
        round_index=7,
        attempt_index=2,
        action_node_id="n/12",
    )

    path = build_attempt_dir(context)

    assert path == (
        tmp_path
        / "out"
        / "task_name"
        / "task_alpha"
        / "runs"
        / "run_alpha"
        / "attempts"
        / "r0007_a02_n_12"
    )


def test_is_telemetry_enabled_honors_falsey_env(monkeypatch):
    monkeypatch.setenv("KSEARCH_TELEMETRY", "off")

    assert is_telemetry_enabled() is False


def test_event_to_dict_omits_none_and_serializes_context():
    event = TelemetryEvent(
        event_type="tool_use",
        context={"round_index": 1},
        tool_name="Read",
        tool_input={"file_path": "kernel/foo.h"},
        total_cost_usd=None,
    )

    payload = event.to_dict()

    assert payload["event_type"] == "tool_use"
    assert payload["context"] == {"round_index": 1}
    assert payload["tool_name"] == "Read"
    assert payload["tool_input"] == {"file_path": "kernel/foo.h"}
    assert "total_cost_usd" not in payload
    json.dumps(payload)


def test_is_telemetry_enabled_defaults_to_true(monkeypatch):
    monkeypatch.delenv("KSEARCH_TELEMETRY", raising=False)

    assert is_telemetry_enabled() is True


def test_is_telemetry_enabled_truthy_values(monkeypatch):
    for value in ("1", "true", "on", "yes"):
        monkeypatch.setenv("KSEARCH_TELEMETRY", value)
        assert is_telemetry_enabled() is True


def test_default_run_id_prefers_ksearch_run_id(monkeypatch):
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-1")
    monkeypatch.setenv("KSEARCH_RUN_START", "run-2")

    assert default_run_id() == "run-1"


def test_default_run_id_falls_back_to_timestamp(monkeypatch):
    monkeypatch.delenv("KSEARCH_RUN_ID", raising=False)
    monkeypatch.delenv("KSEARCH_RUN_START", raising=False)

    result = default_run_id()
    assert len(result) == 15  # YYYYMMDD_HHMMSS
    assert "_" in result


def test_safe_path_component_truncates_long_values():
    long_value = "a" * 200
    result = safe_path_component(long_value, default="x")
    assert len(result) == 96


def test_safe_path_component_uses_default_for_empty():
    assert safe_path_component("", default="fallback") == "fallback"
    assert safe_path_component(None, default="fallback") == "fallback"


def test_build_attempt_dir_with_none_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("KSEARCH_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.setenv("KSEARCH_RUN_ID", "run1")

    context = TelemetryContext(task_name="task")

    path = build_attempt_dir(context)

    assert path.name == "r0000_a00_action_unknown"
    assert "action_unknown" in str(path)


def test_telemetry_context_to_dict_omits_none():
    ctx = TelemetryContext(task_name="x", round_index=3)

    payload = ctx.to_dict()

    assert payload["task_name"] == "x"
    assert payload["round_index"] == 3
    assert "definition" not in payload
    assert "extra" not in payload


def test_jsonl_sink_writes_one_event_per_line(tmp_path):
    path = tmp_path / "agent_trace.jsonl"
    sink = JsonlSink(path)
    sink.write_event(TelemetryEvent(event_type="llm_start", model_name="claude"))
    sink.write_event(
        TelemetryEvent(
            event_type="tool_use", tool_name="Glob", tool_input={"pattern": "**/*.h"}
        )
    )
    sink.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == ["llm_start", "tool_use"]
    assert rows[1]["tool_name"] == "Glob"


def test_markdown_timeline_sink_formats_tool_events(tmp_path):
    path = tmp_path / "tool_timeline.md"
    sink = MarkdownTimelineSink(path)
    sink.write_event(
        TelemetryEvent(
            event_type="llm_start", provider="claude-agent", model_name="claude"
        )
    )
    sink.write_event(
        TelemetryEvent(
            event_type="tool_use",
            tool_name="Read",
            tool_input={"file_path": "kernel/foo.h"},
        )
    )
    sink.write_event(
        TelemetryEvent(
            event_type="tool_result", tool_name="Read", tool_result_excerpt="alpha"
        )
    )
    sink.close()

    text = path.read_text(encoding="utf-8")
    assert "# Claude Agent Timeline" in text
    assert "tool_use: Read" in text
    assert "kernel/foo.h" in text
    assert "tool_result" in text


def test_cost_sink_writes_latest_result_on_close(tmp_path):
    path = tmp_path / "cost.json"
    sink = CostJsonSink(path)
    sink.write_event(TelemetryEvent(event_type="llm_start", model_name="claude"))
    sink.write_event(
        TelemetryEvent(
            event_type="llm_result",
            provider="claude-agent",
            model_name="claude",
            session_id="sess-1",
            total_cost_usd=0.125,
            duration_ms=1000,
            duration_api_ms=800,
            num_turns=3,
            usage={"input_tokens": 10},
            model_usage={"claude": {"output_tokens": 5}},
        )
    )
    sink.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["session_id"] == "sess-1"
    assert payload["summary"]["total_cost_usd"] == 0.125
    assert payload["usage"] == {"input_tokens": 10}


def test_recorder_merges_context_and_swallows_sink_errors(tmp_path):
    class BrokenSink:
        def write_event(self, event):
            raise RuntimeError("disk unhappy")

        def close(self):
            raise RuntimeError("close unhappy")

    path = tmp_path / "trace.jsonl"
    context = TelemetryContext(task_name="x", round_index=4)
    recorder = TelemetryRecorder(context=context, sinks=[BrokenSink(), JsonlSink(path)])

    recorder.emit(TelemetryEvent(event_type="tool_use", tool_name="Grep"))
    recorder.close()

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["context"]["task_name"] == "x"
    assert row["context"]["round_index"] == 4
    assert row["tool_name"] == "Grep"


def test_build_file_recorder_creates_prompt_and_artifact_paths(tmp_path):
    context = TelemetryContext(
        task_name="task", run_id="run", round_index=1, attempt_index=1
    )

    recorder = build_file_recorder(
        context=context, prompt="hello prompt", root=tmp_path
    )
    recorder.close()

    assert recorder.artifacts.trace_path is not None
    assert recorder.artifacts.timeline_path is not None
    assert recorder.artifacts.cost_path is not None
    attempt_dir = Path(recorder.artifacts.trace_path).parent
    assert (attempt_dir / "prompt.md").read_text(encoding="utf-8") == "hello prompt"


def test_noop_recorder_has_empty_artifacts():
    recorder = noop_recorder()

    recorder.emit(TelemetryEvent(event_type="llm_start"))
    recorder.close()

    assert recorder.artifacts.trace_path is None
    assert recorder.events == []
