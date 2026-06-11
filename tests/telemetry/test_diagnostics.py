import json

from k_search.telemetry.diagnostics import diagnose_tool_protocol_failure
from k_search.telemetry.events import TelemetryEvent


def test_diagnoses_unknown_tool_from_in_memory_events():
    events = [
        TelemetryEvent(
            event_type="tool_use",
            tool_use_id="toolu_1",
            tool_name="file_path",
            context={"stage": "action", "round_index": 1, "attempt_index": 1},
        ),
        TelemetryEvent(
            event_type="tool_result",
            tool_use_id="toolu_1",
            is_error=True,
            tool_result_excerpt="<tool_use_error>Error: No such tool available: file_path</tool_use_error>",
            context={"stage": "action", "round_index": 1, "attempt_index": 1},
        ),
    ]

    diagnosis = diagnose_tool_protocol_failure(events=events)

    assert diagnosis is not None
    assert diagnosis.reason == "retryable_tool_protocol_error"
    assert diagnosis.retryable is True
    assert diagnosis.unknown_tool == "file_path"
    assert diagnosis.stage == "action"
    assert "file_path is an argument key, not a tool" in diagnosis.recovery_prompt


def test_diagnoses_unknown_tool_from_trace_jsonl(tmp_path):
    trace_path = tmp_path / "agent_trace.jsonl"
    rows = [
        {"event_type": "tool_use", "tool_use_id": "toolu_1", "tool_name": "file_path"},
        {
            "event_type": "tool_result",
            "tool_use_id": "toolu_1",
            "is_error": True,
            "tool_result_excerpt": "<tool_use_error>Error: No such tool available: file_path</tool_use_error>",
            "context": {"stage": "designer"},
        },
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    diagnosis = diagnose_tool_protocol_failure(trace_path=trace_path)

    assert diagnosis is not None
    assert diagnosis.unknown_tool == "file_path"
    assert diagnosis.stage == "designer"
