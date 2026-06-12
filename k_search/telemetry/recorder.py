from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

from k_search.telemetry.context import (
    TelemetryArtifacts,
    TelemetryContext,
    build_attempt_dir,
    is_telemetry_enabled,
)
from k_search.telemetry.events import TelemetryEvent
from k_search.telemetry.sinks import CostJsonSink, JsonlSink, MarkdownTimelineSink, TelemetrySink
from k_search.utils.paths import get_run_logs_dir


def _record_logging_sink_failure(context: TelemetryContext | None, sink: object, exc: BaseException) -> None:
    try:
        ctx = context or TelemetryContext()
        logs_dir = get_run_logs_dir(task_name=ctx.task_name, run_id=ctx.run_id)
        path = logs_dir / "logging_errors.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        sink_path = getattr(sink, "path", None) or getattr(sink, "file_path", None)
        payload = {
            "schema_version": 1,
            "event_type": "logging_sink_failure",
            "sink": type(sink).__name__,
            "path": str(sink_path) if sink_path is not None else None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "severity": "warning",
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


class TelemetryRecorder:
    def __init__(
        self,
        *,
        context: TelemetryContext | None = None,
        sinks: Iterable[TelemetrySink] = (),
        artifacts: TelemetryArtifacts | None = None,
        enabled: bool = True,
    ) -> None:
        self.context = context or TelemetryContext()
        self.sinks = list(sinks)
        self.artifacts = artifacts or TelemetryArtifacts()
        self.enabled = bool(enabled)
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        if not self.enabled:
            return
        merged_context = self.context.to_dict()
        merged_context.update(event.context or {})
        event.context = merged_context
        self.events.append(event)
        for sink in self.sinks:
            try:
                sink.write_event(event)
            except Exception as exc:
                _record_logging_sink_failure(self.context, sink, exc)

    def close(self) -> None:
        if not self.enabled:
            return
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:
                _record_logging_sink_failure(self.context, sink, exc)


def noop_recorder() -> TelemetryRecorder:
    return TelemetryRecorder(enabled=False)


def build_file_recorder(
    *,
    context: TelemetryContext,
    prompt: str,
    root: Path | None = None,
) -> TelemetryRecorder:
    if not is_telemetry_enabled():
        return noop_recorder()
    try:
        attempt_dir = build_attempt_dir(context, root=root)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "prompt.md").write_text(str(prompt or ""), encoding="utf-8")
        artifacts = TelemetryArtifacts(
            trace_path=str(attempt_dir / "agent_trace.jsonl"),
            timeline_path=str(attempt_dir / "tool_timeline.md"),
            cost_path=str(attempt_dir / "cost.json"),
        )
        return TelemetryRecorder(
            context=context,
            artifacts=artifacts,
            sinks=[
                JsonlSink(attempt_dir / "agent_trace.jsonl"),
                MarkdownTimelineSink(attempt_dir / "tool_timeline.md"),
                CostJsonSink(attempt_dir / "cost.json"),
            ],
        )
    except Exception:
        return noop_recorder()