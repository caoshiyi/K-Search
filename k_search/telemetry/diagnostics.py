from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UNKNOWN_TOOL_PATTERN = re.compile(r"No such tool available:\s*([A-Za-z_][A-Za-z0-9_-]*)")


@dataclass(frozen=True)
class ToolProtocolDiagnosis:
    reason: str
    retryable: bool
    unknown_tool: str
    stage: str | None = None
    tool_use_id: str | None = None
    message: str = ""

    @property
    def recovery_prompt(self) -> str:
        available = "Read/Grep/Glob/Edit/Write, Skill, and Agent"
        if self.unknown_tool == "file_path":
            return (
                "The previous attempt failed because file_path is an argument key, not a tool. "
                f"Available tools are {available}. Continue from the last successful step; "
                "use file_path only inside Read/Write/Edit tool inputs, and report status as plain text "
                "when no file read or write is needed."
            )
        return (
            f"The previous attempt failed because {self.unknown_tool} is not an available tool. "
            f"Available tools are {available}. Continue from the last successful step and use only those tool names."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "retryable": self.retryable,
            "unknown_tool": self.unknown_tool,
            "stage": self.stage,
            "tool_use_id": self.tool_use_id,
            "message": self.message,
            "recovery_prompt": self.recovery_prompt,
        }


def diagnose_tool_protocol_failure(
    *,
    events: Iterable[Any] | None = None,
    trace_path: str | Path | None = None,
) -> ToolProtocolDiagnosis | None:
    """Detect retryable Claude tool-protocol mistakes from telemetry.

    The common pattern is a model emitting an input field name such as
    ``file_path`` as a tool. Claude returns a tool_result error with
    "No such tool available: ...". This is an agent protocol failure, not an
    operator compile/eval failure.
    """

    rows = list(_event_rows(events=events, trace_path=trace_path))
    tool_use_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("event_type") or "") == "tool_use" and row.get("tool_use_id"):
            tool_use_by_id[str(row["tool_use_id"])] = row

    for row in rows:
        if str(row.get("event_type") or "") != "tool_result":
            continue
        if row.get("is_error") is not True:
            continue
        excerpt = str(row.get("tool_result_excerpt") or row.get("error_message") or "")
        match = UNKNOWN_TOOL_PATTERN.search(excerpt)
        if not match:
            continue
        unknown_tool = match.group(1)
        tool_use_id = str(row.get("tool_use_id") or "") or None
        tool_use = tool_use_by_id.get(tool_use_id or "", {})
        stage = _stage_from_row(row) or _stage_from_row(tool_use)
        return ToolProtocolDiagnosis(
            reason="retryable_tool_protocol_error",
            retryable=True,
            unknown_tool=unknown_tool,
            stage=stage,
            tool_use_id=tool_use_id,
            message=excerpt,
        )
    return None


def _event_rows(
    *,
    events: Iterable[Any] | None,
    trace_path: str | Path | None,
) -> Iterable[dict[str, Any]]:
    if events is not None:
        for event in events:
            if isinstance(event, dict):
                yield event
            elif hasattr(event, "to_dict"):
                payload = event.to_dict()
                if isinstance(payload, dict):
                    yield payload
            else:
                payload = {
                    key: getattr(event, key)
                    for key in (
                        "event_type",
                        "tool_use_id",
                        "tool_name",
                        "tool_result_excerpt",
                        "error_message",
                        "is_error",
                        "context",
                    )
                    if hasattr(event, key)
                }
                if payload:
                    yield payload

    if trace_path is None:
        return
    path = Path(trace_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


def _stage_from_row(row: dict[str, Any]) -> str | None:
    context = row.get("context")
    if isinstance(context, dict):
        value = context.get("stage")
        if value is not None and str(value).strip():
            return str(value).strip()
    value = row.get("stage")
    if value is not None and str(value).strip():
        return str(value).strip()
    return None
