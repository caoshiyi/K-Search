from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.kernel_generators.prompt_hygiene import check_prompt_hygiene_or_raise
from k_search.kernel_generators.worktree_context import assert_no_absolute_paths_for_llm

logger = logging.getLogger(__name__)

SUBAGENT_TOOL_NAMES = {"Agent", "Task"}
SUBAGENT_NAME_KEYS = (
    "subagent_type",
    "agent",
    "name",
    "subagent",
    "agent_name",
    "type",
)
SHARED_STAGE_MARKER_FILES = {"CODE_MAP.md"}


@dataclass(frozen=True)
class SubagentStageConfig:
    name: str
    agent: str
    instruction: str
    required_files: tuple[str, ...] = ()
    run_when_missing_files: tuple[str, ...] = ()
    include_base_prompt: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SubagentStageConfig":
        return cls(
            name=_required_str(raw, "name"),
            agent=_required_str(raw, "agent"),
            instruction=_required_str(raw, "instruction"),
            required_files=_str_tuple(raw.get("required_files", ())),
            run_when_missing_files=_str_tuple(raw.get("run_when_missing_files", ())),
            include_base_prompt=bool(raw.get("include_base_prompt", True)),
        )


@dataclass(frozen=True)
class SubagentFlowConfig:
    name: str
    description: str
    stages: tuple[SubagentStageConfig, ...]
    version: int = 1
    trigger: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SubagentFlowConfig":
        stages = tuple(SubagentStageConfig.from_dict(item) for item in raw.get("stages", ()))
        if not stages:
            raise ValueError("subagent flow config must define at least one stage")
        return cls(
            name=_required_str(raw, "name"),
            description=str(raw.get("description", "")),
            stages=stages,
            version=int(raw.get("version", 1)),
            trigger=dict(raw["trigger"]) if isinstance(raw.get("trigger"), dict) else None,
        )


@dataclass(frozen=True)
class SubagentFlowSet:
    name: str
    description: str
    default_flow: str
    flows: dict[str, SubagentFlowConfig]
    version: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SubagentFlowSet":
        if "flows" not in raw:
            flow = SubagentFlowConfig.from_dict(raw)
            return cls(
                name=flow.name,
                description=flow.description,
                default_flow=flow.name,
                flows={flow.name: flow},
                version=flow.version,
            )

        raw_flows = raw.get("flows")
        if not isinstance(raw_flows, dict) or not raw_flows:
            raise ValueError("subagent flow config must define at least one named flow")

        flows: dict[str, SubagentFlowConfig] = {}
        for key, flow_raw in raw_flows.items():
            if not isinstance(flow_raw, dict):
                raise ValueError(f"subagent flow {key!r} must be a JSON object")
            flow_payload = dict(flow_raw)
            flow_payload.setdefault("name", str(key))
            flow_payload.setdefault("version", int(raw.get("version", 1)))
            flows[str(key)] = SubagentFlowConfig.from_dict(flow_payload)

        default_flow = str(raw.get("default_flow", "")).strip() or next(iter(flows))
        if default_flow not in flows:
            raise ValueError(f"default subagent flow {default_flow!r} is not defined")

        return cls(
            name=_required_str(raw, "name"),
            description=str(raw.get("description", "")),
            default_flow=default_flow,
            flows=flows,
            version=int(raw.get("version", 1)),
        )

    def get(self, name: str | None = None) -> SubagentFlowConfig:
        key = name or self.default_flow
        try:
            return self.flows[key]
        except KeyError as exc:
            raise KeyError(f"subagent flow {key!r} is not configured") from exc


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"subagent flow config missing required field: {key}")
    return value


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    return tuple(str(item).strip() for item in items if str(item).strip())


def load_default_subagent_flow() -> SubagentFlowConfig:
    return load_subagent_flows().get()


def load_subagent_flows(path: str | Path | None = None) -> SubagentFlowSet:
    flow_path = path or os.getenv("KSEARCH_SUBAGENT_FLOW_CONFIG", "").strip()
    if flow_path:
        raw = _read_flow_json(Path(flow_path).expanduser())
    else:
        root = files("k_search.kernel_generators.claude_assets")
        raw = _read_flow_json(root / "subagent_flow.json")
    return SubagentFlowSet.from_dict(raw)


def load_subagent_flow(path: str | Path | None = None, *, flow_name: str | None = None) -> SubagentFlowConfig:
    return load_subagent_flows(path).get(flow_name)


def _read_flow_json(path: Any) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"subagent flow config must be a JSON object: {path}")
    return raw


def render_subagent_stage_prompt(
    *,
    flow: SubagentFlowConfig,
    stage: SubagentStageConfig,
    active_stage_index: int,
    active_stage_count: int,
    base_prompt: str,
) -> str:
    required_lines = (
        "\n".join(f"- {path}" for path in stage.required_files)
        if stage.required_files
        else "- (none)"
    )
    parts = [
        f"Stage {active_stage_index}/{active_stage_count}: {stage.name}",
        f"Current agent: {stage.agent}",
        "Project root contract:",
        '- Treat "." as the candidate project root.',
        '- All required reads and outputs are relative to "." unless explicitly stated otherwise.',
        '- Read and write only paths under ".".',
        "- Do not use parent directories, sibling task directories, archive directories, or historical run artifacts.",
        "- Do not use absolute paths in the Agent prompt or final message.",
        "Parent-agent dispatch contract:",
        (
            "When constructing the Agent prompt, preserve the project root contract verbatim. "
            "Do not invent a parent directory, sibling task directory, archive directory, or historical run artifact as CWD."
        ),
        "Do not introduce task-layout assumptions not present in the candidate project.",
        (
            "If the subagent prompt mentions CWD, it must describe the current candidate project directory "
            'as "." and must not name a parent directory.'
        ),
        "Current stage goal:",
        stage.instruction.strip(),
        (
            f"You MUST invoke exactly one native subagent for this stage: {stage.agent}. "
            f"Use the Agent tool with subagent_type={stage.agent!r}. "
            "Do not complete this stage in the parent agent context."
        ),
        f"Use the {stage.agent} subagent for this stage.",
        "Do not invoke any other subagent during this stage.",
        "Required outputs:",
        required_lines,
        "Required reads:",
        "- .ksearch/context/STRATEGY.md",
        "- .ksearch/context/STRATEGY_SUMMARY.md",
        "- .ksearch/context/EVAL_SUMMARY.json",
        "- .ksearch/context/EVAL_LOG.md only when EVAL_SUMMARY.json has has_eval_log=true",
        (
            f"When writing any stage handoff file, include this marker near the top if possible: "
            f"<!-- ksearch-stage: {stage.name}; ksearch-agent: {stage.agent} -->"
        ),
        "The subagent final message must be short and contain only status, files_written, and next.",
        "Do not paste handoff files or source files into the final message.",
    ]
    if stage.include_base_prompt:
        parts.extend(["", "Base attempt context:", str(base_prompt or "").strip()])
    prompt = "\n".join(part for part in parts if part is not None).strip()
    assert_no_absolute_paths_for_llm(prompt)
    return prompt


def _validate_stage_agent_is_in_flow(flow: SubagentFlowConfig, stage: SubagentStageConfig) -> None:
    allowed = {item.agent for item in flow.stages}
    if stage.agent not in allowed:
        raise RuntimeError(
            f"stage agent {stage.agent!r} is not declared in flow {flow.name!r}; "
            f"allowed={sorted(allowed)}"
        )


def run_configured_subagent_flow(
    *,
    editor_client: Any,
    project_dir: str | Path,
    base_prompt: str,
    flow: SubagentFlowConfig,
    telemetry_recorder: Any | None = None,
    session: Any | None = None,
    close_session_on_exit: bool = True,
    stage_prompt_sink: Any | None = None,
    prompt_hygiene_known_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> ClaudeProjectEditResult:
    project_root = Path(project_dir).expanduser().resolve()
    active_stages = [stage for stage in flow.stages if _should_run_stage(stage, project_root)]
    if not active_stages:
        raise RuntimeError(f"subagent flow {flow.name!r} has no active stages")

    owns_session = session is None
    if session is None:
        session = editor_client.open_session(project_dir=project_root, telemetry_recorder=telemetry_recorder)
    stage_results: list[ClaudeProjectEditResult] = []
    transcript = ""
    try:
        for index, stage in enumerate(active_stages, start=1):
            _validate_stage_agent_is_in_flow(flow, stage)
            prompt = render_subagent_stage_prompt(
                flow=flow,
                stage=stage,
                active_stage_index=index,
                active_stage_count=len(active_stages),
                base_prompt=base_prompt,
            )
            hygiene = check_prompt_hygiene_or_raise(
                prompt,
                known_paths=[project_root, *(prompt_hygiene_known_paths or [])],
            )
            if stage_prompt_sink is not None:
                stage_prompt_sink.write(index=index, stage=stage, prompt=prompt, hygiene=hygiene)
            event_start = len(getattr(telemetry_recorder, "events", []) or [])
            result = editor_client.send_prompt(
                session,
                prompt=prompt,
                telemetry_recorder=telemetry_recorder,
            )
            _validate_stage_completion(
                project_root=project_root,
                stage=stage,
                telemetry_recorder=telemetry_recorder,
                event_start=event_start,
                require_agent_tool_use=bool(getattr(editor_client, "require_agent_tool_use", False)),
            )
            stage_results.append(result)
            transcript = _merge_transcript(transcript, result.transcript)
    finally:
        close = getattr(editor_client, "close_session", None)
        if close_session_on_exit and owns_session and callable(close):
            close(session)

    final = stage_results[-1]
    return replace(
        final,
        transcript=transcript,
        prompt=str(base_prompt or ""),
        prompt_chars=len(str(base_prompt or "")),
        prompt_lines=(str(base_prompt or "").count("\n") + 1 if base_prompt else 0),
    )


def supports_configured_subagent_flow(editor_client: Any) -> bool:
    return callable(getattr(editor_client, "open_session", None)) and callable(getattr(editor_client, "send_prompt", None))


def _should_run_stage(stage: SubagentStageConfig, project_root: Path) -> bool:
    if not stage.run_when_missing_files:
        return True
    return any(not (project_root / path).is_file() for path in stage.run_when_missing_files)


def _require_stage_files(project_root: Path, stage: SubagentStageConfig) -> None:
    missing = [path for path in stage.required_files if not (project_root / path).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"subagent stage {stage.name!r} did not produce required file(s): {joined}")


def _validate_stage_completion(
    *,
    project_root: Path,
    stage: SubagentStageConfig,
    telemetry_recorder: Any | None,
    event_start: int,
    require_agent_tool_use: bool,
) -> None:
    _require_stage_files(project_root, stage)
    if require_agent_tool_use and bool(getattr(telemetry_recorder, "enabled", False)):
        _require_agent_tool_invocation(
            telemetry_recorder=telemetry_recorder,
            event_start=event_start,
            stage=stage,
        )
    _warn_on_missing_stage_markers(project_root=project_root, stage=stage)


def _warn_on_missing_stage_markers(*, project_root: Path, stage: SubagentStageConfig) -> None:
    strict = os.getenv("KSEARCH_REQUIRE_STAGE_MARKERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    missing: list[str] = []
    expected = f"ksearch-agent: {stage.agent}"
    for rel in stage.required_files:
        if rel in SHARED_STAGE_MARKER_FILES and stage.agent != "code-reader":
            continue
        path = project_root / rel
        if not path.is_file():
            continue
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:1000]
        except Exception:
            continue
        if expected not in head:
            missing.append(rel)
    if missing:
        msg = f"stage handoff marker missing: stage={stage.name!r}, agent={stage.agent!r}, files={missing}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)


def _require_agent_tool_invocation(*, telemetry_recorder: Any | None, event_start: int, stage: SubagentStageConfig) -> None:
    events = list(getattr(telemetry_recorder, "events", []) or [])[int(event_start) :]
    agent_calls = [
        event
        for event in events
        if getattr(event, "event_type", None) == "tool_use"
        and getattr(event, "tool_name", None) in SUBAGENT_TOOL_NAMES
    ]
    observed_calls = [
        {
            "tool_name": getattr(event, "tool_name", None),
            "subagent": _subagent_name_from_tool_input(getattr(event, "tool_input", None)),
            "tool_input": getattr(event, "tool_input", None),
        }
        for event in agent_calls
    ]
    matching = [
        item
        for item in observed_calls
        if _subagent_name_matches(item.get("subagent"), stage.agent)
    ]
    nonmatching = [
        item
        for item in observed_calls
        if not _subagent_name_matches(item.get("subagent"), stage.agent)
    ]
    if not matching or nonmatching:
        raise RuntimeError(
            "subagent stage invocation validation failed: "
            f"stage={stage.name!r}, expected_agent={stage.agent!r}, "
            f"matching_calls={len(matching)}, total_subagent_calls={len(agent_calls)}, "
            f"required_files={tuple(stage.required_files)!r}, observed_calls={observed_calls!r}"
        )


def _subagent_name_from_tool_input(tool_input: Any) -> str | None:
    def norm(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None
        if s.startswith("@agent-"):
            s = s[len("@agent-") :]
        if s.endswith(" (agent)"):
            s = s[: -len(" (agent)")]
        return s.strip() or None

    if not isinstance(tool_input, dict):
        return None
    for key in SUBAGENT_NAME_KEYS:
        value = norm(tool_input.get(key))
        if value:
            return value
    for nested_key in ("input", "arguments", "params"):
        nested = tool_input.get(nested_key)
        if isinstance(nested, dict):
            for key in SUBAGENT_NAME_KEYS:
                value = norm(nested.get(key))
                if value:
                    return value
    return None


def _subagent_name_matches(observed: str | None, expected: str) -> bool:
    if not observed:
        return False
    obs = str(observed).strip()
    exp = str(expected).strip()
    if obs == exp:
        return True
    if obs.endswith(":" + exp):
        return True
    if obs.endswith("/" + exp):
        return True
    return False


def _merge_transcript(current: str, new: str) -> str:
    current = str(current or "").strip()
    new = str(new or "").strip()
    if not current:
        return new
    if not new:
        return current
    if new.startswith(current):
        return new
    return f"{current}\n{new}"
