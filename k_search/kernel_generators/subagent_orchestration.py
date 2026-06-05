from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult


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
    required = ", ".join(stage.required_files) if stage.required_files else "(none)"
    parts = [
        f"Stage {active_stage_index}/{active_stage_count}: {stage.name}",
        f"Flow: {flow.name}",
        f"Use the {stage.agent} subagent for this stage.",
        "Do not invoke any other subagent during this stage.",
        stage.instruction.strip(),
        f"Required file outputs after this stage: {required}.",
        "The subagent final message must be short and contain only status, files_written, and next.",
        "Do not paste handoff files or source files into the final message.",
    ]
    if stage.include_base_prompt:
        parts.extend(["", "Base attempt context:", str(base_prompt or "").strip()])
    return "\n".join(part for part in parts if part is not None).strip()


def run_configured_subagent_flow(
    *,
    editor_client: Any,
    project_dir: str | Path,
    base_prompt: str,
    flow: SubagentFlowConfig,
    telemetry_recorder: Any | None = None,
    session: Any | None = None,
    close_session_on_exit: bool = True,
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
            prompt = render_subagent_stage_prompt(
                flow=flow,
                stage=stage,
                active_stage_index=index,
                active_stage_count=len(active_stages),
                base_prompt=base_prompt,
            )
            result = editor_client.send_prompt(
                session,
                prompt=prompt,
                telemetry_recorder=telemetry_recorder,
            )
            _require_stage_files(project_root, stage)
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
