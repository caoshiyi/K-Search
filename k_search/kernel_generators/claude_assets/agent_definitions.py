from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


_LIST_SEP = ","


@dataclass(frozen=True)
class NativeAgentSpec:
    name: str
    description: str
    prompt: str
    tools: list[str] | None = None
    disallowedTools: list[str] | None = None
    model: str | None = None
    skills: list[str] | None = None
    maxTurns: int | None = None
    background: bool | None = None
    permissionMode: str | None = None


def should_use_programmatic_agents() -> bool:
    return os.getenv("KSEARCH_DISABLE_PROGRAMMATIC_AGENTS", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def require_programmatic_agents() -> bool:
    return os.getenv("KSEARCH_REQUIRE_PROGRAMMATIC_AGENTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _asset_root() -> Path:
    return Path(str(files("k_search.kernel_generators.claude_assets")))


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    s = str(text or "")
    if not s.startswith("---"):
        return {}, s.strip()
    lines = s.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, s.strip()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, s.strip()
    raw_meta = lines[1:end]
    body = "\n".join(lines[end + 1 :]).strip()
    meta: dict[str, str] = {}
    current_key: str | None = None
    for line in raw_meta:
        stripped = line.strip()
        if not stripped or line.lstrip().startswith("#"):
            continue
        if stripped.startswith("- ") and current_key:
            item = stripped[2:].strip().strip("'\"")
            if item:
                existing = meta.get(current_key, "")
                meta[current_key] = f"{existing}{_LIST_SEP} {item}" if existing else item
            continue
        if ":" not in line:
            current_key = None
            continue
        key, value = line.split(":", 1)
        current_key = key.strip()
        meta[current_key] = value.strip().strip("'\"")
    return meta, body


def _parse_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    items = [item.strip().strip("'\"") for item in text.split(_LIST_SEP)]
    out = [item for item in items if item]
    return out or None


def _parse_int(value: str | None) -> int | None:
    try:
        return int(str(value).strip()) if value is not None and str(value).strip() else None
    except Exception:
        return None


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def _spec_from_markdown(filename: str, text: str) -> NativeAgentSpec:
    meta, body = _split_frontmatter(text)
    fallback_name = Path(filename).stem
    name = (meta.get("name") or fallback_name).strip()
    description = (meta.get("description") or f"K-Search native subagent: {name}").strip()
    return NativeAgentSpec(
        name=name,
        description=description,
        prompt=body.strip(),
        tools=_parse_list(meta.get("tools")),
        disallowedTools=_parse_list(meta.get("disallowedTools")),
        model=(meta.get("model") or None),
        skills=_parse_list(meta.get("skills")),
        maxTurns=_parse_int(meta.get("maxTurns")),
        background=_parse_bool(meta.get("background")),
        permissionMode=(meta.get("permissionMode") or None),
    )


def load_native_agent_specs() -> dict[str, NativeAgentSpec]:
    from k_search.kernel_generators.claude_assets.materializer import NATIVE_AGENT_FILES

    root = _asset_root()
    specs: dict[str, NativeAgentSpec] = {}
    for filename in NATIVE_AGENT_FILES:
        text = (root / "agents" / filename).read_text(encoding="utf-8")
        spec = _spec_from_markdown(filename, text)
        specs[spec.name] = spec
    return specs


def _build_agent_definition_cls() -> Any | None:
    try:
        from claude_agent_sdk import AgentDefinition  # type: ignore

        return AgentDefinition
    except Exception:
        return None


def _agent_definition_from_spec(spec: NativeAgentSpec) -> Any:
    AgentDefinition = _build_agent_definition_cls()
    if AgentDefinition is None:
        raise RuntimeError("claude_agent_sdk.AgentDefinition is unavailable")
    candidate = {
        "description": spec.description,
        "prompt": spec.prompt,
        "tools": spec.tools,
        "disallowedTools": spec.disallowedTools,
        "model": spec.model,
        "skills": spec.skills,
        "maxTurns": spec.maxTurns,
        "background": spec.background,
        "permissionMode": spec.permissionMode,
    }
    if is_dataclass(AgentDefinition):
        valid = {field.name for field in fields(AgentDefinition)}
        kwargs = {key: value for key, value in candidate.items() if key in valid and value is not None}
    else:
        kwargs = {key: value for key, value in candidate.items() if value is not None}
    return AgentDefinition(**kwargs)


def load_native_agent_definitions(
    *,
    enabled_agent_names: list[str] | None = None,
) -> dict[str, Any]:
    specs = load_native_agent_specs()
    allowed = {str(name).strip() for name in enabled_agent_names or [] if str(name).strip()}
    if allowed:
        specs = {name: spec for name, spec in specs.items() if name in allowed}
    return {name: _agent_definition_from_spec(spec) for name, spec in specs.items()}
