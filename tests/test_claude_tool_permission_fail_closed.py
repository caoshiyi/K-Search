import asyncio
from pathlib import Path

from k_search.kernel_generators.claude_agent_project_editor import _make_project_tool_permission_callback


def _permission_behavior(result):
    if isinstance(result, dict):
        return result.get("behavior") or result.get("decision")
    name = type(result).__name__.lower()
    if "deny" in name:
        return "deny"
    if "allow" in name:
        return "allow"
    return getattr(result, "behavior", None)


def _permission_message(result):
    if isinstance(result, dict):
        return result.get("message", "")
    return getattr(result, "message", "")


def _permission_interrupt(result):
    if isinstance(result, dict):
        return result.get("interrupt")
    return getattr(result, "interrupt", None)


def _callback(worktree: Path):
    return _make_project_tool_permission_callback(
        project_root=worktree,
        allowed_tools={"Read", "Write", "Edit", "Glob", "Grep", "Agent", "Task", "Skill"},
        allowed_agents={"codegen", "plan", "reviewer"},
        allowed_skills={"ascendc-codegen"},
    )


def _run(callback, tool, input_data):
    return asyncio.run(callback(tool, input_data, None))


def test_allows_valid_agent(tmp_path):
    result = _run(_callback(tmp_path), "Agent", {"subagent_type": "codegen"})

    assert _permission_behavior(result) == "allow"


def test_allows_valid_task_alias(tmp_path):
    result = _run(_callback(tmp_path), "Task", {"subagent_type": "reviewer"})

    assert _permission_behavior(result) == "allow"


def test_denies_agent_with_missing_subagent_name(tmp_path):
    result = _run(_callback(tmp_path), "Agent", {"description": "do codegen"})

    assert _permission_behavior(result) == "deny"
    assert _permission_interrupt(result) is True
    assert "missing subagent name" in _permission_message(result)


def test_denies_unknown_subagent(tmp_path):
    result = _run(_callback(tmp_path), "Agent", {"subagent_type": "unknown-agent"})

    assert _permission_behavior(result) == "deny"
    assert _permission_interrupt(result) is True
    assert "unknown-agent" in _permission_message(result)


def test_parses_nested_agent_input(tmp_path):
    result = _run(_callback(tmp_path), "Agent", {"input": {"agent_name": "plan"}})

    assert _permission_behavior(result) == "allow"


def test_allows_valid_skill(tmp_path):
    result = _run(_callback(tmp_path), "Skill", {"skill": "ascendc-codegen"})

    assert _permission_behavior(result) == "allow"


def test_denies_skill_with_missing_name(tmp_path):
    result = _run(_callback(tmp_path), "Skill", {"description": "use skill"})

    assert _permission_behavior(result) == "deny"
    assert _permission_interrupt(result) is True
    assert "missing skill name" in _permission_message(result)


def test_denies_unknown_skill(tmp_path):
    result = _run(_callback(tmp_path), "Skill", {"skill": "unknown-skill"})

    assert _permission_behavior(result) == "deny"
    assert _permission_interrupt(result) is True
    assert "unknown-skill" in _permission_message(result)


def test_escape_hatch_allows_missing_agent_name(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ALLOW_UNKNOWN_AGENT_TOOL_INPUT", "1")

    result = _run(_callback(tmp_path), "Agent", {"description": "legacy sdk"})

    assert _permission_behavior(result) == "allow"
