from pathlib import Path

import pytest


def test_materialize_claude_project_assets_writes_agents_and_skills(tmp_path):
    from k_search.kernel_generators.claude_assets import (
        NATIVE_AGENT_FILES,
        NATIVE_SKILL_FILES,
        materialize_claude_project_assets,
    )

    result = materialize_claude_project_assets(tmp_path)

    assert result.claude_dir == tmp_path / ".claude"
    assert sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in result.written_paths) == sorted(
        [f".claude/agents/{name}" for name in NATIVE_AGENT_FILES]
        + [f".claude/skills/{path}" for path in NATIVE_SKILL_FILES]
    )
    assert (tmp_path / ".claude" / "agents" / "code-reader.md").exists()
    assert (tmp_path / ".claude" / "agents" / "plan.md").exists()
    assert (tmp_path / ".claude" / "agents" / "codegen.md").exists()
    assert (tmp_path / ".claude" / "agents" / "reviewer.md").exists()
    assert (tmp_path / ".claude" / "agents" / "bug-fixer.md").exists()
    assert (tmp_path / ".claude" / "skills" / "ascendc-codegen" / "SKILL.md").exists()
    assert (tmp_path / ".claude" / "skills" / "ascendc-api-reference" / "SKILL.md").exists()


def test_materialize_claude_project_assets_copies_reference_dirs_not_symlinks(tmp_path):
    from k_search.kernel_generators.claude_assets import materialize_claude_project_assets

    result = materialize_claude_project_assets(tmp_path)

    assert result.linked_paths == []
    reference_dirs = [
        tmp_path / ".claude" / "references" / "ascendc-design",
        tmp_path / ".claude" / "references" / "attention-patterns",
        tmp_path / ".claude" / "references" / "curation-format",
        tmp_path / ".claude" / "skills" / "ascendc-dumptensor" / "references",
        tmp_path / ".claude" / "skills" / "ascendc-fa-detailed-design" / "references",
    ]
    for reference_dir in reference_dirs:
        assert reference_dir.is_dir()
        assert not reference_dir.is_symlink()


def test_materializer_can_replace_unmanaged_candidate_claude_assets_without_overwriting_source(tmp_path):
    from k_search.kernel_generators.claude_assets import materialize_claude_project_assets

    target = tmp_path / ".claude" / "agents" / "code-reader.md"
    target.parent.mkdir(parents=True)
    target.write_text("user-owned agent file\n", encoding="utf-8")

    result = materialize_claude_project_assets(tmp_path)

    assert result.claude_dir == tmp_path / ".claude"
    assert target.read_text(encoding="utf-8").startswith("<!-- K-Search managed Claude asset -->")


def test_materializer_fails_fast_when_dev_knowledge_references_missing(tmp_path, monkeypatch):
    from k_search.kernel_generators.claude_assets import materialize_claude_project_assets

    monkeypatch.delenv("KSEARCH_ALLOW_MISSING_DEV_KNOWLEDGE", raising=False)

    with pytest.raises(RuntimeError, match="ascendc-dev-knowledge/references"):
        materialize_claude_project_assets(tmp_path)


def test_materializer_can_refresh_managed_files(tmp_path):
    from k_search.kernel_generators.claude_assets import materialize_claude_project_assets

    first = materialize_claude_project_assets(tmp_path)
    managed_file = tmp_path / ".claude" / "agents" / "code-reader.md"
    managed_file.write_text("<!-- K-Search managed Claude asset -->\ncorrupted\n", encoding="utf-8")

    second = materialize_claude_project_assets(tmp_path)

    assert first.claude_dir == second.claude_dir
    text = managed_file.read_text(encoding="utf-8")
    assert "name: code-reader" in text
    assert "corrupted" not in text


def test_asset_files_contain_required_handoff_contracts(tmp_path):
    from k_search.kernel_generators.claude_assets import materialize_claude_project_assets

    materialize_claude_project_assets(tmp_path)

    code_reader = (tmp_path / ".claude" / "agents" / "code-reader.md").read_text(encoding="utf-8")
    planner = (tmp_path / ".claude" / "agents" / "plan.md").read_text(encoding="utf-8")
    codegen = (tmp_path / ".claude" / "agents" / "codegen.md").read_text(encoding="utf-8")
    reviewer = (tmp_path / ".claude" / "agents" / "reviewer.md").read_text(encoding="utf-8")
    bug_fixer = (tmp_path / ".claude" / "agents" / "bug-fixer.md").read_text(encoding="utf-8")

    assert "CODE_MAP.md" in code_reader
    assert "IMPLEMENTATION_PLAN.md" in planner
    assert "IMPLEMENTATION_PLAN.md" in codegen
    assert "IMPLEMENTATION_PLAN.md if present" in reviewer
    assert "REVIEW_NOTES.md" in reviewer
    assert "python evaluation" in bug_fixer.lower()
    assert "tools: Read, Grep, Glob, Edit, Write" in bug_fixer
    assert "next: reviewer" in bug_fixer
    for text in (code_reader, planner, codegen, reviewer, bug_fixer):
        assert "status" in text
        assert "files_written" in text
        assert "next" in text
        assert "Do not paste" in text
