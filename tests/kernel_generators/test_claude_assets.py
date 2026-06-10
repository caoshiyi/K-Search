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
    assert (tmp_path / ".claude" / "agents" / "designer.md").exists()
    assert not (tmp_path / ".claude" / "agents" / "plan.md").exists()
    assert (tmp_path / ".claude" / "agents" / "codegen.md").exists()
    assert (tmp_path / ".claude" / "agents" / "reviewer.md").exists()
    assert (tmp_path / ".claude" / "agents" / "bug-fixer.md").exists()
    assert (tmp_path / ".claude" / "agents" / "improvement-assessor.md").exists()
    assert (tmp_path / ".claude" / "agents" / "knowledge-curator.md").exists()
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
        tmp_path / ".claude" / "references" / "known-pitfalls",
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
    from k_search.kernel_generators.claude_assets import materializer

    monkeypatch.delenv("KSEARCH_ALLOW_MISSING_DEV_KNOWLEDGE", raising=False)
    fake_asset_root = tmp_path / "fake-assets"
    fake_asset_root.mkdir()
    monkeypatch.setattr(materializer, "_asset_root", lambda: fake_asset_root)
    monkeypatch.setattr(materializer, "NATIVE_AGENT_FILES", [])
    monkeypatch.setattr(materializer, "NATIVE_SKILL_FILES", [])
    monkeypatch.setattr(materializer, "NATIVE_SKILL_REFERENCE_DIRS", ["ascendc-dev-knowledge"])
    monkeypatch.setattr(materializer, "NATIVE_REFERENCE_DIRS", [])

    with pytest.raises(RuntimeError, match="ascendc-dev-knowledge/references"):
        materializer.materialize_claude_project_assets(tmp_path)


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


def test_materializer_removes_legacy_managed_plan_agent(tmp_path):
    from k_search.kernel_generators.claude_assets import (
        CLAUDE_ASSET_MANAGED_MARKER,
        materialize_claude_project_assets,
    )

    legacy_plan = tmp_path / ".claude" / "agents" / "plan.md"
    legacy_plan.parent.mkdir(parents=True)
    legacy_plan.write_text(f"{CLAUDE_ASSET_MANAGED_MARKER}\nname: plan\n", encoding="utf-8")

    materialize_claude_project_assets(tmp_path)

    assert not legacy_plan.exists()
    assert (tmp_path / ".claude" / "agents" / "designer.md").exists()


def test_asset_files_contain_required_handoff_contracts(tmp_path):
    from k_search.kernel_generators.claude_assets import NATIVE_HANDOFF_FILES, materialize_claude_project_assets

    materialize_claude_project_assets(tmp_path)

    code_reader = (tmp_path / ".claude" / "agents" / "code-reader.md").read_text(encoding="utf-8")
    designer = (tmp_path / ".claude" / "agents" / "designer.md").read_text(encoding="utf-8")
    codegen = (tmp_path / ".claude" / "agents" / "codegen.md").read_text(encoding="utf-8")
    reviewer = (tmp_path / ".claude" / "agents" / "reviewer.md").read_text(encoding="utf-8")
    bug_fixer = (tmp_path / ".claude" / "agents" / "bug-fixer.md").read_text(encoding="utf-8")
    improvement_assessor = (tmp_path / ".claude" / "agents" / "improvement-assessor.md").read_text(encoding="utf-8")
    attention_checklist = (
        tmp_path / ".claude" / "references" / "ascendc-design" / "attention-checklist.md"
    ).read_text(encoding="utf-8")
    attention_principles = (
        tmp_path / ".claude" / "references" / "ascendc-design" / "attention-design-principles.md"
    ).read_text(encoding="utf-8")
    known_pitfalls_index = (
        tmp_path / ".claude" / "references" / "known-pitfalls" / "README.md"
    ).read_text(encoding="utf-8")
    kp002 = (
        tmp_path
        / ".claude"
        / "references"
        / "known-pitfalls"
        / "KP-002-l1-single-buffer-reuse-reverse-sync.md"
    ).read_text(encoding="utf-8")

    old_design_name = "IMPLEMENTATION_" + "PLAN.md"

    assert "ASCENDC_DESIGN.md" in NATIVE_HANDOFF_FILES
    assert "IMPLEMENTATION_EXECUTION_PLAN.md" in NATIVE_HANDOFF_FILES
    assert "IMPLEMENTATION_HANDOFF.md" in NATIVE_HANDOFF_FILES
    assert "IMPLEMENTATION_DEVIATIONS.md" in NATIVE_HANDOFF_FILES
    assert "IMPROVEMENT_ASSESSMENT.md" in NATIVE_HANDOFF_FILES
    assert old_design_name not in NATIVE_HANDOFF_FILES

    assert "CODE_MAP.md" in code_reader
    assert "next: designer" in code_reader

    assert "name: designer" in designer
    assert "ASCENDC_DESIGN.md" in designer
    assert old_design_name not in designer
    assert "current_task/artifacts/ascendc_design.md" not in designer
    assert "tools: Read, Grep, Glob, Write" in designer
    assert "Do not run Bash" in designer
    assert "Do not edit source files" in designer
    assert "ascendc-hardware" in designer
    assert "ascendc-sync-guide" in designer
    assert "ascendc-dev-knowledge" in designer
    assert "参考资料发现与适用性判定" in designer
    assert "领域专项参考" in designer
    assert "不得硬编码某个算子目录为必读项" in designer
    assert "读取 `current_task/design/tile_level/` 下的 TileLang kernel 实现" not in designer
    assert "读取 `flash_attention/kernel/` 下的 AscendC kernel 代码" not in designer
    assert "attention-design-template.md" in designer
    assert "attention-checklist.md" in designer
    assert "WorkspaceQueue" in designer
    assert "HardEvent" in designer
    assert "KP-002" in designer
    assert "basic_case" in designer
    assert "300-500" in designer
    assert "300-500" in attention_checklist
    assert "300-500" in attention_principles
    assert "400-500" not in attention_checklist
    assert "400-500" not in attention_principles
    assert "flash_attention/kernel/" not in attention_principles
    assert "TileLang kernel 已读（`current_task/design/tile_level/`）" not in attention_checklist
    assert "flash_attention AscendC kernel 已读（`flash_attention/kernel/`）" not in attention_checklist
    assert "相关 tile-level / TileLang 参考已判定" in attention_checklist
    assert "相关 FA AscendC baseline 已判定" in attention_checklist
    assert "MTE1_MTE2" in attention_checklist
    assert "KP-002" in attention_checklist
    assert "MTE1_MTE2" in attention_principles
    assert "known-pitfalls/KP-002" in attention_principles
    assert "KP-002" in known_pitfalls_index
    assert "单缓冲 L1 复用必须补齐 MTE1→MTE2 反向同步" in kp002
    assert "MTE1_MTE2" in kp002
    assert "TileLang" in designer
    assert "next: codegen" in designer

    assert "ASCENDC_DESIGN.md" in codegen
    assert "KP-002" in codegen
    assert "L1 Buffer Lifecycle Table" in codegen
    assert "required synchronization/lifecycle guard" in codegen
    assert "CODE_MAP.md is an index, not evidence" in codegen
    assert "Never edit code based only on CODE_MAP.md summaries" in codegen
    assert "IMPLEMENTATION_EXECUTION_PLAN.md" in codegen
    assert "IMPLEMENTATION_HANDOFF.md" in codegen
    assert "IMPLEMENTATION_DEVIATIONS.md" in codegen

    assert "ASCENDC_DESIGN.md" in reviewer
    assert "IMPLEMENTATION_EXECUTION_PLAN.md" in reviewer
    assert "IMPLEMENTATION_HANDOFF.md" in reviewer
    assert "IMPLEMENTATION_DEVIATIONS.md" in reviewer
    assert "eval_failure_repair" in reviewer
    assert "do not fail solely" in reviewer
    assert "handoff files are absent" in reviewer
    assert "KP-002" in reviewer
    assert "eval_ready: false" in reviewer
    assert "naked L1" in reviewer
    assert "REVIEW_NOTES.md" in reviewer
    assert "python evaluation" in bug_fixer.lower()
    assert "KP-002" in bug_fixer
    assert "tools: Read, Grep, Glob, Edit, Write" in bug_fixer
    assert "debug_packet.json" in bug_fixer
    assert "ASCENDC_DESIGN.md, IMPLEMENTATION_EXECUTION_PLAN.md, IMPLEMENTATION_HANDOFF.md" in bug_fixer
    assert "next: reviewer" in bug_fixer
    assert "tools: Read, Grep, Glob, Write" in improvement_assessor
    assert "IMPROVEMENT_ASSESSMENT.md" in improvement_assessor
    assert "Referenced strategy" in improvement_assessor
    assert "ASCENDC_DESIGN.md" in improvement_assessor
    assert "implementation deviation" in improvement_assessor.lower()
    assert "Do not edit source files" in improvement_assessor
    assert "next: codegen" in improvement_assessor
    for text in (code_reader, designer, codegen, reviewer, bug_fixer, improvement_assessor):
        assert "status" in text
        assert "files_written" in text
        assert "next" in text
        assert "Do not paste" in text
