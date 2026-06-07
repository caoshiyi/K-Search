import json
from pathlib import Path

import pytest

from k_search.kernel_generators.worktree_context import (
    assert_no_absolute_paths_for_llm,
    materialize_worktree_context,
    to_worktree_relative_path,
)


def test_context_files_are_materialized_with_relative_paths(tmp_path):
    ctx = materialize_worktree_context(
        project_dir=tmp_path,
        strategy_markdown="# Strategy\n\nFull body.",
        strategy_summary="Short bounded summary.",
        eval_summary={"schema_version": 1, "eval_context_status": "no_prior_eval"},
        eval_log="# Evaluation Log\n\nNo prior eval.",
    )

    assert ctx.strategy_md == ".ksearch/context/STRATEGY.md"
    assert ctx.strategy_summary_md == ".ksearch/context/STRATEGY_SUMMARY.md"
    assert ctx.eval_summary_json == ".ksearch/context/EVAL_SUMMARY.json"
    assert ctx.eval_log_md == ".ksearch/context/EVAL_LOG.md"
    assert ctx.manifest_json == ".ksearch/context/CONTEXT_MANIFEST.json"
    for rel in ctx.__dict__.values():
        assert not Path(rel).is_absolute()
        assert ".." not in Path(rel).parts
        assert (tmp_path / rel).is_file()

    manifest = json.loads((tmp_path / ctx.manifest_json).read_text(encoding="utf-8"))
    assert manifest["context_root"] == ".ksearch/context"
    assert manifest["strategy_md"] == ctx.strategy_md


def test_context_manifest_records_strategy_dependency_audit(tmp_path):
    ctx = materialize_worktree_context(
        project_dir=tmp_path,
        strategy_markdown="# Strategy\n\nFull body.",
        strategy_summary="Short bounded summary.",
        eval_summary={"schema_version": 1, "eval_context_status": "no_prior_eval"},
        eval_log="# Evaluation Log\n\nNo prior eval.",
        strategy_context={
            "strategy_id": "fa_multibuffer_soft_pipeline",
            "requires": ["fa_qkv_two_level_l1_reuse"],
            "dependencies_satisfied": True,
            "parent_strategy_lineage": ["fa_qkv_two_level_l1_reuse"],
            "parent_solution_id": "round_0001_attempt_0001",
        },
    )

    manifest = json.loads((tmp_path / ctx.manifest_json).read_text(encoding="utf-8"))

    assert manifest["strategy"] == {
        "strategy_id": "fa_multibuffer_soft_pipeline",
        "requires": ["fa_qkv_two_level_l1_reuse"],
        "dependencies_satisfied": True,
        "parent_strategy_lineage": ["fa_qkv_two_level_l1_reuse"],
        "parent_solution_id": "round_0001_attempt_0001",
    }


def test_to_worktree_relative_path_rejects_escaped_paths(tmp_path):
    inside = tmp_path / ".ksearch" / "context" / "STRATEGY.md"
    inside.parent.mkdir(parents=True)
    inside.write_text("x", encoding="utf-8")

    assert to_worktree_relative_path(tmp_path, inside) == ".ksearch/context/STRATEGY.md"

    with pytest.raises(ValueError, match="escaped project root"):
        to_worktree_relative_path(tmp_path, tmp_path.parent / "outside.md")


def test_llm_text_absolute_path_assertion_rejects_host_paths():
    assert_no_absolute_paths_for_llm("Read .ksearch/context/STRATEGY.md")

    with pytest.raises(AssertionError, match="/tmp/"):
        assert_no_absolute_paths_for_llm("bad /tmp/ksearch_worktrees/project")
