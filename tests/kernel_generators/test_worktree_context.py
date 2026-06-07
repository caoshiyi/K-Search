import hashlib
import json
from pathlib import Path

import pytest

from k_search.kernel_generators.worktree_context import (
    assert_no_absolute_paths_for_llm,
    materialize_strategy_context_from_catalog_entry,
    materialize_worktree_context,
    to_worktree_relative_path,
)
from k_search.kernel_generators.strategy_injection import StrategyCatalogEntry


def test_context_files_are_materialized_with_relative_paths(tmp_path):
    strategy_source = tmp_path / "source_strategy.md"
    strategy_source.write_text("# Strategy\n\nFull body.\n", encoding="utf-8")

    ctx = materialize_worktree_context(
        project_dir=tmp_path,
        canonical_strategy_markdown_path=strategy_source,
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
    assert (tmp_path / ctx.strategy_md).read_bytes() == strategy_source.read_bytes()
    assert manifest["strategy_source"]["source_sha256"] == hashlib.sha256(
        strategy_source.read_bytes()
    ).hexdigest()
    assert (
        manifest["strategy_source"]["materialized_sha256"]
        == manifest["strategy_source"]["source_sha256"]
    )
    assert manifest["strategy_source"]["source_chars"] == len(
        strategy_source.read_text(encoding="utf-8")
    )
    assert (
        manifest["strategy_source"]["materialized_chars"]
        == manifest["strategy_source"]["source_chars"]
    )


def test_context_manifest_records_strategy_dependency_audit(tmp_path):
    strategy_source = tmp_path / "source_strategy.md"
    strategy_source.write_text("# Strategy\n\nFull body.\n", encoding="utf-8")

    ctx = materialize_worktree_context(
        project_dir=tmp_path,
        canonical_strategy_markdown_path=strategy_source,
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


def test_context_can_be_materialized_from_strategy_catalog_entry(tmp_path):
    strategy_source = tmp_path / "source_strategy.md"
    strategy_source.write_text("# Catalog Strategy\n\nFull catalog body.\n", encoding="utf-8")
    entry = StrategyCatalogEntry(
        id="strategy_a",
        title="Strategy A",
        summary="Use strategy A.",
        markdown_ref="source_strategy.md",
        markdown_path=strategy_source,
    )

    ctx = materialize_strategy_context_from_catalog_entry(
        project_dir=tmp_path,
        entry=entry,
        strategy_summary="Use strategy A.",
        eval_summary={"schema_version": 1},
        eval_log="",
    )

    manifest = json.loads((tmp_path / ctx.manifest_json).read_text(encoding="utf-8"))
    assert (tmp_path / ctx.strategy_md).read_text(encoding="utf-8") == strategy_source.read_text(
        encoding="utf-8"
    )
    assert manifest["strategy_source"]["kind"] == "catalog_entry"
    assert manifest["strategy_source"]["strategy_id"] == "strategy_a"
    assert manifest["strategy_source"]["markdown_ref"] == "source_strategy.md"


def test_context_materializer_rejects_truncated_canonical_strategy_source(tmp_path):
    strategy_source = tmp_path / "source_strategy.md"
    strategy_source.write_text(
        "# Strategy\n\nBody\n\n[truncated strategy markdown]\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="truncation marker"):
        materialize_worktree_context(
            project_dir=tmp_path,
            canonical_strategy_markdown_path=strategy_source,
            strategy_summary="Short bounded summary.",
            eval_summary={"schema_version": 1, "eval_context_status": "no_prior_eval"},
            eval_log="# Evaluation Log\n\nNo prior eval.",
        )


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
