from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTEXT_ROOT = ".ksearch/context"


@dataclass(frozen=True)
class WorktreeContextPaths:
    strategy_md: str
    strategy_summary_md: str
    eval_summary_json: str
    eval_log_md: str
    manifest_json: str


def to_worktree_relative_path(project_dir: Path, path: Path) -> str:
    root = Path(project_dir).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    try:
        rel = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"context path escaped project root: {path}") from exc
    s = rel.as_posix()
    if not s or s == ".." or s.startswith("../") or Path(s).is_absolute():
        raise ValueError(f"context path escaped project root: {path}")
    return s


def assert_no_absolute_paths_for_llm(text: str) -> None:
    forbidden = (
        "/tmp/",
        "/mnt/",
        "/home/",
        "file://",
        "ksearch_worktrees/",
        "claude-",
    )
    value = str(text or "")
    for token in forbidden:
        if token in value:
            raise AssertionError(f"absolute or host path leaked to LLM text: {token}")


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def materialize_worktree_context(
    *,
    project_dir: Path,
    strategy_markdown: str,
    strategy_summary: str,
    eval_summary: dict[str, Any],
    eval_log: str,
    strategy_context: dict[str, Any] | None = None,
) -> WorktreeContextPaths:
    root = Path(project_dir).expanduser().resolve()
    context_dir = root / CONTEXT_ROOT
    context_dir.mkdir(parents=True, exist_ok=True)

    strategy_md = context_dir / "STRATEGY.md"
    strategy_summary_md = context_dir / "STRATEGY_SUMMARY.md"
    eval_summary_json = context_dir / "EVAL_SUMMARY.json"
    eval_log_md = context_dir / "EVAL_LOG.md"
    manifest_json = context_dir / "CONTEXT_MANIFEST.json"

    strategy_md.write_text(str(strategy_markdown or "").strip() + "\n", encoding="utf-8")
    strategy_summary_md.write_text(str(strategy_summary or "").strip() + "\n", encoding="utf-8")
    eval_summary_json.write_text(_json_dumps(dict(eval_summary or {})), encoding="utf-8")
    eval_log_md.write_text(str(eval_log or "").strip() + "\n", encoding="utf-8")

    paths = WorktreeContextPaths(
        strategy_md=to_worktree_relative_path(root, strategy_md),
        strategy_summary_md=to_worktree_relative_path(root, strategy_summary_md),
        eval_summary_json=to_worktree_relative_path(root, eval_summary_json),
        eval_log_md=to_worktree_relative_path(root, eval_log_md),
        manifest_json=to_worktree_relative_path(root, manifest_json),
    )
    manifest = {
        "schema_version": 1,
        "context_root": CONTEXT_ROOT,
        "strategy_md": paths.strategy_md,
        "strategy_summary_md": paths.strategy_summary_md,
        "eval_summary_json": paths.eval_summary_json,
        "eval_log_md": paths.eval_log_md,
    }
    if strategy_context:
        manifest["strategy"] = dict(strategy_context)
    manifest_json.write_text(_json_dumps(manifest), encoding="utf-8")
    return paths
