from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from k_search.kernel_generators.strategy_injection import StrategyCatalogEntry


CONTEXT_ROOT = ".ksearch/context"
TRUNCATED_STRATEGY_MARKDOWN_MARKER = "[truncated strategy markdown]"
TRUNCATED_AGENTIC_PROMPT_MARKER = "[truncated for agentic prompt budget]"
TRUNCATION_MARKERS = (
    TRUNCATED_STRATEGY_MARKDOWN_MARKER,
    TRUNCATED_AGENTIC_PROMPT_MARKER,
)


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_no_strategy_truncation_markers(text: str, *, source: str) -> None:
    value = str(text or "")
    for marker in TRUNCATION_MARKERS:
        if marker in value:
            raise RuntimeError(
                f"canonical strategy source contains truncation marker {marker}: {source}"
            )


def _safe_source_path_ref(project_dir: Path, source_path: Path) -> str:
    try:
        return to_worktree_relative_path(project_dir, source_path)
    except ValueError:
        return source_path.name


def _strategy_source_from_entry(
    project_dir: Path,
    entry: "StrategyCatalogEntry",
) -> tuple[bytes, dict[str, Any]]:
    from k_search.kernel_generators.strategy_injection import StrategyCatalogEntry

    if not isinstance(entry, StrategyCatalogEntry):
        raise TypeError("strategy_entry must be a StrategyCatalogEntry")

    source_path = Path(entry.markdown_path)
    source_bytes = source_path.read_bytes()
    source_text = source_bytes.decode("utf-8", errors="replace")
    _assert_no_strategy_truncation_markers(source_text, source=f"strategy {entry.id}")
    return source_bytes, {
        "kind": "catalog_entry",
        "strategy_id": entry.id,
        "markdown_ref": entry.markdown_ref,
        "source_ref": _safe_source_path_ref(project_dir, source_path),
    }


def _strategy_source_from_path(
    project_dir: Path,
    source_path: Path,
) -> tuple[bytes, dict[str, Any]]:
    path = Path(source_path).expanduser()
    source_bytes = path.read_bytes()
    source_text = source_bytes.decode("utf-8", errors="replace")
    _assert_no_strategy_truncation_markers(
        source_text,
        source=_safe_source_path_ref(project_dir, path),
    )
    return source_bytes, {
        "kind": "markdown_path",
        "source_ref": _safe_source_path_ref(project_dir, path),
    }


def _canonical_strategy_source(
    *,
    project_dir: Path,
    strategy_entry: "StrategyCatalogEntry" | None,
    canonical_strategy_markdown_path: Path | None,
) -> tuple[bytes, dict[str, Any]]:
    if strategy_entry is not None and canonical_strategy_markdown_path is not None:
        raise ValueError(
            "pass either strategy_entry or canonical_strategy_markdown_path, not both"
        )
    if strategy_entry is not None:
        return _strategy_source_from_entry(project_dir, strategy_entry)
    if canonical_strategy_markdown_path is not None:
        return _strategy_source_from_path(project_dir, canonical_strategy_markdown_path)
    return b"", {"kind": "none"}


def materialize_worktree_context(
    *,
    project_dir: Path,
    strategy_entry: "StrategyCatalogEntry" | None = None,
    canonical_strategy_markdown_path: Path | None = None,
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

    strategy_bytes, strategy_source = _canonical_strategy_source(
        project_dir=root,
        strategy_entry=strategy_entry,
        canonical_strategy_markdown_path=canonical_strategy_markdown_path,
    )
    source_text = strategy_bytes.decode("utf-8", errors="replace")
    source_sha256 = _sha256_bytes(strategy_bytes)
    source_chars = len(source_text)

    strategy_md.write_bytes(strategy_bytes)
    materialized_bytes = strategy_md.read_bytes()
    materialized_text = materialized_bytes.decode("utf-8", errors="replace")
    materialized_sha256 = _sha256_bytes(materialized_bytes)
    materialized_chars = len(materialized_text)
    if source_sha256 != materialized_sha256 or source_chars != materialized_chars:
        raise RuntimeError(
            "materialized STRATEGY.md does not match canonical strategy source "
            f"(source_sha256={source_sha256}, materialized_sha256={materialized_sha256}, "
            f"source_chars={source_chars}, materialized_chars={materialized_chars})"
        )
    _assert_no_strategy_truncation_markers(materialized_text, source="materialized STRATEGY.md")

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
        "strategy_source": {
            **strategy_source,
            "source_sha256": source_sha256,
            "materialized_sha256": materialized_sha256,
            "source_chars": source_chars,
            "materialized_chars": materialized_chars,
        },
    }
    if strategy_context:
        manifest["strategy"] = dict(strategy_context)
    manifest_json.write_text(_json_dumps(manifest), encoding="utf-8")
    return paths


def materialize_strategy_context_from_catalog_entry(
    *,
    project_dir: Path,
    entry: "StrategyCatalogEntry",
    strategy_summary: str,
    eval_summary: dict[str, Any],
    eval_log: str,
    strategy_context: dict[str, Any] | None = None,
) -> WorktreeContextPaths:
    return materialize_worktree_context(
        project_dir=project_dir,
        strategy_entry=entry,
        strategy_summary=strategy_summary,
        eval_summary=eval_summary,
        eval_log=eval_log,
        strategy_context=strategy_context,
    )
