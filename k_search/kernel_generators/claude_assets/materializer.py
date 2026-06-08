from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from k_search.kernel_generators.runtime_artifacts import NATIVE_HANDOFF_FILES


logger = logging.getLogger(__name__)

CLAUDE_ASSET_MANAGED_MARKER = "<!-- K-Search managed Claude asset -->"

NATIVE_AGENT_FILES = [
    "code-reader.md",
    "designer.md",
    "codegen.md",
    "reviewer.md",
    "bug-fixer.md",
    "knowledge-curator.md",
]

LEGACY_NATIVE_AGENT_FILES = [
    "plan.md",
]

# Skills whose only payload is SKILL.md (materialized as managed text).
NATIVE_SKILL_FILES = [
    "ascendc-codegen/SKILL.md",
    "ascendc-api-reference/SKILL.md",
    "ascendc-hardware/SKILL.md",
    "ascendc-sync-guide/SKILL.md",
    "ascendc-verify/SKILL.md",
    "ascendc-dumptensor/SKILL.md",
    "ascendc-fa-detailed-design/SKILL.md",
    "ascendc-dev-knowledge/SKILL.md",
]

NATIVE_SKILLS = [
    "ascendc-codegen",
    "ascendc-api-reference",
    "ascendc-hardware",
    "ascendc-sync-guide",
    "ascendc-verify",
    "ascendc-dumptensor",
    "ascendc-fa-detailed-design",
    "ascendc-dev-knowledge",
]

# Skills that ship a sibling ``references/`` directory. These are copied into
# each candidate worktree so agent writes cannot mutate repo-side assets.
NATIVE_SKILL_REFERENCE_DIRS = [
    "ascendc-dumptensor",
    "ascendc-fa-detailed-design",
    "ascendc-dev-knowledge",
]

# Shared reference packs (design principles, attention patterns, curation format)
# materialized under .claude/references/<name>.
NATIVE_REFERENCE_DIRS = [
    "ascendc-design",
    "attention-patterns",
    "curation-format",
    "known-pitfalls",
]

NATIVE_AGENT_TOOL_NAMES = ["Agent"]

@dataclass(frozen=True)
class ClaudeAssetMaterializationResult:
    claude_dir: Path
    written_paths: list[Path]
    linked_paths: list[Path] = field(default_factory=list)


def _asset_root() -> Path:
    return Path(str(files("k_search.kernel_generators.claude_assets")))


def _asset_text(relative_path: str) -> str:
    return (_asset_root() / relative_path).read_text(encoding="utf-8")


def _write_managed_text(target: Path, text: str) -> None:
    if target.exists():
        if target.is_file() or target.is_symlink():
            current = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
            if not current.startswith(CLAUDE_ASSET_MANAGED_MARKER):
                _remove_existing_path(target)
        else:
            _remove_existing_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = text if text.startswith(CLAUDE_ASSET_MANAGED_MARKER) else f"{CLAUDE_ASSET_MANAGED_MARKER}\n{text}"
    target.write_text(body, encoding="utf-8")


def _remove_existing_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _remove_legacy_managed_file(path: Path) -> None:
    if not path.is_file():
        return
    current = path.read_text(encoding="utf-8", errors="replace")
    if current.startswith(CLAUDE_ASSET_MANAGED_MARKER):
        path.unlink()


def _allow_missing_dev_knowledge() -> bool:
    return os.getenv("KSEARCH_ALLOW_MISSING_DEV_KNOWLEDGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _copy_dir(source: Path, target: Path, copied: list[Path], *, required: bool = False) -> None:
    """Copy ``source`` into ``target`` for isolated candidate-local references.

    Missing non-critical source is a soft failure. Required source packs fail fast
    unless the caller explicitly opts into degraded operation.
    """
    if not source.is_dir():
        if required and not _allow_missing_dev_knowledge():
            raise RuntimeError(
                "required Claude asset reference dir missing: "
                f"{source}. Install ascendc-dev-knowledge/references or set "
                "KSEARCH_ALLOW_MISSING_DEV_KNOWLEDGE=1 to run without it."
            )
        logger.warning("claude asset reference dir missing, skipping copy: %s", source)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _remove_existing_path(target)
    shutil.copytree(source, target, symlinks=False)
    copied.append(target)


def materialize_claude_project_assets(project_dir: str | Path) -> ClaudeAssetMaterializationResult:
    root = Path(project_dir).expanduser().resolve()
    claude_dir = root / ".claude"
    written: list[Path] = []
    copied: list[Path] = []
    asset_root = _asset_root()

    for name in LEGACY_NATIVE_AGENT_FILES:
        _remove_legacy_managed_file(claude_dir / "agents" / name)

    for name in NATIVE_AGENT_FILES:
        text = _asset_text(f"agents/{name}")
        target = claude_dir / "agents" / name
        _write_managed_text(target, text)
        written.append(target)

    for rel in NATIVE_SKILL_FILES:
        text = _asset_text(f"skills/{rel}")
        target = claude_dir / "skills" / rel
        _write_managed_text(target, text)
        written.append(target)

    # Large skill reference dirs: copy so the candidate worktree is isolated.
    for skill in NATIVE_SKILL_REFERENCE_DIRS:
        source = asset_root / "skills" / skill / "references"
        target = claude_dir / "skills" / skill / "references"
        _copy_dir(source, target, copied, required=(skill == "ascendc-dev-knowledge"))

    # Shared reference packs under .claude/references/.
    for name in NATIVE_REFERENCE_DIRS:
        source = asset_root / "references" / name
        target = claude_dir / "references" / name
        _copy_dir(source, target, copied)

    return ClaudeAssetMaterializationResult(claude_dir=claude_dir, written_paths=written, linked_paths=[])
