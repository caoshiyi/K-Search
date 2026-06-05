from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


CLAUDE_ASSET_MANAGED_MARKER = "<!-- K-Search managed Claude asset -->"

NATIVE_AGENT_FILES = [
    "code-reader.md",
    "plan.md",
    "codegen.md",
    "reviewer.md",
    "bug-fixer.md",
]

NATIVE_SKILL_FILES = [
    "ascendc-codegen/SKILL.md",
    "ascendc-api-reference/SKILL.md",
]

NATIVE_SKILLS = ["ascendc-codegen", "ascendc-api-reference"]
NATIVE_AGENT_TOOL_NAMES = ["Agent"]

NATIVE_HANDOFF_FILES = {
    "CODE_MAP.md",
    "IMPLEMENTATION_PLAN.md",
    "REVIEW_NOTES.md",
}


@dataclass(frozen=True)
class ClaudeAssetMaterializationResult:
    claude_dir: Path
    written_paths: list[Path]


def _asset_text(relative_path: str) -> str:
    root = files("k_search.kernel_generators.claude_assets")
    return (root / relative_path).read_text(encoding="utf-8")


def _write_managed_text(target: Path, text: str) -> None:
    if target.exists():
        current = target.read_text(encoding="utf-8", errors="replace")
        if not current.startswith(CLAUDE_ASSET_MANAGED_MARKER):
            raise FileExistsError(f"refusing to overwrite unmanaged Claude asset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    body = text if text.startswith(CLAUDE_ASSET_MANAGED_MARKER) else f"{CLAUDE_ASSET_MANAGED_MARKER}\n{text}"
    target.write_text(body, encoding="utf-8")


def materialize_claude_project_assets(project_dir: str | Path) -> ClaudeAssetMaterializationResult:
    root = Path(project_dir).expanduser().resolve()
    claude_dir = root / ".claude"
    written: list[Path] = []

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

    return ClaudeAssetMaterializationResult(claude_dir=claude_dir, written_paths=written)
