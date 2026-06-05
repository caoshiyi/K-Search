from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


SNAPSHOT_SKIP_DIRS = {
    ".git",
    ".ksearch",
    ".claude",
    "CODE_MAP.md",
    "IMPLEMENTATION_PLAN.md",
    "REVIEW_NOTES.md",
    "__pycache__",
    "build",
    "cmake-build-debug",
    "logs",
    "llm_logs",
}


@dataclass(frozen=True)
class FileMeta:
    path: str
    sha256: str
    size: int
    mode: str
    kind: Literal["file", "symlink"]
    link_target: str | None = None


@dataclass(frozen=True)
class ProjectSnapshot:
    snapshot_id: str
    parent_snapshot_id: str | None
    base_commit: str | None
    project_root: str
    manifest: dict[str, FileMeta]
    archive_path: str | None
    diff_from_parent: str | None
    created_by_round: int
    eval_result: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest"] = {path: asdict(meta) for path, meta in sorted(self.manifest.items())}
        return data


def _is_skipped(rel: Path) -> bool:
    return any(part in SNAPSHOT_SKIP_DIRS for part in rel.parts)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mode_string(mode: int) -> str:
    return oct(stat.S_IMODE(mode))


def _copy_snapshot_payload(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if _is_skipped(rel):
            continue
        target = dst / rel
        if p.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.readlink(p), target)
        elif p.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        elif p.is_dir():
            target.mkdir(parents=True, exist_ok=True)


def create_project_snapshot(
    *,
    project_dir: str | Path,
    snapshot_id: str,
    parent_snapshot_id: str | None,
    base_commit: str | None,
    created_by_round: int,
    eval_result: dict[str, Any] | None = None,
    diff_from_parent: str | None = None,
    archive_dir: str | Path | None = None,
    run_id: str | None = None,  # New parameter for organizing runs
) -> ProjectSnapshot:
    root = Path(project_dir).expanduser().resolve()
    manifest: dict[str, FileMeta] = {}
    for p in sorted(root.rglob("*")):
        rel_path = p.relative_to(root)
        if _is_skipped(rel_path):
            continue
        rel = str(rel_path).replace("\\", "/")
        try:
            st = p.lstat()
            if p.is_symlink():
                target = os.readlink(p)
                manifest[rel] = FileMeta(
                    path=rel,
                    sha256=hashlib.sha256(target.encode("utf-8", errors="replace")).hexdigest(),
                    size=len(target.encode("utf-8", errors="replace")),
                    mode=_mode_string(st.st_mode),
                    kind="symlink",
                    link_target=target,
                )
            elif p.is_file():
                manifest[rel] = FileMeta(
                    path=rel,
                    sha256=_file_sha256(p),
                    size=st.st_size,
                    mode=_mode_string(st.st_mode),
                    kind="file",
                )
        except OSError:
            continue

    archive_path: str | None = None
    if archive_dir is not None:
        archive_root = Path(archive_dir).expanduser().resolve()
        archive_dst = archive_root / str(snapshot_id)
        _copy_snapshot_payload(root, archive_dst)
        archive_path = str(archive_dst)

    return ProjectSnapshot(
        snapshot_id=str(snapshot_id),
        parent_snapshot_id=parent_snapshot_id,
        base_commit=base_commit,
        project_root=str(root),
        manifest=manifest,
        archive_path=archive_path,
        diff_from_parent=diff_from_parent,
        created_by_round=int(created_by_round),
        eval_result=eval_result,
    )


def materialize_project_snapshot(snapshot: ProjectSnapshot, destination: str | Path) -> Path:
    src = Path(snapshot.archive_path or snapshot.project_root).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    _copy_snapshot_payload(src, dst)
    return dst
