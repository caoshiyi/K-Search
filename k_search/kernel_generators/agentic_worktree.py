from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from k_search.kernel_generators.runtime_artifacts import NATIVE_RUNTIME_DIRS


class AgenticWorktreeError(RuntimeError):
    """Raised when K-Search cannot prepare or inspect an agentic worktree."""


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=check,
    )


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd, check=check)


def _git_stdout(cwd: Path, *args: str) -> str:
    return _git(cwd, *args).stdout.strip()


def _find_git_root(path: Path) -> Optional[Path]:
    proc = _git(path, "rev-parse", "--show-toplevel", check=False)
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return Path(out).resolve() if out else None


def _copy_project(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(".git", *NATIVE_RUNTIME_DIRS, "__pycache__", "build", "cmake-build-debug", "logs")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _mirror_project_state(src: Path, dst: Path, *, worktree_root: Path) -> None:
    """Make the candidate project match the current task directory on disk."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst == worktree_root:
        dst.mkdir(parents=True, exist_ok=True)
        for child in list(dst.iterdir()):
            if child.name == ".git":
                continue
            _remove_path(child)
    else:
        if dst.exists() or dst.is_symlink():
            _remove_path(dst)
        dst.mkdir(parents=True, exist_ok=True)
    _copy_project(src, dst)


def _git_status_paths(root: Path) -> list[str]:
    proc = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "-z")
    entries = [entry for entry in proc.stdout.split("\0") if entry]
    paths: list[str] = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        status = entry[:2]
        path = entry[3:] if len(entry) > 3 else ""
        if path:
            paths.append(path)
        # Rename/copy records carry an additional path entry in porcelain -z.
        i += 2 if status[:1] in {"R", "C"} or status[1:2] in {"R", "C"} else 1
    return sorted(dict.fromkeys(paths))


def _is_tracked_path(root: Path, rel_path: str) -> bool:
    proc = _git(root, "ls-files", "--error-unmatch", "--", rel_path, check=False)
    return proc.returncode == 0


def _untracked_file_diff(root: Path, rel_path: str) -> str:
    path = root / rel_path
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(
        difflib.unified_diff(
            [],
            content.splitlines(),
            fromfile="/dev/null",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
    )


@dataclass
class AgenticWorktreeSession:
    worktree_root: Path
    project_dir: Path
    repo_root: Optional[Path]
    is_real_worktree: bool
    keep: bool
    baseline_commit: str

    def changed_paths(self) -> list[str]:
        return _git_status_paths(self.worktree_root)

    def project_rel_path(self) -> str:
        try:
            rel = self.project_dir.resolve().relative_to(self.worktree_root.resolve())
        except ValueError:
            return "."
        text = str(rel).replace("\\", "/")
        return text if text else "."

    def project_changed_paths(self) -> list[str]:
        prefix = self.project_rel_path()
        if prefix == ".":
            return self.changed_paths()
        project_prefix = prefix.rstrip("/") + "/"
        out: list[str] = []
        for path in self.changed_paths():
            if path == prefix:
                out.append(Path(path).name)
            elif path.startswith(project_prefix):
                out.append(path[len(project_prefix) :])
        return sorted(dict.fromkeys(out))

    def has_changes(self) -> bool:
        return bool(self.changed_paths())

    def diff_text(self) -> str:
        parts = [_git_stdout(self.worktree_root, "diff", "HEAD", "--")]
        for rel_path in self.changed_paths():
            if not _is_tracked_path(self.worktree_root, rel_path):
                parts.append(_untracked_file_diff(self.worktree_root, rel_path))
        return "\n".join(part for part in parts if part)

    def project_diff_text(self) -> str:
        prefix = self.project_rel_path()
        if prefix == ".":
            return self.diff_text()
        parts = [_git_stdout(self.worktree_root, "diff", f"--relative={prefix}", "HEAD", "--", prefix)]
        for rel_path in self.changed_paths():
            project_prefix = prefix.rstrip("/") + "/"
            if not rel_path.startswith(project_prefix):
                continue
            project_rel = rel_path[len(project_prefix) :]
            if not _is_tracked_path(self.worktree_root, rel_path):
                parts.append(_untracked_file_diff(self.project_dir, project_rel))
        return "\n".join(part for part in parts if part)

    def commit_all(self, message: str) -> str:
        self.baseline_commit = _commit_all(self.worktree_root, message)
        return self.baseline_commit

    def cleanup(self) -> None:
        if self.keep:
            return
        if self.is_real_worktree and self.repo_root is not None:
            _git(self.repo_root, "worktree", "remove", "--force", str(self.worktree_root), check=False)
            return
        shutil.rmtree(self.worktree_root, ignore_errors=True)


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    proc = _git(root, "diff", "--cached", "--quiet", check=False)
    if proc.returncode == 1:
        _git(
            root,
            "-c",
            "user.email=ksearch@example.invalid",
            "-c",
            "user.name=K Search Agentic Codegen",
            "commit",
            "-m",
            message,
        )
    elif _git(root, "rev-parse", "--verify", "HEAD", check=False).returncode != 0:
        _git(
            root,
            "-c",
            "user.email=ksearch@example.invalid",
            "-c",
            "user.name=K Search Agentic Codegen",
            "commit",
            "--allow-empty",
            "-m",
            message,
        )
    return _git_stdout(root, "rev-parse", "HEAD")


def create_agentic_worktree(*, task_path: str | Path | None) -> AgenticWorktreeSession:
    if task_path is None:
        raise AgenticWorktreeError("AscendC agentic codegen requires task_path")

    task_root = Path(task_path).expanduser().resolve()
    if not task_root.exists() or not task_root.is_dir():
        raise AgenticWorktreeError(f"task_path is not a directory: {task_root}")

    keep = os.getenv("KSEARCH_KEEP_AGENTIC_WORKTREES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    repo_root = _find_git_root(task_root)

    if repo_root is not None:
        temp_root = Path(tempfile.mkdtemp(prefix="ksearch_agentic_worktree_")).resolve()
        try:
            _git(repo_root, "worktree", "add", "--detach", str(temp_root), "HEAD")
            rel_project = task_root.relative_to(repo_root)
            project_dir = temp_root / rel_project
            _mirror_project_state(task_root, project_dir, worktree_root=temp_root)
            baseline_commit = _commit_all(temp_root, "ksearch agentic baseline")
            return AgenticWorktreeSession(
                worktree_root=temp_root,
                project_dir=project_dir,
                repo_root=repo_root,
                is_real_worktree=True,
                keep=keep,
                baseline_commit=baseline_commit,
            )
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)

    fallback_root = Path(tempfile.mkdtemp(prefix="ksearch_agentic_temp_repo_")).resolve()
    _copy_project(task_root, fallback_root)
    _git(fallback_root, "init")
    baseline_commit = _commit_all(fallback_root, "ksearch agentic baseline")
    return AgenticWorktreeSession(
        worktree_root=fallback_root,
        project_dir=fallback_root,
        repo_root=None,
        is_real_worktree=False,
        keep=keep,
        baseline_commit=baseline_commit,
    )
