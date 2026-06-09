import os
import shutil
import subprocess
from pathlib import Path

from k_search.kernel_generators.agentic_worktree import create_agentic_worktree


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.email", "ksearch@example.invalid")
    _git(root, "config", "user.name", "K Search Tests")
    (root / "kernel").mkdir()
    (root / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (root / "spec.md").write_text("Optimize this project.", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")


def test_create_agentic_worktree_from_git_repo_tracks_project_subdir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    task_path = repo / "kernel"
    session = create_agentic_worktree(task_path=task_path)

    try:
        assert session.project_dir == session.worktree_root / "kernel"
        assert session.project_dir.exists()
        assert session.is_real_worktree is True
        assert session.repo_root == repo.resolve()
        assert (session.project_dir / "foo.h").read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"

        (session.project_dir / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
        assert session.changed_paths() == ["kernel/foo.h"]
        assert "-beta" in session.diff_text()
        assert "+BETA" in session.diff_text()
    finally:
        path = session.worktree_root
        session.cleanup()

    assert not path.exists()


def test_real_worktree_mirrors_dirty_task_path_and_preserves_repo_git_config(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    _git(repo, "config", "user.email", "real@example.test")
    _git(repo, "config", "user.name", "Real User")
    (repo / "kernel" / "foo.h").write_text("dirty\n", encoding="utf-8")

    session = create_agentic_worktree(task_path=repo / "kernel")

    try:
        assert (session.project_dir / "foo.h").read_text(encoding="utf-8") == "dirty\n"
        assert _git(repo, "config", "user.email") == "real@example.test"
        assert _git(repo, "config", "user.name") == "Real User"
    finally:
        session.cleanup()


def test_real_worktree_materializes_untracked_task_subdir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "ksearch@example.invalid")
    _git(repo, "config", "user.name", "K Search Tests")
    (repo / "README.md").write_text("root\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    task_path = repo / "untracked_task"
    task_path.mkdir()
    (task_path / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")

    session = create_agentic_worktree(task_path=task_path)

    try:
        assert session.project_dir.exists()
        assert (session.project_dir / "kernel.cpp").read_text(encoding="utf-8") == "void run() {}\n"
    finally:
        session.cleanup()


def test_real_worktree_removes_ancestor_project_memory_without_touching_project_memory(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "ksearch@example.invalid")
    _git(repo, "config", "user.name", "K Search Tests")
    parent_scope = repo / "workspace_scope"
    task_path = parent_scope / "candidate_project"
    task_path.mkdir(parents=True)
    (repo / "CLAUDE.md").write_text("repo-root instructions\n", encoding="utf-8")
    (parent_scope / "CLAUDE.md").write_text("parent-scope instructions\n", encoding="utf-8")
    (parent_scope / "AGENTS.md").write_text("parent-scope agent instructions\n", encoding="utf-8")
    (task_path / "CLAUDE.md").write_text("task-local instructions\n", encoding="utf-8")
    (task_path / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    session = create_agentic_worktree(task_path=task_path)

    try:
        assert not (session.worktree_root / "CLAUDE.md").exists()
        assert not (session.worktree_root / "workspace_scope" / "CLAUDE.md").exists()
        assert not (session.worktree_root / "workspace_scope" / "AGENTS.md").exists()
        assert (session.project_dir / "CLAUDE.md").read_text(encoding="utf-8") == "task-local instructions\n"
        assert session.changed_paths() == []
    finally:
        session.cleanup()

    assert (repo / "CLAUDE.md").read_text(encoding="utf-8") == "repo-root instructions\n"
    assert (parent_scope / "CLAUDE.md").read_text(encoding="utf-8") == "parent-scope instructions\n"
    assert (parent_scope / "AGENTS.md").read_text(encoding="utf-8") == "parent-scope agent instructions\n"


def test_create_agentic_worktree_falls_back_for_non_git_task_path(tmp_path):
    task_path = tmp_path / "plain_task"
    task_path.mkdir()
    (task_path / "kernel.cpp").write_text("int old_value = 1;\n", encoding="utf-8")
    (task_path / "build").mkdir()
    (task_path / "build" / "artifact.o").write_text("ignored\n", encoding="utf-8")

    session = create_agentic_worktree(task_path=task_path)

    try:
        assert session.is_real_worktree is False
        assert session.repo_root is None
        assert session.project_dir == session.worktree_root
        assert (session.project_dir / "kernel.cpp").exists()
        assert not (session.project_dir / "build").exists()
        assert session.baseline_commit
        (session.project_dir / "kernel.cpp").write_text("int old_value = 2;\n", encoding="utf-8")
        assert session.changed_paths() == ["kernel.cpp"]
    finally:
        path = session.worktree_root
        session.cleanup()

    assert not path.exists()


def test_create_agentic_worktree_uses_requested_parent_dir(tmp_path):
    task_path = tmp_path / "plain_task"
    task_path.mkdir()
    (task_path / "kernel.cpp").write_text("int old_value = 1;\n", encoding="utf-8")
    parent = tmp_path / "out" / "task" / "task1" / "runs" / "run1" / "worktrees"

    session = create_agentic_worktree(task_path=task_path, worktree_parent_dir=parent)

    try:
        assert session.worktree_root.parent == parent.resolve()
        assert session.worktree_root.name.startswith("ksearch_agentic_temp_repo_")
        assert (session.project_dir / "kernel.cpp").exists()
    finally:
        path = session.worktree_root
        session.cleanup()

    assert parent.exists()
    assert not path.exists()


def test_changed_paths_and_diff_include_new_untracked_files(tmp_path):
    task_path = tmp_path / "plain_task"
    task_path.mkdir()
    (task_path / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")

    session = create_agentic_worktree(task_path=task_path)

    try:
        (session.project_dir / "new_helper.h").write_text("int helper;\n", encoding="utf-8")
        assert session.changed_paths() == ["new_helper.h"]
        assert "new_helper.h" in session.diff_text()
        assert "+int helper;" in session.diff_text()
    finally:
        path = session.worktree_root
        session.cleanup()

    assert not path.exists()


def test_non_git_fallback_does_not_leave_unused_temp_root(tmp_path, monkeypatch):
    import k_search.kernel_generators.agentic_worktree as agentic_worktree

    task_path = tmp_path / "plain_task"
    task_path.mkdir()
    (task_path / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")
    created: list[Path] = []

    def fake_mkdtemp(prefix: str) -> str:
        path = tmp_path / f"{prefix}{len(created)}"
        path.mkdir()
        created.append(path)
        return str(path)

    monkeypatch.setattr(agentic_worktree.tempfile, "mkdtemp", fake_mkdtemp)

    session = create_agentic_worktree(task_path=task_path)
    session.cleanup()

    assert created
    assert all(not path.exists() for path in created)


def test_agentic_worktree_preserve_env_keeps_directory(tmp_path, monkeypatch):
    task_path = tmp_path / "plain_task"
    task_path.mkdir()
    (task_path / "kernel.cpp").write_text("int x = 1;\n", encoding="utf-8")
    monkeypatch.setenv("KSEARCH_KEEP_AGENTIC_WORKTREES", "1")

    session = create_agentic_worktree(task_path=task_path)
    path = session.worktree_root
    session.cleanup()

    try:
        assert path.exists()
    finally:
        shutil.rmtree(path, ignore_errors=True)
