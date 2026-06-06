from pathlib import Path

import pytest

from k_search.utils import paths as paths_mod
from k_search.utils.paths import (
    get_ksearch_task_artifacts_dir,
    get_ksearch_task_dir,
    get_ksearch_run_dir,
    get_ksearch_artifacts_dir,
    get_ksearch_worktrees_dir,
    get_run_id,
    get_run_logs_dir,
    get_task_id,
    resolve_output_base,
    safe_path_component,
)


@pytest.fixture(autouse=True)
def reset_generated_task_id(monkeypatch):
    monkeypatch.setattr(paths_mod, "_generated_task_id", None, raising=False)


def test_get_run_id_prefers_run_id_over_run_start(monkeypatch):
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-1")
    monkeypatch.setenv("KSEARCH_RUN_START", "run-2")
    assert get_run_id() == "run-1"


def test_get_run_id_falls_back_to_run_start(monkeypatch):
    monkeypatch.delenv("KSEARCH_RUN_ID", raising=False)
    monkeypatch.setenv("KSEARCH_RUN_START", "20260530_101112")
    assert get_run_id() == "20260530_101112"


def test_get_run_id_falls_back_to_local_timestamp(monkeypatch):
    monkeypatch.delenv("KSEARCH_RUN_ID", raising=False)
    monkeypatch.delenv("KSEARCH_RUN_START", raising=False)
    result = get_run_id()
    assert len(result) == 15  # YYYYMMDD_HHMMSS
    assert result[8] == "_"


def test_get_task_id_prefers_task_id_over_task_start(monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-1")
    monkeypatch.setenv("KSEARCH_TASK_START", "task-2")
    assert get_task_id() == "task-1"


def test_get_task_id_falls_back_to_task_start(monkeypatch):
    monkeypatch.delenv("KSEARCH_TASK_ID", raising=False)
    monkeypatch.setenv("KSEARCH_TASK_START", "20260606_101112")
    assert get_task_id() == "20260606_101112"


def test_get_task_id_falls_back_to_local_timestamp(monkeypatch):
    monkeypatch.delenv("KSEARCH_TASK_ID", raising=False)
    monkeypatch.delenv("KSEARCH_TASK_START", raising=False)
    result = get_task_id()
    assert len(result) == 15  # YYYYMMDD_HHMMSS
    assert result[8] == "_"


def test_get_task_id_no_env_result_is_stable(monkeypatch):
    monkeypatch.delenv("KSEARCH_TASK_ID", raising=False)
    monkeypatch.delenv("KSEARCH_TASK_START", raising=False)
    values = iter(["20260606_101112", "20260606_101113"])
    monkeypatch.setattr(paths_mod, "_timestamp_id", lambda: next(values))

    assert get_task_id() == "20260606_101112"
    assert get_task_id() == "20260606_101112"


def test_resolve_output_base_priority(monkeypatch, tmp_path):
    # Explicit arg wins.
    monkeypatch.setenv("KSEARCH_ARTIFACTS_DIR", str(tmp_path / "env"))
    assert resolve_output_base(tmp_path / "arg") == (tmp_path / "arg").resolve()
    # Env next.
    assert resolve_output_base() == (tmp_path / "env").resolve()
    # Default last.
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_output_base() == (tmp_path / ".ksearch").resolve()


def test_get_run_logs_dir_layout(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("KSEARCH_TASK_ID", "task:alpha")
    p = get_run_logs_dir(base_dir=tmp_path, task_name="vec/add", run_id="run:1", sub="llm")
    assert p == (tmp_path / "vec_add" / "task_alpha" / "runs" / "run_1" / "logs" / "llm").resolve()
    # No sub -> run logs root.
    root = get_run_logs_dir(base_dir=tmp_path, task_name="vec/add", run_id="run:1")
    assert root == (tmp_path / "vec_add" / "task_alpha" / "runs" / "run_1" / "logs").resolve()


def test_run_artifacts_logs_and_worktrees_share_one_run_root(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("KSEARCH_TASK_ID", "task:alpha")
    monkeypatch.setenv("KSEARCH_RUN_ID", "run:1")
    task_root = get_ksearch_task_dir(base_dir=tmp_path, task_name="vec/add")
    run_root = get_ksearch_run_dir(base_dir=tmp_path, task_name="vec/add")

    assert task_root == (tmp_path / "vec_add" / "task_alpha").resolve()
    assert run_root == task_root / "runs" / "run_1"
    assert get_ksearch_task_artifacts_dir(base_dir=tmp_path, task_name="vec/add") == task_root / "artifacts"
    assert get_ksearch_artifacts_dir(base_dir=tmp_path, task_name="vec/add") == run_root / "artifacts"
    assert get_run_logs_dir(base_dir=tmp_path, task_name="vec/add") == run_root / "logs"
    assert get_ksearch_worktrees_dir(base_dir=tmp_path, task_name="vec/add") == run_root / "worktrees"


def test_artifacts_dir_can_use_task_layout_without_run(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("KSEARCH_TASK_ID", "task:alpha")
    p = get_ksearch_artifacts_dir(base_dir=tmp_path, task_name="vec/add", include_run=False)
    assert p == (tmp_path / "vec_add" / "task_alpha" / "artifacts").resolve()


def test_explicit_task_id_flows_through_run_scoped_helpers(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("KSEARCH_TASK_ID", "task:env")
    monkeypatch.setenv("KSEARCH_RUN_ID", "run:env")

    task_root = tmp_path / "vec_add" / "task_explicit"
    run_root = task_root / "runs" / "run_explicit"

    assert (
        get_ksearch_run_dir(
            base_dir=tmp_path,
            task_name="vec/add",
            task_id="task:explicit",
            run_id="run:explicit",
        )
        == run_root.resolve()
    )
    assert (
        get_ksearch_artifacts_dir(
            base_dir=tmp_path,
            task_name="vec/add",
            task_id="task:explicit",
            run_id="run:explicit",
        )
        == run_root.resolve() / "artifacts"
    )
    assert (
        get_ksearch_artifacts_dir(
            base_dir=tmp_path,
            task_name="vec/add",
            task_id="task:explicit",
            include_run=False,
        )
        == task_root.resolve() / "artifacts"
    )
    assert (
        get_run_logs_dir(
            base_dir=tmp_path,
            task_name="vec/add",
            task_id="task:explicit",
            run_id="run:explicit",
            sub="telemetry",
        )
        == run_root.resolve() / "logs" / "telemetry"
    )
    assert (
        get_ksearch_worktrees_dir(
            base_dir=tmp_path,
            task_name="vec/add",
            task_id="task:explicit",
            run_id="run:explicit",
        )
        == run_root.resolve() / "worktrees"
    )


def test_safe_path_component_basics():
    assert safe_path_component("", default="fallback") == "fallback"
    assert safe_path_component("a/b c", default="x") == "a_b_c"
    assert len(safe_path_component("a" * 200, default="x")) == 96
