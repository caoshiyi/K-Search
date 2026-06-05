from pathlib import Path

from k_search.utils.paths import (
    get_ksearch_artifacts_dir,
    get_run_id,
    get_run_logs_dir,
    resolve_output_base,
    safe_path_component,
)


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
    p = get_run_logs_dir(base_dir=tmp_path, task_name="vec/add", run_id="run:1", sub="llm")
    assert p == (tmp_path / "logs" / "vec_add" / "run_1" / "llm").resolve()
    # No sub -> run root.
    root = get_run_logs_dir(base_dir=tmp_path, task_name="vec/add", run_id="run:1")
    assert root == (tmp_path / "logs" / "vec_add" / "run_1").resolve()


def test_artifacts_dir_is_run_scoped_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("KSEARCH_RUN_ID", "run:1")
    p = get_ksearch_artifacts_dir(base_dir=tmp_path, task_name="vec/add")
    assert p == (tmp_path / "vec_add" / "runs" / "run_1").resolve()


def test_artifacts_dir_can_use_task_layout_without_run(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    p = get_ksearch_artifacts_dir(base_dir=tmp_path, task_name="vec/add", include_run=False)
    assert p == (tmp_path / "vec_add").resolve()


def test_safe_path_component_basics():
    assert safe_path_component("", default="fallback") == "fallback"
    assert safe_path_component("a/b c", default="x") == "a_b_c"
    assert len(safe_path_component("a" * 200, default="x")) == 96
