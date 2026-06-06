from pathlib import Path

from k_search.kernel_generators.memory import CODE_MAP, MemoryKind, MemoryStore


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(artifacts_dir=str(tmp_path / "artifacts"), task_name="opx")


def test_load_missing_returns_none(tmp_path):
    assert _store(tmp_path).load(CODE_MAP) is None


def test_save_then_load_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.save(CODE_MAP, "# CODE_MAP\nhello\n")
    assert store.load(CODE_MAP) == "# CODE_MAP\nhello\n"


def test_save_uses_task_artifacts_root_not_run_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-one")
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-one")
    store = MemoryStore(artifacts_dir=tmp_path / "out", task_name="opx")

    store.save(CODE_MAP, "# CODE_MAP\nhello\n")

    task_memory = tmp_path / "out" / "opx" / "task-one" / "artifacts" / "memory" / "code_map" / "CODE_MAP.md"
    run_memory = (
        tmp_path
        / "out"
        / "opx"
        / "task-one"
        / "runs"
        / "run-one"
        / "artifacts"
        / "memory"
        / "code_map"
        / "CODE_MAP.md"
    )
    assert task_memory.is_file()
    assert not run_memory.exists()


def test_save_empty_is_noop(tmp_path):
    store = _store(tmp_path)
    store.save(CODE_MAP, "   ")
    assert store.load(CODE_MAP) is None


def test_materialize_and_read_from_worktree(tmp_path):
    store = _store(tmp_path)
    project = tmp_path / "wt"
    project.mkdir()
    assert store.materialize(CODE_MAP, project) is False  # nothing saved yet
    store.save(CODE_MAP, "mapped\n")
    assert store.materialize(CODE_MAP, project) is True
    assert (project / "CODE_MAP.md").read_text(encoding="utf-8") == "mapped\n"
    assert store.read_from_worktree(CODE_MAP, project) == "mapped\n"


def test_kind_is_generic(tmp_path):
    plan = MemoryKind("plan", "PLAN.md", gated_writeback=True)
    store = _store(tmp_path)
    store.save(plan, "step 1\n")
    assert store.load(plan) == "step 1\n"
    assert store.load(CODE_MAP) is None  # kinds are isolated


def test_save_code_map_if_adopted_only_on_new_best(tmp_path):
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore, save_code_map_if_adopted

    class _Task:
        artifacts_dir = str(tmp_path / "artifacts")
        definition_name = "opx"

    task = _Task()
    save_code_map_if_adopted(task=task, code_map_text="A\n", adopted=False)
    assert MemoryStore.for_task(task).load(CODE_MAP) is None
    save_code_map_if_adopted(task=task, code_map_text="B\n", adopted=True)
    assert MemoryStore.for_task(task).load(CODE_MAP) == "B\n"
    save_code_map_if_adopted(task=task, code_map_text=None, adopted=True)
    assert MemoryStore.for_task(task).load(CODE_MAP) == "B\n"
