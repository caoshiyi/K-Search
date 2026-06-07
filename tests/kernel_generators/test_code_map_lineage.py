from k_search.kernel_generators.code_map_lineage import (
    CodeMapReuseContext,
    evaluate_code_map_reuse,
    should_reuse_code_map,
)
from k_search.kernel_generators.memory import CODE_MAP, MemoryStore, save_code_map_if_adopted


def test_unadopted_code_map_is_not_reused_for_new_action():
    meta = {
        "candidate_id": "round_0001_attempt_0001",
        "solution_id": "round_0001_attempt_0001",
        "branch_id": "root/s2",
        "adopted": False,
    }
    ctx = CodeMapReuseContext(mode="action", parent_solution_id="root", parent_branch_id="root")

    assert should_reuse_code_map(meta, ctx) is False
    decision = evaluate_code_map_reuse(meta, ctx)
    assert decision.reused is False
    assert decision.reason == "not_adopted_or_lineage_mismatch"


def test_repair_can_reuse_same_candidate_code_map():
    meta = {
        "candidate_id": "round_0002_attempt_0001",
        "branch_id": "root/s1",
        "adopted": False,
    }
    ctx = CodeMapReuseContext(
        mode="repair",
        current_candidate_id="round_0002_attempt_0001",
        branch_id="root/s1",
    )

    assert should_reuse_code_map(meta, ctx) is True


def test_new_action_can_reuse_only_adopted_parent_code_map():
    meta = {
        "solution_id": "sol_s1_adopted",
        "branch_id": "root/s1",
        "adopted": True,
    }
    ctx = CodeMapReuseContext(
        mode="action",
        parent_solution_id="sol_s1_adopted",
        parent_branch_id="root/s1",
    )

    assert should_reuse_code_map(meta, ctx) is True


def test_memory_store_materialize_code_map_requires_valid_lineage(tmp_path):
    task = type("Task", (), {"artifacts_dir": str(tmp_path / "artifacts"), "definition_name": "x"})()
    store = MemoryStore.for_task(task)
    store.save(
        CODE_MAP,
        "# CODE_MAP\npreseeded\n",
        meta={"schema_version": 1, "adopted": False, "candidate_id": "old", "branch_id": "root/s2"},
    )
    project = tmp_path / "project"
    project.mkdir()

    reused = store.materialize(
        CODE_MAP,
        project,
        code_map_reuse_context=CodeMapReuseContext(
            mode="action",
            parent_solution_id="sol_parent",
            parent_branch_id="root/s1",
        ),
    )

    assert reused is False
    assert not (project / "CODE_MAP.md").exists()


def test_save_code_map_if_adopted_writes_sidecar_meta(tmp_path):
    task = type("Task", (), {"artifacts_dir": str(tmp_path / "artifacts"), "definition_name": "x"})()

    save_code_map_if_adopted(
        task=task,
        code_map_text="# CODE_MAP\nadopted\n",
        adopted=True,
        solution_id="sol_s1_adopted",
        branch_id="root/s1",
        candidate_id="round_0001_attempt_0001",
        action_node_id="s1",
        strategy_id="fa_qkv_two_level_l1_reuse",
        eval_status="performance_measured",
        speedup_vs_parent=1.2,
    )

    store = MemoryStore.for_task(task)
    assert store.load(CODE_MAP) == "# CODE_MAP\nadopted\n"
    meta = store.load_meta(CODE_MAP)
    assert meta is not None
    assert meta["adopted"] is True
    assert meta["solution_id"] == "sol_s1_adopted"
    assert meta["branch_id"] == "root/s1"
    assert meta["strategy_id"] == "fa_qkv_two_level_l1_reuse"

    project = tmp_path / "project"
    project.mkdir()
    assert store.materialize(
        CODE_MAP,
        project,
        code_map_reuse_context=CodeMapReuseContext(
            mode="action",
            parent_solution_id="sol_s1_adopted",
            parent_branch_id="root/s1",
        ),
    )
    assert (project / "CODE_MAP.md").read_text(encoding="utf-8") == "# CODE_MAP\nadopted\n"
