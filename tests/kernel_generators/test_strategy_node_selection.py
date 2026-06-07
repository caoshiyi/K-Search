import json
from pathlib import Path

from k_search.kernel_generators.kernel_generator_world_model import (
    WorldModelKernelGeneratorWithBaseline,
    resolve_strategy_catalog_entry,
)
from k_search.kernel_generators.strategy_injection import StrategyCatalogEntry
from k_search.kernel_generators.world_model import dump_world_model_obj, load_world_model_obj
from k_search.kernel_generators.world_model_manager import WorldModelManager


def _entry(
    tmp_path: Path,
    sid: str = "s1",
    *,
    score: float = 0.5,
    requires: tuple[str, ...] = (),
    allow_reexecute: bool = False,
) -> StrategyCatalogEntry:
    md = tmp_path / f"{sid}.md"
    md.write_text(f"# {sid}\n\nFull strategy.", encoding="utf-8")
    entry = StrategyCatalogEntry(
        id=sid,
        title=f"Strategy {sid}",
        summary=f"Summary {sid}",
        markdown_ref=f"{sid}.md",
        markdown_path=md,
        score_0_to_1=score,
    )
    object.__setattr__(entry, "requires", tuple(requires))
    object.__setattr__(entry, "allow_reexecute", bool(allow_reexecute))
    return entry


def test_child_node_without_strategy_file_does_not_resolve_to_catalog_entry(tmp_path):
    catalog = [_entry(tmp_path, "s1"), _entry(tmp_path, "s2")]
    node = {
        "node_id": "s2c1",
        "parent_id": "s2",
        "action": {
            "title": "GQA-aware reuse",
            "score_0_to_1": 0.99,
        },
    }

    assert resolve_strategy_catalog_entry(node, catalog) is None


def test_catalog_backed_node_is_executable(tmp_path):
    catalog = [_entry(tmp_path, "fa_qkv_two_level_l1_reuse")]
    node = {
        "node_id": "s1",
        "parent_id": "root",
        "action": {
            "title": "Two-level L1 reuse",
            "strategy_ref": {"id": "fa_qkv_two_level_l1_reuse"},
        },
    }

    entry = resolve_strategy_catalog_entry(node, catalog)

    assert entry is not None
    assert entry.id == "fa_qkv_two_level_l1_reuse"
    assert entry.markdown_ref.endswith(".md")


def test_strategy_selection_blocks_title_only_child_and_selects_catalog_node(tmp_path):
    catalog = [_entry(tmp_path, "s1"), _entry(tmp_path, "s2")]
    wm_obj = {
        "decision_tree": {
            "root_id": "root",
            "nodes": [
                {"node_id": "root"},
                {
                    "node_id": "s1",
                    "parent_id": "root",
                    "action": {
                        "title": "Strategy s1",
                        "score_0_to_1": 0.2,
                        "strategy_ref": {"id": "s1", "markdown_ref": "s1.md"},
                    },
                },
                {
                    "node_id": "s2",
                    "parent_id": "root",
                    "action": {
                        "title": "Strategy s2",
                        "score_0_to_1": 0.9,
                        "strategy_ref": {"id": "s2", "markdown_ref": "s2.md"},
                    },
                    "solution_ref": {"solution_id": "candidate-s2"},
                },
                {
                    "node_id": "s2c1",
                    "parent_id": "s2",
                    "action": {
                        "title": "Title-only child action",
                        "score_0_to_1": 0.99,
                    },
                },
            ],
        }
    }
    generator = object.__new__(WorldModelKernelGeneratorWithBaseline)
    generator._strategy_catalog = catalog
    generator._wm = WorldModelManager(llm_call=lambda prompt: "", target_gpu="ascend", language="ascendc")
    generator._wm.set("task", dump_world_model_obj(wm_obj))

    selected, blocked = generator._choose_executable_strategy_action_node_id(definition_name="task")

    assert selected == "s1"
    assert blocked == [
        {
            "node_id": "s2c1",
            "reason": "strategy_file_required_but_missing",
            "policy": "only_catalog_backed_strategy_nodes_are_executable",
        }
    ]
    updated = load_world_model_obj(generator._wm.get("task") or "")
    child = next(node for node in updated["decision_tree"]["nodes"] if node["node_id"] == "s2c1")
    assert child["action"]["status"] == "blocked"
    assert "strategy_file_required_but_missing" in child["notes"]


def test_strategy_selection_blocks_dependency_until_prerequisite_adopted(tmp_path):
    catalog = [
        _entry(tmp_path, "fa_qkv_two_level_l1_reuse", score=0.7),
        _entry(
            tmp_path,
            "fa_multibuffer_soft_pipeline",
            score=0.95,
            requires=("fa_qkv_two_level_l1_reuse",),
        ),
    ]
    wm_obj = {
        "decision_tree": {
            "root_id": "root",
            "nodes": [
                {"node_id": "root"},
                {
                    "node_id": "s1",
                    "parent_id": "root",
                    "action": {
                        "title": "QKV two-level tiling with L1 reuse",
                        "score_0_to_1": 0.7,
                        "strategy_ref": {
                            "id": "fa_qkv_two_level_l1_reuse",
                            "markdown_ref": "fa_qkv_two_level_l1_reuse.md",
                        },
                    },
                },
                {
                    "node_id": "s2",
                    "parent_id": "root",
                    "action": {
                        "title": "Multi-buffer soft pipeline after two-level tiling",
                        "score_0_to_1": 0.95,
                        "strategy_ref": {
                            "id": "fa_multibuffer_soft_pipeline",
                            "markdown_ref": "fa_multibuffer_soft_pipeline.md",
                        },
                    },
                },
            ],
        }
    }
    generator = object.__new__(WorldModelKernelGeneratorWithBaseline)
    generator._strategy_catalog = catalog
    generator._wm = WorldModelManager(llm_call=lambda prompt: "", target_gpu="ascend", language="ascendc")
    generator._wm.set("task", dump_world_model_obj(wm_obj))

    selected, blocked = generator._choose_executable_strategy_action_node_id(definition_name="task")

    assert selected == "s1"
    assert blocked == [
        {
            "node_id": "s2",
            "strategy_id": "fa_multibuffer_soft_pipeline",
            "reason": "missing_prerequisites",
            "missing": ["fa_qkv_two_level_l1_reuse"],
            "policy": "hard_strategy_dependency_gating",
        }
    ]


def test_strategy_selection_persists_blocked_actions_and_strategy_state(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-id")
    catalog = [
        _entry(tmp_path, "fa_qkv_two_level_l1_reuse", score=0.7),
        _entry(
            tmp_path,
            "fa_multibuffer_soft_pipeline",
            score=0.95,
            requires=("fa_qkv_two_level_l1_reuse",),
        ),
    ]
    wm_obj = {
        "decision_tree": {
            "root_id": "root",
            "nodes": [
                {"node_id": "root"},
                {
                    "node_id": "s1",
                    "parent_id": "root",
                    "action": {
                        "title": "QKV two-level tiling with L1 reuse",
                        "score_0_to_1": 0.7,
                        "strategy_ref": {"id": "fa_qkv_two_level_l1_reuse"},
                    },
                },
                {
                    "node_id": "s2",
                    "parent_id": "root",
                    "action": {
                        "title": "Multi-buffer soft pipeline after two-level tiling",
                        "score_0_to_1": 0.95,
                        "strategy_ref": {"id": "fa_multibuffer_soft_pipeline"},
                    },
                },
            ],
        }
    }
    generator = object.__new__(WorldModelKernelGeneratorWithBaseline)
    generator._strategy_catalog = catalog
    generator._artifacts_dir = str(tmp_path / "artifacts")
    generator._wm = WorldModelManager(llm_call=lambda prompt: "", target_gpu="ascend", language="ascendc")
    generator._wm.set("task", dump_world_model_obj(wm_obj))

    class Task:
        name = "task"

    selected, blocked = generator._choose_executable_strategy_action_node_id(
        definition_name="task",
        task=Task(),
        run_id="run-id",
        round_index=1,
    )

    assert selected == "s1"
    assert blocked[0]["reason"] == "missing_prerequisites"
    world_model_dir = tmp_path / "artifacts" / "task" / "task-id" / "runs" / "run-id" / "artifacts" / "world_model"
    blocked_lines = (world_model_dir / "blocked_actions.jsonl").read_text(encoding="utf-8").splitlines()
    blocked_event = json.loads(blocked_lines[0])
    assert blocked_event["action_node_id"] == "s2"
    assert blocked_event["reason"] == "missing_prerequisites"
    assert blocked_event["missing"] == ["fa_qkv_two_level_l1_reuse"]

    state = json.loads((world_model_dir / "strategy_state.json").read_text(encoding="utf-8"))
    assert state["adopted_strategy_ids"] == []
    assert state["blocked_actions"][0]["reason"] == "missing_prerequisites"


def test_strategy_selection_allows_dependency_after_prerequisite_adopted(tmp_path):
    catalog = [
        _entry(tmp_path, "fa_qkv_two_level_l1_reuse", score=0.7),
        _entry(
            tmp_path,
            "fa_multibuffer_soft_pipeline",
            score=0.95,
            requires=("fa_qkv_two_level_l1_reuse",),
        ),
    ]
    wm_obj = {
        "decision_tree": {
            "root_id": "root",
            "nodes": [
                {"node_id": "root"},
                {
                    "node_id": "s1",
                    "parent_id": "root",
                    "action": {
                        "title": "QKV two-level tiling with L1 reuse",
                        "score_0_to_1": 0.7,
                        "strategy_ref": {
                            "id": "fa_qkv_two_level_l1_reuse",
                            "markdown_ref": "fa_qkv_two_level_l1_reuse.md",
                        },
                    },
                    "solution_ref": {"solution_id": "round_0001_attempt_0001"},
                },
                {
                    "node_id": "s2",
                    "parent_id": "root",
                    "action": {
                        "title": "Multi-buffer soft pipeline after two-level tiling",
                        "score_0_to_1": 0.95,
                        "strategy_ref": {
                            "id": "fa_multibuffer_soft_pipeline",
                            "markdown_ref": "fa_multibuffer_soft_pipeline.md",
                        },
                    },
                },
            ],
        }
    }
    generator = object.__new__(WorldModelKernelGeneratorWithBaseline)
    generator._strategy_catalog = catalog
    generator._wm = WorldModelManager(llm_call=lambda prompt: "", target_gpu="ascend", language="ascendc")
    generator._wm.set("task", dump_world_model_obj(wm_obj))

    selected, blocked = generator._choose_executable_strategy_action_node_id(definition_name="task")

    assert selected == "s2"
    assert blocked == []
    updated = load_world_model_obj(generator._wm.get("task") or "")
    selected_node = next(node for node in updated["decision_tree"]["nodes"] if node["node_id"] == "s2")
    assert selected_node["parent_id"] == "s1"
