from pathlib import Path

from k_search.kernel_generators.kernel_generator_world_model import (
    WorldModelKernelGeneratorWithBaseline,
    resolve_strategy_catalog_entry,
)
from k_search.kernel_generators.strategy_injection import StrategyCatalogEntry
from k_search.kernel_generators.world_model import dump_world_model_obj, load_world_model_obj
from k_search.kernel_generators.world_model_manager import WorldModelManager


def _entry(tmp_path: Path, sid: str = "s1") -> StrategyCatalogEntry:
    md = tmp_path / f"{sid}.md"
    md.write_text(f"# {sid}\n\nFull strategy.", encoding="utf-8")
    return StrategyCatalogEntry(
        id=sid,
        title=f"Strategy {sid}",
        summary=f"Summary {sid}",
        markdown_ref=f"{sid}.md",
        markdown_path=md,
    )


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
