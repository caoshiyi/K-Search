"""Tests for markdown-backed natural-language strategy injection."""

import argparse
import json
from pathlib import Path

import pytest

from k_search.kernel_generators.kernel_generator_world_model import (
    WorldModelKernelGeneratorWithBaseline,
)
from k_search.kernel_generators.strategy_injection import (
    StrategyCatalogEntry,
    _build_action_node,
    _render_api_references_section,
    _render_anti_patterns_section,
    build_wm_from_strategies,
    load_strategy_catalog,
    render_strategy_action_text,
    render_strategy_as_action_text,
)
from k_search.kernel_generators.world_model import render_chosen_action_node_block


def _write_catalog(
    tmp_path: Path,
    *,
    markdown_ref: str = "strategies/ub_reuse.md",
    entry_overrides: dict | None = None,
    catalog_overrides: dict | None = None,
) -> Path:
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "ub_reuse.md").write_text(
        "# Reuse UB cache\n\n"
        "Full strategy body here.\n\n"
        "Implementation checklist:\n"
        "1. Identify repeated GM reads.\n"
        "2. Check UB capacity.\n",
        encoding="utf-8",
    )
    entry = {
        "id": "ub_reuse",
        "title": "Reuse UB cache",
        "summary": "Cache reused tiles in UB to reduce GM traffic.",
        "markdown_ref": markdown_ref,
        "tags": ["memory", "ub"],
        "difficulty_1_to_5": 3,
        "score_0_to_1": 0.7,
        "expected_vs_baseline_factor": 1.05,
        "requires": [],
        "allow_reexecute": False,
    }
    if entry_overrides:
        entry.update(entry_overrides)
    catalog = {
        "version": 2,
        "strategy_form": "natural_language",
        "strategies": [entry],
    }
    if catalog_overrides:
        catalog.update(catalog_overrides)
    catalog_path = tmp_path / "strategy_catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog_path


class TestStrategyCatalogLoading:
    def test_valid_catalog_loads_and_resolves_markdown_ref(self, tmp_path):
        catalog_path = _write_catalog(tmp_path)

        entries = load_strategy_catalog(catalog_path)

        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, StrategyCatalogEntry)
        assert entry.id == "ub_reuse"
        assert entry.title == "Reuse UB cache"
        assert entry.summary == "Cache reused tiles in UB to reduce GM traffic."
        assert entry.markdown_ref == "strategies/ub_reuse.md"
        assert entry.markdown_path == (tmp_path / "strategies" / "ub_reuse.md").resolve()
        assert entry.tags == ("memory", "ub")
        assert entry.difficulty_1_to_5 == 3
        assert entry.score_0_to_1 == 0.7
        assert entry.expected_vs_baseline_factor == 1.05
        assert entry.requires == ()
        assert entry.allow_reexecute is False

    def test_catalog_loads_strategy_dependencies(self, tmp_path):
        catalog_path = _write_catalog(tmp_path)
        strategy_dir = tmp_path / "strategies"
        (strategy_dir / "pipeline.md").write_text("# Pipeline\n", encoding="utf-8")
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        data["strategies"].append(
            {
                "id": "soft_pipeline",
                "title": "Soft pipeline",
                "summary": "Pipeline after UB reuse.",
                "markdown_ref": "strategies/pipeline.md",
                "requires": ["ub_reuse"],
                "allow_reexecute": True,
            }
        )
        catalog_path.write_text(json.dumps(data), encoding="utf-8")

        entries = load_strategy_catalog(catalog_path)

        pipeline = next(entry for entry in entries if entry.id == "soft_pipeline")
        assert pipeline.requires == ("ub_reuse",)
        assert pipeline.allow_reexecute is True

    def test_unknown_strategy_dependency_rejected(self, tmp_path):
        catalog_path = _write_catalog(
            tmp_path,
            entry_overrides={"requires": ["missing_strategy"]},
        )

        with pytest.raises(ValueError, match="requires unknown strategy id: missing_strategy"):
            load_strategy_catalog(catalog_path)

    def test_strategy_dependency_cycle_rejected(self, tmp_path):
        catalog_path = _write_catalog(tmp_path, entry_overrides={"requires": ["soft_pipeline"]})
        strategy_dir = tmp_path / "strategies"
        (strategy_dir / "pipeline.md").write_text("# Pipeline\n", encoding="utf-8")
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        data["strategies"].append(
            {
                "id": "soft_pipeline",
                "title": "Soft pipeline",
                "summary": "Pipeline after UB reuse.",
                "markdown_ref": "strategies/pipeline.md",
                "requires": ["ub_reuse"],
            }
        )
        catalog_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="strategy dependency cycle detected"):
            load_strategy_catalog(catalog_path)

    def test_absolute_markdown_ref_rejected(self, tmp_path):
        absolute = str((tmp_path / "strategies" / "ub_reuse.md").resolve())
        catalog_path = _write_catalog(tmp_path, markdown_ref=absolute)

        with pytest.raises(ValueError, match="must be relative"):
            load_strategy_catalog(catalog_path)

    def test_parent_path_escape_rejected(self, tmp_path):
        outside = tmp_path.parent / "outside.md"
        outside.write_text("# outside\n", encoding="utf-8")
        catalog_path = _write_catalog(tmp_path, markdown_ref="../outside.md")

        with pytest.raises(ValueError, match="escapes catalog directory"):
            load_strategy_catalog(catalog_path)

    def test_non_md_markdown_ref_rejected(self, tmp_path):
        (tmp_path / "strategies").mkdir(parents=True, exist_ok=True)
        (tmp_path / "strategies" / "ub_reuse.txt").write_text("text", encoding="utf-8")
        catalog_path = _write_catalog(tmp_path, markdown_ref="strategies/ub_reuse.txt")

        with pytest.raises(ValueError, match=r"\.md"):
            load_strategy_catalog(catalog_path)

    def test_missing_markdown_file_rejected(self, tmp_path):
        catalog_path = _write_catalog(tmp_path, markdown_ref="strategies/missing.md")

        with pytest.raises(FileNotFoundError, match="strategy markdown file not found"):
            load_strategy_catalog(catalog_path)

    def test_duplicate_id_rejected(self, tmp_path):
        catalog_path = _write_catalog(tmp_path)
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        data["strategies"].append(dict(data["strategies"][0]))
        catalog_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="duplicate strategy id"):
            load_strategy_catalog(catalog_path)

    @pytest.mark.parametrize("field", ["structured_params", "dsl"])
    def test_structured_params_and_dsl_rejected(self, tmp_path, field):
        catalog_path = _write_catalog(tmp_path, entry_overrides={field: {"x": 1}})

        with pytest.raises(ValueError, match=f"unsupported field {field}"):
            load_strategy_catalog(catalog_path)

    def test_inline_natural_language_rejected_by_default(self, tmp_path):
        catalog_path = _write_catalog(
            tmp_path,
            entry_overrides={"natural_language": "Full inline strategy text."},
        )

        with pytest.raises(ValueError, match="inline natural_language"):
            load_strategy_catalog(catalog_path)

    def test_inline_natural_language_can_be_loaded_with_explicit_compat_env(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("KSEARCH_ALLOW_INLINE_STRATEGY", "1")
        catalog_path = tmp_path / "legacy_catalog.json"
        catalog_path.write_text(
            json.dumps(
                {
                    "strategy_catalog": [
                        {
                            "id": "inline_strategy",
                            "name": "Inline strategy",
                            "summary": "Legacy inline summary.",
                            "natural_language": "Legacy full inline body.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        entry = load_strategy_catalog(catalog_path)[0]
        rendered = render_strategy_action_text(entry=entry)

        assert entry.id == "inline_strategy"
        assert entry.title == "Inline strategy"
        assert "Legacy inline summary." in rendered
        assert "Legacy full inline body." in rendered


class TestStrategyActionRendering:
    def test_render_strategy_action_text_includes_full_markdown(self, tmp_path):
        entry = load_strategy_catalog(_write_catalog(tmp_path))[0]

        rendered = render_strategy_action_text(entry=entry)

        assert "Strategy ID: ub_reuse" in rendered
        assert "Strategy title: Reuse UB cache" in rendered
        assert "Strategy summary: Cache reused tiles in UB" in rendered
        assert "Strategy tags: memory, ub" in rendered
        assert "Full natural-language strategy markdown:" in rendered
        assert "Implementation checklist" in rendered

    def test_render_strategy_as_action_text_rejects_old_forms(self, tmp_path):
        entry = load_strategy_catalog(_write_catalog(tmp_path))[0]

        with pytest.raises(ValueError, match="Only natural_language"):
            render_strategy_as_action_text(entry, form="dsl")

    def test_render_strategy_action_text_truncates_long_markdown(self, tmp_path):
        entry = load_strategy_catalog(_write_catalog(tmp_path))[0]

        rendered = render_strategy_action_text(entry=entry, max_markdown_chars=12)

        assert "# Reuse UB c" in rendered
        assert "[truncated strategy markdown]" in rendered


class TestWorldModelStrategySeeding:
    def test_world_model_seed_contains_summary_but_not_full_markdown(self, tmp_path):
        entry = load_strategy_catalog(_write_catalog(tmp_path))[0]

        wm = build_wm_from_strategies(
            [entry],
            definition_name="mqa",
            kernel_summary="MQA kernel.",
        )
        dumped = json.dumps(wm)

        assert "Cache reused tiles in UB to reduce GM traffic." in dumped
        assert "Implementation checklist" not in dumped
        action_node = wm["decision_tree"]["nodes"][1]
        assert action_node["action"]["strategy_ref"] == {
            "id": "ub_reuse",
            "markdown_ref": "strategies/ub_reuse.md",
        }

    def test_build_action_node_uses_entry_metadata(self, tmp_path):
        entry = load_strategy_catalog(_write_catalog(tmp_path))[0]

        node = _build_action_node(0, entry, "natural_language")

        assert node["action"]["title"] == "Reuse UB cache"
        assert node["action"]["description"] == "Cache reused tiles in UB to reduce GM traffic."
        assert node["action"]["score_0_to_1"] == 0.7
        assert node["action"]["difficulty_1_to_5"] == 3
        assert node["action"]["expected_speedup"] == {
            "factor": 1.05,
            "relative_to": "unspecified_legacy",
            "source": "legacy_expected_vs_baseline_factor",
        }
        assert "expected_vs_baseline_factor" not in node["action"]
        assert "Implementation checklist" not in json.dumps(node)

    def test_chosen_action_prompt_includes_referenced_markdown(self, tmp_path):
        entry = load_strategy_catalog(_write_catalog(tmp_path))[0]
        wm = build_wm_from_strategies([entry], definition_name="mqa")
        node = wm["decision_tree"]["nodes"][1]
        generator = object.__new__(WorldModelKernelGeneratorWithBaseline)
        generator._strategy_catalog = [entry]

        chosen_action_text = render_chosen_action_node_block(node).strip()
        strategy_text = generator._strategy_text_for_node(node)
        final_action_text = (
            chosen_action_text
            + "\n\nReferenced strategy document:\n"
            + strategy_text
        )

        assert "Cache reused tiles in UB to reduce GM traffic." in chosen_action_text
        assert "Implementation checklist" not in chosen_action_text
        assert "Implementation checklist" in final_action_text


class TestRenderApiReferencesSection:
    """Tests for _render_api_references_section standalone function."""

    def test_empty_list_returns_empty_string(self):
        result = _render_api_references_section([])
        assert result == ""

    def test_single_api_reference(self):
        refs = [
            {
                "api_name": "DataCopy",
                "doc_path": "ascendc/api/DataCopy.md",
                "summary": "DataCopy(src, dst, len) - copy data between buffers; alignment must be 32 bytes",
            },
        ]
        result = _render_api_references_section(refs)
        assert "=== AscendC API Reference ===" in result
        assert "DataCopy" in result
        assert "DataCopy(src, dst, len)" in result
        assert "references/ascendc/api/DataCopy.md" in result


class TestRenderAntiPatternsSection:
    """Tests for _render_anti_patterns_section standalone function."""

    def test_no_anti_patterns_returns_empty_string(self):
        refs = [
            {
                "api_name": "DataCopy",
                "doc_path": "ascendc/api/DataCopy.md",
                "summary": "DataCopy(src, dst, len)",
                "anti_patterns": [],
            },
        ]
        result = _render_anti_patterns_section(refs, "S01")
        assert result == ""

    def test_single_anti_pattern(self):
        refs = [
            {
                "api_name": "DataCopy",
                "doc_path": "ascendc/api/DataCopy.md",
                "summary": "DataCopy(src, dst, len)",
                "anti_patterns": [
                    {
                        "id": "ap1",
                        "pattern": "DataCopy inside a loop",
                        "reason": "Causes redundant UB-to-UB transfers",
                        "source": "ascendc/api/DataCopy.md",
                    },
                ],
            },
        ]
        result = _render_anti_patterns_section(refs, "S01")
        assert "=== Anti-Pattern Warnings ===" in result
        assert "[AP-S01-ap1]" in result
        assert "Do NOT use DataCopy inside a loop" in result
        assert "Causes redundant UB-to-UB transfers" in result
        assert "Discovered from ascendc/api/DataCopy.md" in result
