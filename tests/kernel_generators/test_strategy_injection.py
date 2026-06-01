"""Tests for strategy_injection API reference and anti-pattern rendering."""

import pytest

from k_search.kernel_generators.strategy_injection import (
    _build_action_node,
    _render_api_references_section,
    _render_anti_patterns_section,
    render_strategy_as_action_text,
)


# ---------------------------------------------------------------------------
# _render_api_references_section
# ---------------------------------------------------------------------------

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

    def test_multiple_api_references(self):
        refs = [
            {
                "api_name": "DataCopy",
                "doc_path": "ascendc/api/DataCopy.md",
                "summary": "DataCopy(src, dst, len) - copy between buffers",
            },
            {
                "api_name": "LocalToGlobal",
                "doc_path": "ascendc/api/LocalToGlobal.md",
                "summary": "LocalToGlobal(dst, src) - move from UB to GM",
            },
        ]
        result = _render_api_references_section(refs)
        assert "DataCopy" in result
        assert "LocalToGlobal" in result
        assert "references/ascendc/api/DataCopy.md" in result
        assert "references/ascendc/api/LocalToGlobal.md" in result

    def test_doc_path_hint_present(self):
        refs = [
            {
                "api_name": "DataCopy",
                "doc_path": "ascendc/api/DataCopy.md",
                "summary": "DataCopy(src, dst, len) - copy between buffers",
            },
        ]
        result = _render_api_references_section(refs)
        assert "If you need more details" in result


# ---------------------------------------------------------------------------
# _render_anti_patterns_section
# ---------------------------------------------------------------------------

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

    def test_no_anti_patterns_key_returns_empty_string(self):
        refs = [
            {
                "api_name": "DataCopy",
                "doc_path": "ascendc/api/DataCopy.md",
                "summary": "DataCopy(src, dst, len)",
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
                        "reason": "Causes redundant UB-to-UB transfers; use batch copy instead",
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

    def test_multiple_anti_patterns_across_apis(self):
        refs = [
            {
                "api_name": "DataCopy",
                "doc_path": "ascendc/api/DataCopy.md",
                "summary": "DataCopy(src, dst, len)",
                "anti_patterns": [
                    {
                        "id": "ap1",
                        "pattern": "DataCopy inside a loop",
                        "reason": "Causes redundant transfers",
                        "source": "ascendc/api/DataCopy.md",
                    },
                ],
            },
            {
                "api_name": "LocalToGlobal",
                "doc_path": "ascendc/api/LocalToGlobal.md",
                "summary": "LocalToGlobal(dst, src)",
                "anti_patterns": [
                    {
                        "id": "ap2",
                        "pattern": "LocalToGlobal on misaligned address",
                        "reason": "Address must be 32-byte aligned",
                        "source": "ascendc/api/LocalToGlobal.md",
                    },
                ],
            },
        ]
        result = _render_anti_patterns_section(refs, "S02")
        assert "[AP-S02-ap1]" in result
        assert "[AP-S02-ap2]" in result
        assert "DataCopy inside a loop" in result
        assert "LocalToGlobal on misaligned address" in result

    def test_anti_pattern_id_format(self):
        refs = [
            {
                "api_name": "DataCopy",
                "doc_path": "ascendc/api/DataCopy.md",
                "summary": "DataCopy(src, dst, len)",
                "anti_patterns": [
                    {
                        "id": "ap3",
                        "pattern": "bad pattern",
                        "reason": "bad reason",
                        "source": "source.md",
                    },
                ],
            },
        ]
        result = _render_anti_patterns_section(refs, "S07")
        assert "[AP-S07-ap3]" in result


# ---------------------------------------------------------------------------
# render_strategy_as_action_text with api_references
# ---------------------------------------------------------------------------

class TestRenderStrategyWithApiReferences:
    """Tests for render_strategy_as_action_text when api_references are present."""

    def _make_strategy(self, api_references=None, **overrides):
        base = {
            "id": "S01",
            "name": "Vectorize inner loop",
            "category": "compute",
            "difficulty": 3,
            "impact": "medium",
            "natural_language": "Replace the scalar inner loop with vector operations for better throughput.",
        }
        if api_references is not None:
            base["api_references"] = api_references
        base.update(overrides)
        return base

    def test_strategy_without_api_references_rendered_unchanged(self):
        strategy = self._make_strategy()
        result = render_strategy_as_action_text(strategy, form="natural_language")
        assert "=== AscendC API Reference ===" not in result
        assert "=== Anti-Pattern Warnings ===" not in result
        assert "Strategy S01" in result

    def test_strategy_with_api_references_includes_both_sections(self):
        strategy = self._make_strategy(
            api_references=[
                {
                    "api_name": "DataCopy",
                    "doc_path": "ascendc/api/DataCopy.md",
                    "summary": "DataCopy(src, dst, len) - copy between buffers",
                    "anti_patterns": [
                        {
                            "id": "ap1",
                            "pattern": "DataCopy inside a loop",
                            "reason": "Causes redundant transfers",
                            "source": "ascendc/api/DataCopy.md",
                        },
                    ],
                },
            ],
        )
        result = render_strategy_as_action_text(strategy, form="natural_language")
        assert "=== AscendC API Reference ===" in result
        assert "=== Anti-Pattern Warnings ===" in result
        assert "Strategy S01" in result
        assert "DataCopy" in result

    def test_anti_pattern_section_only_when_anti_patterns_exist(self):
        strategy = self._make_strategy(
            api_references=[
                {
                    "api_name": "DataCopy",
                    "doc_path": "ascendc/api/DataCopy.md",
                    "summary": "DataCopy(src, dst, len) - copy between buffers",
                    "anti_patterns": [],
                },
            ],
        )
        result = render_strategy_as_action_text(strategy, form="natural_language")
        assert "=== AscendC API Reference ===" in result
        assert "=== Anti-Pattern Warnings ===" not in result

    def test_api_references_with_structured_params_form(self):
        strategy = {
            "id": "S02",
            "name": "Tiling optimization",
            "category": "tiling",
            "difficulty": 4,
            "impact": "high",
            "structured_params": {"tile_size": 128, "alignment": 32},
            "api_references": [
                {
                    "api_name": "DataCopy",
                    "doc_path": "ascendc/api/DataCopy.md",
                    "summary": "DataCopy(src, dst, len)",
                    "anti_patterns": [
                        {
                            "id": "ap1",
                            "pattern": "misaligned copy",
                            "reason": "alignment required",
                            "source": "doc.md",
                        },
                    ],
                },
            ],
        }
        result = render_strategy_as_action_text(strategy, form="structured_params")
        assert "=== AscendC API Reference ===" in result
        assert "=== Anti-Pattern Warnings ===" in result

    def test_api_references_with_dsl_form(self):
        strategy = {
            "id": "S03",
            "name": "Pipeline DSL",
            "category": "pipeline",
            "difficulty": 5,
            "impact": "high",
            "dsl": "PIPELINE(stage1, stage2) -> fused_compute",
            "api_references": [
                {
                    "api_name": "Pipeline",
                    "doc_path": "ascendc/api/Pipeline.md",
                    "summary": "Pipeline(stages) - multi-stage compute",
                },
            ],
        }
        result = render_strategy_as_action_text(strategy, form="dsl")
        assert "=== AscendC API Reference ===" in result
        assert "=== Anti-Pattern Warnings ===" not in result

    def test_doc_path_hint_in_rendered_output(self):
        strategy = self._make_strategy(
            api_references=[
                {
                    "api_name": "DataCopy",
                    "doc_path": "ascendc/api/DataCopy.md",
                    "summary": "DataCopy(src, dst, len)",
                },
            ],
        )
        result = render_strategy_as_action_text(strategy, form="natural_language")
        assert "If you need more details" in result

    def test_existing_value_error_handling_preserved(self):
        """Verify that existing ValueError for missing fields still works."""
        strategy = {"id": "S04", "name": "No content", "category": "compute"}
        with pytest.raises(ValueError, match="no natural_language field"):
            render_strategy_as_action_text(strategy, form="natural_language")


# ---------------------------------------------------------------------------
# _build_action_node
# ---------------------------------------------------------------------------


class TestBuildActionNode:
    """Tests for _build_action_node expected_speedup handling."""

    def test_build_action_node_with_min_none(self):
        """Test that min=None does not cause TypeError and returns None."""
        strategy = {
            "id": "S5",
            "name": "Min None",
            "category": "tiling",
            "impact": "medium",
            "difficulty": 2,
            "natural_language": "Description",
            "structured_params": {
                "expected_speedup": {"min": None},
            },
        }
        node = _build_action_node(4, strategy, "natural_language")

        action = node.get("action", {})
        expected_speedup = action.get("expected_vs_baseline_factor")

        # min=None 应被跳过，expected_speedup 为 None
        assert expected_speedup is None

    def test_build_action_node_with_valid_min(self):
        """Test that valid min value is correctly extracted."""
        strategy = {
            "id": "S6",
            "name": "Valid Min",
            "category": "tiling",
            "impact": "high",
            "difficulty": 3,
            "natural_language": "Description",
            "structured_params": {
                "expected_speedup": {"min": 2.5},
            },
        }
        node = _build_action_node(5, strategy, "natural_language")

        action = node.get("action", {})
        expected_speedup = action.get("expected_vs_baseline_factor")

        assert expected_speedup == 2.5

    def test_build_action_node_with_expected_speedup_interval(self):
        """Test that expected_speedup_interval.likely takes priority."""
        strategy = {
            "id": "S7",
            "name": "Interval Priority",
            "category": "compute",
            "impact": "high",
            "difficulty": 4,
            "natural_language": "Description",
            "expected_speedup_interval": {"likely": 3.0},
            "structured_params": {
                "expected_speedup": {"min": 2.0},
            },
        }
        node = _build_action_node(6, strategy, "natural_language")

        action = node.get("action", {})
        expected_speedup = action.get("expected_vs_baseline_factor")

        # expected_speedup_interval.likely takes priority over structured_params.expected_speedup.min
        assert expected_speedup == 3.0

    def test_build_action_node_with_missing_min_key(self):
        """Test that missing min key returns None."""
        strategy = {
            "id": "S8",
            "name": "No Min Key",
            "category": "tiling",
            "impact": "low",
            "difficulty": 1,
            "natural_language": "Description",
            "structured_params": {
                "expected_speedup": {},  # no min key
            },
        }
        node = _build_action_node(7, strategy, "natural_language")

        action = node.get("action", {})
        expected_speedup = action.get("expected_vs_baseline_factor")

        assert expected_speedup is None