"""Tests for implementation_priorities rendering."""

from k_search.kernel_generators.strategy_injection import (
    StrategyCatalogEntry,
    _build_action_node,
    _render_priorities_section,
)


def test_render_priorities_section_basic():
    """Test basic rendering with P0, P1, P2 priorities."""
    priorities = [
        {"priority": "P0", "action": "Change tile size constants", "dependency": None},
        {"priority": "P1", "action": "Adjust buffer allocations", "dependency": "P0 verified"},
        {"priority": "P2", "action": "Reduce loop overhead", "dependency": "P0+P1 verified"},
    ]
    result = _render_priorities_section(priorities)

    assert "=== Implementation Priority ===" in result
    assert "P0: Change tile size constants" in result
    assert "P1: Adjust buffer allocations" in result
    assert "P2: Reduce loop overhead" in result
    assert "Requires: P0 verified" in result
    assert "Requires: P0+P1 verified" in result
    assert "Recommended 1-2 files per round" in result
    assert "Do NOT combine P0+P1+P2" in result


def test_render_priorities_section_empty_list():
    """Test with empty priorities list returns empty string."""
    result = _render_priorities_section([])
    assert result == ""


def test_render_priorities_section_no_dependencies():
    """Test rendering without dependencies."""
    priorities = [
        {"priority": "P0", "action": "Single step", "dependency": None},
    ]
    result = _render_priorities_section(priorities)

    assert "P0: Single step" in result
    assert "Requires:" not in result


def test_render_priorities_section_missing_fields():
    """Test rendering with missing optional fields."""
    priorities = [
        {"priority": "P0"},  # missing action
    ]
    result = _render_priorities_section(priorities)

    assert "P0: unknown action" in result


# ==============================================================================
# Tests for _build_action_node with new natural-language catalog entries
# ==============================================================================


def _entry(**overrides):
    entry = StrategyCatalogEntry(
        id="S1",
        title="Test",
        summary="Summary description",
        markdown_ref="strategies/s1.md",
        markdown_path=__file__,
        tags=("tiling",),
        difficulty_1_to_5=2,
        score_0_to_1=0.8,
        expected_vs_baseline_factor=1.47,
    )
    return StrategyCatalogEntry(
        id=overrides.get("id", entry.id),
        title=overrides.get("title", entry.title),
        summary=overrides.get("summary", entry.summary),
        markdown_ref=overrides.get("markdown_ref", entry.markdown_ref),
        markdown_path=overrides.get("markdown_path", entry.markdown_path),
        tags=overrides.get("tags", entry.tags),
        difficulty_1_to_5=overrides.get("difficulty_1_to_5", entry.difficulty_1_to_5),
        score_0_to_1=overrides.get("score_0_to_1", entry.score_0_to_1),
        expected_vs_baseline_factor=overrides.get(
            "expected_vs_baseline_factor",
            entry.expected_vs_baseline_factor,
        ),
    )


def test_build_action_node_with_expected_vs_baseline_factor():
    """Test that expected speedup uses entry.expected_vs_baseline_factor."""
    node = _build_action_node(0, _entry(), "natural_language")

    action = node.get("action", {})
    expected_speedup = action.get("expected_vs_baseline_factor")

    assert expected_speedup == 1.47


def test_build_action_node_without_expected_vs_baseline_factor():
    """Test that expected_speedup is None when metadata is missing."""
    node = _build_action_node(
        1,
        _entry(id="S2", expected_vs_baseline_factor=None, score_0_to_1=0.5),
        "natural_language",
    )

    action = node.get("action", {})
    expected_speedup = action.get("expected_vs_baseline_factor")

    assert expected_speedup is None


def test_build_action_node_records_strategy_ref():
    """Test action node keeps the markdown reference, not full strategy text."""
    node = _build_action_node(2, _entry(id="S3", markdown_ref="strategies/s3.md"), "natural_language")

    action = node.get("action", {})

    assert action["strategy_ref"] == {
        "id": "S3",
        "markdown_ref": "strategies/s3.md",
    }


def test_build_action_node_rejects_non_natural_language_form():
    """Test that old strategy forms fail fast."""
    import pytest

    with pytest.raises(ValueError, match="Only natural_language"):
        _build_action_node(3, _entry(id="S4"), "dsl")
