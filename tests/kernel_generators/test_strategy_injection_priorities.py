"""Tests for implementation_priorities rendering."""

import pytest
from k_search.kernel_generators.strategy_injection import (
    _render_priorities_section,
    render_strategy_as_action_text,
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
# Tests for render_strategy_as_action_text with implementation_priorities
# ==============================================================================


def test_render_strategy_with_priorities():
    """Test that priorities section appears after natural_language."""
    strategy = {
        "id": "S1",
        "name": "Test Strategy",
        "category": "tiling",
        "impact": "high",
        "difficulty": 2,
        "natural_language": "This is the strategy description.",
        "implementation_priorities": [
            {"priority": "P0", "action": "First step", "dependency": None},
            {"priority": "P1", "action": "Second step", "dependency": "P0 verified"},
        ],
    }
    result = render_strategy_as_action_text(strategy, form="natural_language")

    # 策略描述应该在前面
    strategy_desc_pos = result.find("Strategy S1: Test Strategy")
    priorities_pos = result.find("=== Implementation Priority ===")

    assert strategy_desc_pos < priorities_pos
    assert "This is the strategy description" in result
    assert "P0: First step" in result
    assert "P1: Second step" in result


def test_render_strategy_without_priorities():
    """Test that strategy without priorities still renders correctly."""
    strategy = {
        "id": "S2",
        "name": "No Priorities",
        "category": "compute",
        "impact": "medium",
        "difficulty": 1,
        "natural_language": "Simple strategy without priorities.",
    }
    result = render_strategy_as_action_text(strategy, form="natural_language")

    assert "=== Implementation Priority ===" not in result
    assert "Simple strategy without priorities" in result


def test_render_strategy_with_priorities_and_api_refs():
    """Test ordering: description -> priorities -> api_refs -> anti_patterns."""
    strategy = {
        "id": "S4",
        "name": "Complex Strategy",
        "category": "compute",
        "impact": "high",
        "difficulty": 2,
        "natural_language": "Strategy description.",
        "implementation_priorities": [
            {"priority": "P0", "action": "Step one", "dependency": None},
        ],
        "api_references": [
            {
                "api_name": "RowMuls",
                "doc_path": "some/path.md",
                "summary": "API summary",
            }
        ],
    }
    result = render_strategy_as_action_text(strategy, form="natural_language")

    # 检查顺序
    desc_pos = result.find("Strategy description")
    priorities_pos = result.find("=== Implementation Priority ===")
    api_ref_pos = result.find("=== AscendC API Reference ===")

    assert desc_pos < priorities_pos
    assert priorities_pos < api_ref_pos


# ==============================================================================
# Tests for _build_action_node with expected_speedup_interval
# ==============================================================================


def test_build_action_node_with_speedup_interval():
    """Test that expected_speedup uses 'likely' value from interval."""
    from k_search.kernel_generators.strategy_injection import _build_action_node

    strategy = {
        "id": "S1",
        "name": "Test",
        "category": "tiling",
        "impact": "high",
        "difficulty": 2,
        "natural_language": "Description",
        "expected_speedup_interval": {
            "min": 1.30,
            "likely": 1.47,
            "max": 1.50,
        },
    }
    node = _build_action_node(0, strategy, "natural_language")

    action = node.get("action", {})
    expected_speedup = action.get("expected_vs_baseline_factor")

    assert expected_speedup == 1.47


def test_build_action_node_without_speedup_interval():
    """Test that expected_speedup is None when interval is missing."""
    from k_search.kernel_generators.strategy_injection import _build_action_node

    strategy = {
        "id": "S2",
        "name": "No Interval",
        "category": "compute",
        "impact": "medium",
        "difficulty": 1,
        "natural_language": "Description",
    }
    node = _build_action_node(1, strategy, "natural_language")

    action = node.get("action", {})
    expected_speedup = action.get("expected_vs_baseline_factor")

    assert expected_speedup is None


def test_build_action_node_with_structured_params_speedup():
    """Test fallback to structured_params.expected_speedup.min when interval missing."""
    from k_search.kernel_generators.strategy_injection import _build_action_node

    strategy = {
        "id": "S3",
        "name": "Structured",
        "category": "tiling",
        "impact": "medium",
        "difficulty": 2,
        "natural_language": "Description",
        "structured_params": {
            "expected_speedup": {"min": 1.5, "max": 2.5},
        },
    }
    node = _build_action_node(2, strategy, "natural_language")

    action = node.get("action", {})
    expected_speedup = action.get("expected_vs_baseline_factor")

    assert expected_speedup == 1.5


def test_build_action_node_interval_overrides_structured_params():
    """Test that expected_speedup_interval takes precedence over structured_params."""
    from k_search.kernel_generators.strategy_injection import _build_action_node

    strategy = {
        "id": "S4",
        "name": "Both",
        "category": "tiling",
        "impact": "high",
        "difficulty": 2,
        "natural_language": "Description",
        "expected_speedup_interval": {
            "min": 1.30,
            "likely": 1.47,
            "max": 1.50,
        },
        "structured_params": {
            "expected_speedup": {"min": 1.5, "max": 2.5},
        },
    }
    node = _build_action_node(3, strategy, "natural_language")

    action = node.get("action", {})
    expected_speedup = action.get("expected_vs_baseline_factor")

    # expected_speedup_interval.likely should take precedence
    assert expected_speedup == 1.47