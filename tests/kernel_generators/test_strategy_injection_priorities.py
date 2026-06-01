"""Tests for implementation_priorities rendering."""

import pytest
from k_search.kernel_generators.strategy_injection import _render_priorities_section


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