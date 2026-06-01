"""Tests for anti-pattern learning logic in strategy_injection.

Covers _text_similarity_ratio and learn_anti_pattern_from_failure.
"""

import copy
import pytest

from k_search.kernel_generators.strategy_injection import (
    _text_similarity_ratio,
    learn_anti_pattern_from_failure,
)


# ---------------------------------------------------------------------------
# _text_similarity_ratio
# ---------------------------------------------------------------------------

class TestTextSimilarityRatio:
    """Tests for _text_similarity_ratio word-level overlap computation."""

    def test_identical_strings_returns_one(self):
        assert _text_similarity_ratio("DataCopy inside a loop", "DataCopy inside a loop") == 1.0

    def test_similar_strings_above_threshold(self):
        # Same words in slightly different order should be >0.8
        ratio = _text_similarity_ratio(
            "DataCopy inside a loop",
            "a DataCopy loop inside",
        )
        assert ratio > 0.8

    def test_different_strings_below_threshold(self):
        ratio = _text_similarity_ratio(
            "DataCopy inside a loop",
            "Pipeline stage compute fusion",
        )
        assert ratio < 0.5

    def test_both_empty_returns_one(self):
        assert _text_similarity_ratio("", "") == 1.0

    def test_one_empty_returns_zero(self):
        assert _text_similarity_ratio("some text", "") == 0.0
        assert _text_similarity_ratio("", "some text") == 0.0

    def test_case_insensitive(self):
        ratio = _text_similarity_ratio("DataCopy Loop", "datacopy loop")
        assert ratio == 1.0

    def test_partial_overlap(self):
        ratio = _text_similarity_ratio(
            "DataCopy inside a loop with alignment",
            "DataCopy inside a loop without alignment",
        )
        # Most words overlap, should be > 0.5
        assert ratio > 0.5


# ---------------------------------------------------------------------------
# learn_anti_pattern_from_failure
# ---------------------------------------------------------------------------

def _make_catalog():
    """Build a minimal catalog fixture with two strategies."""
    return [
        {
            "id": "S01",
            "name": "Vectorize inner loop",
            "category": "compute",
            "difficulty": 3,
            "impact": "medium",
            "natural_language": "Replace the scalar inner loop with vector operations.",
            "api_references": [
                {
                    "api_name": "DataCopy",
                    "doc_path": "ascendc/api/DataCopy.md",
                    "summary": "DataCopy(src, dst, len) - copy between buffers",
                    "anti_patterns": [
                        {
                            "id": "ap1",
                            "pattern": "DataCopy inside a loop",
                            "reason": "Causes redundant UB-to-UB transfers",
                            "source": "ascendc/api/DataCopy.md",
                            "discovered_at": 0,
                        },
                    ],
                },
            ],
        },
        {
            "id": "S02",
            "name": "Tiling optimization",
            "category": "tiling",
            "difficulty": 4,
            "impact": "high",
            "natural_language": "Apply tiling for better cache utilization.",
            "api_references": [
                {
                    "api_name": "DataCopyPad",
                    "doc_path": "ascendc/api/DataCopyPad.md",
                    "summary": "DataCopyPad(src, dst, len) - padded copy between buffers",
                    "anti_patterns": [
                        {
                            "id": "ap2",
                            "pattern": "DataCopyPad with zero length",
                            "reason": "Zero-length copy is undefined behavior",
                            "source": "ascendc/api/DataCopyPad.md",
                            "discovered_at": 0,
                        },
                    ],
                },
            ],
        },
    ]


class TestLearnAntiPatternFromFailure:
    """Tests for learn_anti_pattern_from_failure catalog mutation."""

    def test_basic_learning_adds_new_anti_pattern(self):
        catalog = _make_catalog()
        new_ap = {
            "pattern": "DataCopy with misaligned buffer",
            "reason": "Buffers must be 32-byte aligned for DataCopy",
        }
        result = learn_anti_pattern_from_failure(
            catalog, "S01", "DataCopy", new_ap, round_index=3,
        )
        # Should modify in-place and return the same catalog
        assert result is catalog
        # Find the DataCopy api_reference
        s01 = result[0]
        dc_ref = s01["api_references"][0]
        assert len(dc_ref["anti_patterns"]) == 2  # ap1 + new
        new_entry = dc_ref["anti_patterns"][1]
        assert new_entry["id"] == "ap3"  # ap1+ap2 exist, next is ap3
        assert new_entry["pattern"] == "DataCopy with misaligned buffer"
        assert new_entry["reason"] == "Buffers must be 32-byte aligned for DataCopy"
        assert new_entry["source"] == "round_3_failure"
        assert new_entry["discovered_at"] == 3

    def test_dedup_skips_similar_pattern(self):
        catalog = _make_catalog()
        # New pattern sharing all same words as existing "DataCopy inside a loop"
        # (4 common words out of 4 union words = similarity 1.0 > 0.8)
        new_ap = {
            "pattern": "DataCopy a inside loop",
            "reason": "Causes redundant UB-to-UB transfers in loops",
        }
        result = learn_anti_pattern_from_failure(
            catalog, "S01", "DataCopy", new_ap, round_index=5,
        )
        # Should not add because similarity > 0.8
        dc_ref = result[0]["api_references"][0]
        assert len(dc_ref["anti_patterns"]) == 1  # unchanged

    def test_auto_id_generation_after_existing_ids(self):
        catalog = _make_catalog()
        # Existing ids are ap1 (S01/DataCopy) and ap2 (S02/DataCopyPad)
        new_ap = {
            "pattern": "LocalToGlobal on wrong buffer",
            "reason": "Wrong buffer type causes runtime error",
        }
        result = learn_anti_pattern_from_failure(
            catalog, "S02", "DataCopyPad", new_ap, round_index=7,
        )
        dp_ref = result[1]["api_references"][0]
        new_entry = dp_ref["anti_patterns"][1]
        assert new_entry["id"] == "ap3"

    def test_auto_id_generation_when_no_existing_patterns(self):
        catalog = _make_catalog()
        # Remove all existing anti_patterns
        for s in catalog:
            for ref in s.get("api_references", []):
                ref["anti_patterns"] = []
        new_ap = {
            "pattern": "First discovered pattern",
            "reason": "Some reason",
        }
        result = learn_anti_pattern_from_failure(
            catalog, "S01", "DataCopy", new_ap, round_index=1,
        )
        dc_ref = result[0]["api_references"][0]
        assert len(dc_ref["anti_patterns"]) == 1
        assert dc_ref["anti_patterns"][0]["id"] == "ap1"

    def test_strategy_id_not_found_returns_unchanged(self):
        catalog = _make_catalog()
        catalog_copy = copy.deepcopy(catalog)
        new_ap = {
            "pattern": "Something new",
            "reason": "Some reason",
        }
        result = learn_anti_pattern_from_failure(
            catalog, "S99", "DataCopy", new_ap, round_index=2,
        )
        assert result is catalog
        # Catalog should be unchanged
        assert result == catalog_copy

    def test_api_name_not_found_returns_unchanged(self):
        catalog = _make_catalog()
        catalog_copy = copy.deepcopy(catalog)
        new_ap = {
            "pattern": "Something new",
            "reason": "Some reason",
        }
        result = learn_anti_pattern_from_failure(
            catalog, "S01", "NonExistentApi", new_ap, round_index=2,
        )
        assert result is catalog
        # Catalog should be unchanged
        assert result == catalog_copy

    def test_new_anti_pattern_ignores_api_name_hint(self):
        catalog = _make_catalog()
        new_ap = {
            "pattern": "DataCopy on unaligned address",
            "reason": "Alignment required",
            "api_name_hint": "ThisShouldBeIgnored",
        }
        result = learn_anti_pattern_from_failure(
            catalog, "S01", "DataCopy", new_ap, round_index=4,
        )
        dc_ref = result[0]["api_references"][0]
        new_entry = dc_ref["anti_patterns"][1]
        # "api_name_hint" should NOT appear in the stored entry
        assert "api_name_hint" not in new_entry
        assert new_entry["id"] == "ap3"
        assert new_entry["pattern"] == "DataCopy on unaligned address"