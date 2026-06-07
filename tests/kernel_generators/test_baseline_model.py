import pytest

from k_search.kernel_generators.baseline_model import (
    ms_to_us,
    normalize_expected_speedup,
    validate_expected_speedup,
)
from k_search.kernel_generators.strategy_injection import load_strategy_catalog


def test_expected_speedup_parent_strategy_schema():
    entries = {
        entry.id: entry
        for entry in load_strategy_catalog("strategies/flash_attention_round_design_strategies/catalog.json")
    }

    speedup = entries["fa_multibuffer_soft_pipeline"].expected_speedup

    assert speedup is not None
    assert speedup["relative_to"] == "parent_strategy"
    assert speedup["parent_strategy_id"] == "fa_qkv_two_level_l1_reuse"
    assert speedup["factor"] == pytest.approx(1.63)
    assert speedup["parent_latency_us"] == pytest.approx(1590.0)
    assert speedup["target_latency_us"] == pytest.approx(977.0)


def test_validate_expected_speedup_rejects_unspecified_for_new_schema():
    with pytest.raises(ValueError, match="relative_to"):
        validate_expected_speedup({"factor": 1.2, "relative_to": "unspecified_legacy"})


def test_legacy_expected_vs_baseline_is_normalized_not_dropped():
    speedup = normalize_expected_speedup(
        {"expected_vs_baseline_factor": 1.63},
        strategy_requires=("fa_qkv_two_level_l1_reuse",),
    )

    assert speedup["factor"] == 1.63
    assert speedup["relative_to"] == "parent_strategy"
    assert speedup["parent_strategy_id"] == "fa_qkv_two_level_l1_reuse"
    assert speedup["source"] == "legacy_expected_vs_baseline_factor"


def test_ms_to_us_uses_explicit_unit_conversion():
    assert ms_to_us(2.391) == pytest.approx(2391.0)
    assert ms_to_us(None) is None
