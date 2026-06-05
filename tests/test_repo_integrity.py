from pathlib import Path


def test_default_mqa_strategy_catalog_exists_and_loads():
    from k_search.kernel_generators.strategy_injection import load_strategy_catalog

    root = Path(__file__).resolve().parents[1]
    catalog = root / "strategies" / "mqa_strategies_catalog.json"

    assert catalog.is_file()
    assert len(load_strategy_catalog(catalog)) == 12
