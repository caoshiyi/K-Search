#!/usr/bin/env python3
"""
从策略变体测试文件中提取单个策略，为每个变体生成单独的JSON文件。

用法:
    python3 scripts/extract_strategy_variants.py \
        --source strategies/mqa_strategy_variants_test.json \
        --output-dir strategies/mqa_experiments
"""

import argparse
import json
import sys
from pathlib import Path


def extract_strategy_variants(source_path: str, output_dir: str) -> int:
    """从源文件提取策略变体并保存为单独文件"""
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        print(f"ERROR: Source file not found: {source}")
        return 1

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)

    catalog = data.get("strategy_catalog", [])
    if not isinstance(catalog, list):
        print(f"ERROR: Expected 'strategy_catalog' to be a list, got {type(catalog).__name__}")
        return 1

    print(f"[EXTRACT] Found {len(catalog)} strategy variants in {source}")

    for strategy in catalog:
        variant_id = strategy.get("id", "unknown")
        variant_name = strategy.get("name", "unknown")

        # 创建单策略文件（保持完整的策略结构）
        single_strategy_data = {
            "strategy_catalog": [strategy]
        }

        output_file = output / f"{variant_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(single_strategy_data, f, ensure_ascii=False, indent=2)

        # 计算策略长度（不包括外层包装）
        strategy_chars = len(json.dumps(strategy, ensure_ascii=False))
        print(f"[CREATE] {variant_id}.json ({strategy_chars} chars) - {variant_name}")

    print(f"[DONE] Extracted {len(catalog)} strategy files to {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Extract strategy variants to separate files")
    parser.add_argument("--source", required=True, help="Source strategy catalog JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for individual strategy files")
    args = parser.parse_args()

    sys.exit(extract_strategy_variants(args.source, args.output_dir))


if __name__ == "__main__":
    main()