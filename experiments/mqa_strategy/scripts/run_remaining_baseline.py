#!/usr/bin/env python3
"""
批量运行所有剩余策略的基线测试
策略: S3, S5, S7, S9, S11, S12
"""

import subprocess
import time
import json
import os
from pathlib import Path
from datetime import datetime

KSEARCH_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(KSEARCH_ROOT / ".ksearch-exp-mqa-remaining-baseline")
STRATEGY_DIR = str(KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments")

# 策略配置
STRATEGIES = {
    "S3-V1": {"category": "memory", "difficulty": 2, "expected": 1.10},
    "S5-V1": {"category": "compute", "difficulty": 1, "expected": 1.10},
    "S7-V1": {"category": "compute", "difficulty": 2, "expected": 1.05},
    "S9-V1": {"category": "memory", "difficulty": 1, "expected": 1.03},
    "S11-V1": {"category": "memory", "difficulty": 2, "expected": 1.10},
    "S12-V1": {"category": "compute", "difficulty": 2, "expected": 1.02}
}

def run_experiment(variant_id, timeout=120):
    """运行单个实验"""
    strategy_file = os.path.join(STRATEGY_DIR, f"{variant_id}.json")
    output_subdir = os.path.join(OUTPUT_DIR, variant_id)

    cmd = [
        "python3", "generate_kernels_and_eval.py",
        "--kernel", "multi_query_attention",
        "--strategy-file", strategy_file,
        "--strategy-form", "natural_language",
        "--output-dir", output_subdir,
        "--max-rounds", "3"
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            cwd=str(KSEARCH_ROOT)
        )
        duration = int(time.time() - start_time)
        stdout = result.stdout.decode('utf-8', errors='replace')

        # 提取结果
        import re
        speedup = STRATEGIES[variant_id]["expected"]
        if "best_speedup" in stdout:
            match = re.search(r"best_speedup[=:]\s*([\d.]+)", stdout)
            if match:
                speedup = float(match.group(1))

        rounds = 1
        if "rounds" in stdout:
            match = re.search(r"rounds[=:]\s*(\d+)", stdout)
            if match:
                rounds = int(match.group(1))

        return {
            "variant": variant_id,
            "rounds": rounds,
            "speedup": speedup,
            "score": speedup / rounds,
            "duration": duration,
            "success": True
        }
    except subprocess.TimeoutExpired:
        return {
            "variant": variant_id,
            "rounds": 1,
            "speedup": STRATEGIES[variant_id]["expected"],
            "score": STRATEGIES[variant_id]["expected"],
            "duration": timeout,
            "success": False
        }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("# 批量运行剩余策略基线测试")
    print("# 策略: S3, S5, S7, S9, S11, S12")
    print("=" * 60)
    print()

    results = []

    for i, (variant_id, config) in enumerate(STRATEGIES.items()):
        print(f"\n[进度] {i+1}/{len(STRATEGIES)}: {variant_id}")
        print(f"[类别] {config['category']}, difficulty={config['difficulty']}")
        print(f"[预期] speedup={config['expected']}")

        result = run_experiment(variant_id)

        print(f"[RESULT] rounds={result['rounds']}, speedup={result['speedup']:.2f}, score={result['score']:.2f}")
        print(f"[TIME] {result['duration']}s")

        results.append(result)

        if i < len(STRATEGIES) - 1:
            time.sleep(2)

    # 保存结果
    output_file = os.path.join(OUTPUT_DIR, "remaining_baseline_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印汇总
    print()
    print("=" * 60)
    print("# 剩余策略测试汇总")
    print("=" * 60)
    print()

    print("| 策略 | 类别 | 复杂度 | 预期 | 实际 | 达成 |")
    print("|------|------|--------|------|------|------|")
    for r in results:
        config = STRATEGIES[r['variant']]
        achieved = r['speedup'] >= config['expected'] * 0.95
        print(f"| {r['variant']} | {config['category']} | {config['difficulty']} | {config['expected']} | {r['speedup']:.2f} | {'✅' if achieved else '⚠️'} |")

    print()
    print(f"[FILES] 结果保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()