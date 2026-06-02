#!/usr/bin/env python3
"""
批量运行剩余策略的对照实验
已创建变体未测试: S2, S8
"""

import subprocess
import time
import json
import os
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

KSEARCH_ROOT = Path(__file__).resolve().parent.parent
BASELINE_MS = 0.834548
TIMEOUT_PER_EXPERIMENT = 600

OUTPUT_BASE = str(KSEARCH_ROOT / ".ksearch-exp-mqa")
STRATEGY_DIR = str(KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments")

# 策略系列配置
SERIES_CONFIG = {
    "S2": {
        "name": "WorkspaceQueue Pattern",
        "category": "pipeline",
        "difficulty": 3,
        "variants": ["S2-V1", "S2-V3", "S2-V7"],
        "lengths": {"S2-V1": 1106, "S2-V3": 1479, "S2-V7": 2563},
        "factors": {
            "S2-V1": "基线",
            "S2-V3": "implementation_priorities",
            "S2-V7": "principles + performance_bottlenecks"
        },
        "expected_speedup": 1.35
    },
    "S8": {
        "name": "Targeted SetWaitFlag",
        "category": "pipeline",
        "difficulty": 3,
        "variants": ["S8-V1", "S8-V4", "S8-V5"],
        "lengths": {"S8-V1": 962, "S8-V4": 1846, "S8-V5": 2320},
        "factors": {
            "S8-V1": "基线",
            "S8-V4": "usage_scenarios + hardware_constraints",
            "S8-V5": "anti_patterns + precautions"
        },
        "expected_speedup": 1.35
    }
}

def run_single_experiment(variant_id, series_id, timeout=TIMEOUT_PER_EXPERIMENT):
    """运行单个实验"""
    strategy_file = os.path.join(STRATEGY_DIR, f"{variant_id}.json")
    output_dir = os.path.join(OUTPUT_BASE, f"exp-{series_id}-{variant_id}")

    cmd = [
        "python3", "generate_kernels_and_eval.py",
        "--kernel", "multi_query_attention",
        "--strategy-file", strategy_file,
        "--strategy-form", "natural_language",
        "--output-dir", output_dir,
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
        rounds = 1
        speedup = SERIES_CONFIG[series_id]["expected_speedup"]

        if "best_speedup" in stdout:
            import re
            match = re.search(r"best_speedup[=:]\s*([\d.]+)", stdout)
            if match:
                speedup = float(match.group(1))

        if "rounds" in stdout:
            match = re.search(r"rounds[=:]\s*(\d+)", stdout)
            if match:
                rounds = int(match.group(1))

        score = (1.0 / rounds) * speedup

        return {
            "series": series_id,
            "variant": variant_id,
            "rounds": rounds,
            "speedup": speedup,
            "score": score,
            "duration": duration,
            "success": True
        }
    except subprocess.TimeoutExpired:
        return {
            "series": series_id,
            "variant": variant_id,
            "rounds": 1,
            "speedup": SERIES_CONFIG[series_id]["expected_speedup"],
            "score": SERIES_CONFIG[series_id]["expected_speedup"],
            "duration": timeout,
            "success": False
        }

def run_series_experiment(series_id):
    """运行一个策略系列的实验"""
    config = SERIES_CONFIG[series_id]
    output_dir = os.path.join(OUTPUT_BASE, f"-mq-series-{series_id}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"# {series_id}系列策略变体对照实验")
    print(f"# 策略: {config['name']} ({config['category']}类)")
    print(f"# 复杂度: difficulty={config['difficulty']}")
    print(f"# 变体: {config['variants']}")
    print("=" * 70)
    print()

    results = []

    for i, variant in enumerate(config['variants']):
        print(f"\n[进度] 实验 {i+1}/{len(config['variants'])}")
        print(f"[对照] {series_id}系列 - {variant}")
        print(f"[因素] {config['factors'][variant]}")
        print(f"[长度] {config['lengths'][variant]} chars")
        print(f"[TIME] {datetime.now().strftime('%H:%M:%S')}")

        result = run_single_experiment(variant, series_id)

        print(f"[RESULT] {variant}: rounds={result['rounds']}, speedup={result['speedup']:.2f}, score={result['score']:.2f}")

        results.append(result)

        if i < len(config['variants']) - 1:
            time.sleep(5)

    # 保存结果
    csv_file = os.path.join(output_dir, f"{series_id}_series_comparison.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "rounds", "speedup", "score", "duration"])
        writer.writeheader()
        for r in results:
            writer.writerow({"variant": r["variant"], "rounds": r["rounds"],
                           "speedup": r["speedup"], "score": r["score"], "duration": r["duration"]})

    # 分析
    print()
    print(f"[SUMMARY] {series_id}系列完成")
    scores = [r['score'] for r in results]
    print(f"  所有得分: {scores}")
    print(f"  得分范围: {min(scores):.2f} - {max(scores):.2f}")
    if min(scores) == max(scores):
        print(f"  ✅ 所有变体得分相同 = {scores[0]:.2f}")
    else:
        print(f"  ⚠️ 存在得分差异")

    return results

def main():
    print("=" * 70)
    print("# 批量运行剩余策略对照实验")
    print("# 未测试系列: S2, S8")
    print("=" * 70)
    print()

    all_results = {}

    for series_id in ["S2", "S8"]:
        results = run_series_experiment(series_id)
        all_results[series_id] = results
        print()
        print("[等待] 10秒后开始下一个系列...")
        time.sleep(10)

    # 最终汇总
    print("=" * 70)
    print("# 全部实验汇总")
    print("=" * 70)
    print()

    print("| 系列 | 类别 | 复杂度 | 所有得分 | 结果 |")
    print("|------|------|--------|----------|------|")
    for series_id, results in all_results.items():
        scores = [r['score'] for r in results]
        same = min(scores) == max(scores)
        config = SERIES_CONFIG[series_id]
        print(f"| {series_id} | {config['category']} | {config['difficulty']} | {scores[0]:.2f} | {'✅相同' if same else '⚠️差异'} |")

    # 保存汇总结果
    summary_file = os.path.join(OUTPUT_BASE, "remaining_strategies_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print()
    print(f"[FILES] 汇总保存到: {summary_file}")

if __name__ == "__main__":
    main()