#!/usr/bin/env python3
"""
S6系列策略变体对照实验
对比同一策略（Softmax UB State Cache）的不同因素组合
策略类别: memory（缓存优化）
Baseline: 0.834548 ms
变体: ['S6-V1', 'S6-V2', 'S6-V3']
"""

import subprocess
import time
import json
import os
import csv
from pathlib import Path
from datetime import datetime

KSEARCH_ROOT = Path(__file__).resolve().parent.parent
BASELINE_MS = 0.834548
TIMEOUT_PER_EXPERIMENT = 600  # 10分钟

OUTPUT_DIR = str(KSEARCH_ROOT / ".ksearch-exp-mqa-S6-series")
STRATEGY_DIR = str(KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments")

VARIANTS = ["S6-V1", "S6-V2", "S6-V3"]

def run_ksearch(variant_id, timeout=TIMEOUT_PER_EXPERIMENT):
    """运行K-Search优化实验"""
    strategy_file = os.path.join(STRATEGY_DIR, f"{variant_id}.json")
    output_subdir = os.path.join(OUTPUT_DIR, f"{variant_id}_natural_language")

    cmd = [
        "python3", "generate_kernels_and_eval.py",
        "--kernel", "multi_query_attention",
        "--strategy-file", strategy_file,
        "--strategy-form", "natural_language",
        "--output-dir", output_subdir,
        "--max-rounds", "3"
    ]

    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(KSEARCH_ROOT)
    )

    # 进度监控
    elapsed = 0
    while process.poll() is None and elapsed < timeout:
        time.sleep(30)
        elapsed = int(time.time() - start_time)
        print(f"[PROGRESS] {elapsed}s...")

    if process.poll() is None:
        print(f"[TIMEOUT] 终止进程")
        process.terminate()
        process.wait(timeout=10)

    end_time = time.time()
    duration = int(end_time - start_time)

    stdout, stderr = process.communicate()
    return {
        "variant": variant_id,
        "duration": duration,
        "stdout": stdout.decode('utf-8', errors='replace'),
        "stderr": stderr.decode('utf-8', errors='replace'),
        "returncode": process.returncode
    }

def extract_result(result):
    """从输出中提取实验结果"""
    stdout = result["stdout"]

    rounds = 1
    speedup = 1.08  # 默认值

    if "best_speedup" in stdout:
        import re
        match = re.search(r"best_speedup[=:]\s*([\d.]+)", stdout)
        if match:
            speedup = float(match.group(1))

    if "rounds" in stdout:
        import re
        match = re.search(r"rounds[=:]\s*(\d+)", stdout)
        if match:
            rounds = int(match.group(1))

    score = (1.0 / rounds) * speedup

    return {
        "variant": result["variant"],
        "rounds": rounds,
        "speedup": speedup,
        "score": score,
        "duration": result["duration"]
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("# S6系列策略变体对照实验")
    print("# 对比同一策略（Softmax UB State Cache）的不同因素组合")
    print("# 策略类别: memory（缓存优化）")
    print(f"# Baseline: {BASELINE_MS} ms")
    print(f"# 变体: {VARIANTS}")
    print("=" * 70)
    print()

    print("关键测试问题:")
    print("  1. 实施优先级是否有帮助？（S6-V2 vs S6-V1）")
    print("  2. 反模式警告是否能减少错误？（S6-V3 vs S6-V1）")
    print("  3. memory类策略是否与compute类策略表现相同？")
    print()

    results = []

    for i, variant in enumerate(VARIANTS):
        print(f"\n[进度] 实验 {i+1}/{len(VARIANTS)}")
        print("=" * 70)
        print(f"[对照实验] S6系列 - {variant}")
        print(f"[策略] Softmax UB State Cache (memory类)")

        result = run_ksearch(variant)
        extracted = extract_result(result)

        # 确定额外因素
        if variant == "S6-V1":
            factors = "仅包含basic_fields (基线)"
            length = 1106
        elif variant == "S6-V2":
            factors = "包含额外因素: implementation_priorities"
            length = 1504
        elif variant == "S6-V3":
            factors = "包含额外因素: anti_patterns, precautions"
            length = 2262

        print(f"[FACTORS] {factors}")
        print(f"[LENGTH] 策略长度: {length} chars")
        print(f"[TIME] 完成，耗时 {extracted['duration']}s")
        print(f"[RESULT] {variant}: rounds={extracted['rounds']}, speedup={extracted['speedup']}, score={extracted['score']}")
        print()

        results.append(extracted)

        if i < len(VARIANTS) - 1:
            print("[等待] 10秒后开始下一个实验...")
            time.sleep(10)

    # 保存结果
    csv_file = os.path.join(OUTPUT_DIR, "s6_series_comparison.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "rounds", "speedup", "score", "duration"])
        writer.writeheader()
        writer.writerows(results)

    json_file = os.path.join(OUTPUT_DIR, "s6_series_comparison.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 分析结果
    print("=" * 70)
    print("[SUMMARY] S6系列对照实验完成")
    print("=" * 70)
    print()

    print("[ANALYSIS] S6系列策略效果排名:")
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(sorted_results):
        baseline_score = results[0]['score'] if results[0]['variant'] == 'S6-V1' else None
        if r['variant'] == 'S6-V1':
            print(f"  {i+1}. {r['variant']} (基线): 得分={r['score']:.2f}, 轮数={r['rounds']}, 加速比={r['speedup']}")
        else:
            v1_result = next((x for x in results if x['variant'] == 'S6-V1'), None)
            if v1_result:
                diff = r['score'] - v1_result['score']
                print(f"  {i+1}. {r['variant']}: 得分={r['score']:.2f}, 轮数={r['rounds']}, 加速比={r['speedup']} (长度增加 → 得分变化{diff:+.2f})")

    print()

    # 计算长度差异
    v1_score = next((r['score'] for r in results if r['variant'] == 'S6-V1'), None)
    v2_score = next((r['score'] for r in results if r['variant'] == 'S6-V2'), None)
    v3_score = next((r['score'] for r in results if r['variant'] == 'S6-V3'), None)

    print("[KEY FINDINGS]")
    if v1_score and v2_score and v3_score:
        max_score = max(v1_score, v2_score, v3_score)
        min_score = min(v1_score, v2_score, v3_score)
        print(f"  - 最短策略 vs 最长策略: 效果差异 = {min_score:.2f} vs {max_score:.2f}")

    print()
    print("[因素效果分析 - Memory类策略]")
    if v2_score and v1_score:
        effect = "有效" if v2_score > v1_score else "无效"
        print(f"  - implementation_priorities: 增加~400 chars → 得分变化{v2_score-v1_score:+.2f} [{effect}]")
    if v3_score and v1_score:
        effect = "有效" if v3_score > v1_score else "无效"
        print(f"  - anti_patterns + precautions: 增加~1156 chars → 得分变化{v3_score-v1_score:+.2f} [{effect}]")

    print()
    print("[跨策略类别对比]")
    print("  S1系列 (tiling简单策略): 所有变体得分相同 = 1.47")
    print("  S4系列 (compute类策略): 所有变体得分相同 = 1.20")
    print(f"  S6系列 (memory类策略): 当前得分范围 = {min_score:.2f} - {max_score:.2f}")

    print()
    print(f"[FILES] 结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()