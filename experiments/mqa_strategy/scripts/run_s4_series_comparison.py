#!/usr/bin/env python3
"""
S4系列策略变体对照实验 - 测试向量化行操作策略的不同因素组合

对比同一策略（Vectorized RowMuls/RowDivs）的3个变体：
- S4-V1: basic_fields (基线)
- S4-V5: basic + anti_patterns + precautions (反模式版)
- S4-V6: basic + code_examples + api_references (示例代码版，已测试得分1.20)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import csv
import re

BASELINE_MS = 0.834548
KSEARCH_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = Path(os.environ.get("TASK_DIR", "/mnt/workspace/cv_agent/tile2asc/multi_query_attention"))
STRATEGIES_DIR = KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments"
RESULTS_DIR = KSEARCH_ROOT / ".ksearch-exp-mqa-S4-series"

S4_VARIANTS = ["S4-V1", "S4-V5", "S4-V6"]

def run_experiment(variant_id: str, results_csv) -> dict:
    """运行单个策略变体实验"""
    strategy_file = STRATEGIES_DIR / f"{variant_id}.json"

    print(f"\n{'='*70}")
    print(f"[对照实验] S4系列 - {variant_id}")
    print(f"[策略] Vectorized RowMuls/RowDivs (compute类)")
    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    with open(strategy_file) as f:
        strategy_data = json.load(f)
    strategy = strategy_data["strategy_catalog"][0]
    strategy_chars = len(json.dumps(strategy, ensure_ascii=False))

    # 显示因素组合
    factors = []
    if "anti_patterns" in strategy:
        factors.append("anti_patterns")
    if "precautions" in strategy:
        factors.append("precautions")
    if "code_examples" in strategy:
        factors.append("code_examples")
    if "api_references" in strategy:
        factors.append("api_references")

    if factors:
        print(f"[FACTORS] 包含额外因素: {', '.join(factors)}")
    else:
        print(f"[FACTORS] 仅包含basic_fields (基线)")

    print(f"[LENGTH] 策略长度: {strategy_chars} chars")

    # 显示关键API信息（如果存在）
    if "api_references" in strategy:
        api_refs = strategy.get("api_references", [])
        if api_refs:
            print(f"[APIs] 涉及API数量: {len(api_refs)}")
            for ref in api_refs[:2]:
                print(f"       - {ref.get('api_name', 'unknown')}")

    # 显示反模式数量（如果存在）
    anti_count = 0
    if "api_references" in strategy:
        for ref in strategy.get("api_references", []):
            patterns = ref.get("anti_patterns", [])
            anti_count += len(patterns)
    if anti_count > 0:
        print(f"[WARNINGS] 反模式警告数量: {anti_count}")

    output_dir = RESULTS_DIR / f"{variant_id}_natural_language"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_start = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cmd = [
        "python", "-u", "generate_kernels_and_eval.py",
        "--task-source", "ascendc",
        "--task-path", str(TASK_DIR),
        "--definition", "multi_query_attention",
        "--model-name", "glm-5.1",
        "--llm-provider", "claude-agent",
        "--language", "ascendc",
        "--target-gpu", "Ascend910B3",
        "--ascendc-build-cmd", "./ksearch_build.sh",
        "--ascendc-test-cmd", "./ksearch_test.sh",
        "--ascendc-bench-cmd", "./ksearch_bench.sh",
        "--ascendc-reference-latency-ms", str(BASELINE_MS),
        "--ascendc-timeout-seconds", "600",
        "--world-model",
        "--strategy-file", str(strategy_file),
        "--strategy-form", "natural_language",
        "--max-opt-rounds", "3",
        "--artifacts-dir", str(output_dir),
        "--save-solutions",
    ]

    log_file = output_dir / "experiment_log.txt"
    elapsed_total = 0

    with open(log_file, "w") as log:
        env = dict(subprocess.os.environ)
        env["BASELINE_MS"] = str(BASELINE_MS)
        env["KSEARCH_RUN_START"] = run_start

        process = subprocess.Popen(
            cmd,
            cwd=str(KSEARCH_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        start_time = time.time()
        timeout_seconds = 600

        while process.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"\n[TIMEOUT] 终止进程")
                process.kill()
                process.wait()
                break
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"[PROGRESS] {int(elapsed)}s...")
            time.sleep(5)

        elapsed_total = time.time() - start_time
        print(f"[TIME] 完成，耗时 {int(elapsed_total)}s")

    rounds = "NA"
    speedup = "NA"
    score = "NA"
    status = "completed"

    with open(log_file) as f:
        log_content = f.read()

        round_matches = re.findall(r"round[:\s]+([0-9]+)", log_content, re.IGNORECASE)
        if round_matches:
            rounds = round_matches[-1]

        speedup_matches = re.findall(r"speedup[:\s]+([0-9.]+)", log_content, re.IGNORECASE)
        if speedup_matches:
            speedup = speedup_matches[-1]

        latency_matches = re.findall(r"latency_ms[:\s]+([0-9.]+)", log_content, re.IGNORECASE)
        if latency_matches:
            latency = float(latency_matches[-1])
            speedup = BASELINE_MS / latency

        if "ERROR" in log_content.upper():
            status = "partial_failure"

    if rounds != "NA" and speedup != "NA":
        score = (1.0 / int(rounds)) * float(speedup)

    result = {
        "variant_id": variant_id,
        "strategy_chars": strategy_chars,
        "rounds": rounds,
        "speedup": speedup,
        "score": score,
        "status": status,
        "elapsed_seconds": int(elapsed_total),
    }

    with open(results_csv, "a") as f:
        writer = csv.writer(f)
        writer.writerow([result["variant_id"], result["strategy_chars"],
                        result["rounds"], result["speedup"], result["score"],
                        result["status"], result["elapsed_seconds"]])

    print(f"\n[RESULT] {variant_id}: rounds={rounds}, speedup={speedup}, score={score}")
    return result


def main():
    print(f"\n{'#'*70}")
    print(f"# S4系列策略变体对照实验")
    print(f"# 对比同一策略（Vectorized RowMuls/RowDivs）的不同因素组合")
    print(f"# 策略类别: compute（向量化计算）")
    print(f"# Baseline: {BASELINE_MS} ms")
    print(f"# 变体: {S4_VARIANTS}")
    print(f"{'#'*70}\n")

    print(f"关键测试问题:")
    print(f"  1. 示例代码是否比简洁描述更有效？（S4-V6 vs S4-V1）")
    print(f"  2. 反模式警告是否能减少实现错误？（S4-V5 vs S4-V1）")
    print(f"  3. API参考文档是否有帮助？（S4-V6 vs S4-V1）")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_csv = RESULTS_DIR / "s4_series_comparison.csv"
    with open(results_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["variant_id", "strategy_chars", "rounds", "speedup",
                        "score", "status", "elapsed_seconds"])

    all_results = []
    for i, variant_id in enumerate(S4_VARIANTS, 1):
        print(f"\n[进度] 实验 {i}/{len(S4_VARIANTS)}")
        result = run_experiment(variant_id, results_csv)
        all_results.append(result)

        if i < len(S4_VARIANTS):
            print(f"\n[等待] 10秒后开始下一个实验...")
            time.sleep(10)

    print(f"\n{'='*70}")
    print(f"[SUMMARY] S4系列对照实验完成")
    print(f"{'='*70}\n")

    # 按得分排序
    valid_results = [r for r in all_results if r["score"] != "NA"]
    sorted_results = sorted(valid_results, key=lambda x: float(x["score"]), reverse=True)

    print(f"[ANALYSIS] S4系列策略效果排名:\n")
    for i, r in enumerate(sorted_results, 1):
        factors_desc = ""
        if "V5" in r["variant_id"]:
            factors_desc = " (反模式版)"
        elif "V6" in r["variant_id"]:
            factors_desc = " (示例代码版)"
        else:
            factors_desc = " (基线)"

        improvement = ""
        baseline = next((x for x in sorted_results if x["variant_id"] == "S4-V1"), None)
        if baseline and r["variant_id"] != "S4-V1":
            diff = float(r["score"]) - float(baseline["score"])
            length_diff = r["strategy_chars"] - baseline["strategy_chars"]
            improvement = f" (长度+{length_diff} chars → 得分{diff:+.2f})"

        print(f"  {i}. {r['variant_id']}{factors_desc}: "
              f"得分={r['score']}, 轮数={r['rounds']}, 加速比={r['speedup']}{improvement}")

    print(f"\n[KEY FINDINGS]")
    print(f"  - 最短策略 vs 最长策略: 效果差异 = {sorted_results[0]['score']} vs {sorted_results[-1]['score']}")

    # 因素效果分析
    print(f"\n[因素效果分析 - Compute类策略]")
    baseline = next((r for r in sorted_results if r["variant_id"] == "S4-V1"), None)
    if baseline:
        for r in sorted_results:
            if r["variant_id"] != "S4-V1":
                diff = float(r["score"]) - float(baseline["score"])
                length_diff = r["strategy_chars"] - baseline["strategy_chars"]
                efficiency = "无效" if diff == 0 else ("有效" if diff > 0 else "负面效果")

                factor_type = "反模式警告" if "V5" in r["variant_id"] else "示例代码+API参考"
                print(f"  - {factor_type}: 增加{length_diff} chars → 得分变化{diff:+.2f} [{efficiency}]")

    # 与S1系列对比
    print(f"\n[跨策略类别对比]")
    print(f"  S1系列 (tiling简单策略): 所有变体得分相同 = 1.47")
    print(f"  S4系列 (compute复杂策略): 当前得分范围 = {min(float(r['score']) for r in sorted_results):.2f} - {max(float(r['score']) for r in sorted_results):.2f}")

    results_json = RESULTS_DIR / "s4_series_comparison.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[FILES] 结果已保存到: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())