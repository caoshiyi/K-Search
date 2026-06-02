#!/usr/bin/env python3
"""
MQA策略变体批量实验执行脚本

自动化运行所有策略变体，收集结果，计算评估指标。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import csv

# 配置
BASELINE_MS = 0.834548
KSEARCH_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = Path(os.environ.get("TASK_DIR", "/mnt/workspace/cv_agent/tile2asc/multi_query_attention"))
STRATEGIES_DIR = KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments"
RESULTS_DIR = KSEARCH_ROOT / ".ksearch-exp-mqa-batch"

# 策略变体列表（按策略ID排序）
VARIANTS = [
    "S1-V1", "S1-V2", "S1-V4", "S1-V5",
    "S2-V1", "S2-V3", "S2-V7",
    "S4-V1", "S4-V5", "S4-V6",
    "S8-V1", "S8-V4", "S8-V5",
    "S10-V1", "S10-V4", "S10-V7",
]

def run_experiment(variant_id: str, results_csv) -> dict:
    """运行单个策略变体实验"""
    strategy_file = STRATEGIES_DIR / f"{variant_id}.json"
    if not strategy_file.exists():
        print(f"[ERROR] Strategy file not found: {strategy_file}")
        return {
            "variant_id": variant_id,
            "status": "strategy_file_missing",
            "rounds": "NA",
            "speedup": "NA",
            "score": "NA",
        }

    print(f"\n[EXPERIMENT] Testing variant: {variant_id}")
    print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 计算策略长度
    with open(strategy_file) as f:
        strategy_data = json.load(f)
    strategy_chars = len(json.dumps(strategy_data["strategy_catalog"][0], ensure_ascii=False))

    # 输出目录
    output_dir = RESULTS_DIR / f"{variant_id}_natural_language"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行K-Search
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
        "--ascendc-timeout-seconds", "600",  # 缩短超时到10分钟
        "--world-model",
        "--strategy-file", str(strategy_file),
        "--strategy-form", "natural_language",
        "--max-opt-rounds", "5",  # 减少轮数到5轮以加快实验
        "--artifacts-dir", str(output_dir),
        "--save-solutions",
    ]

    # 写入结果CSV（实时更新）
    log_file = output_dir / "experiment_log.txt"

    try:
        print(f"[CMD] Running: {' '.join(cmd[:10])}...")
        with open(log_file, "w") as log:
            env = {
                "BASELINE_MS": str(BASELINE_MS),
                "KSEARCH_RUN_START": run_start,
                "PATH": subprocess.os.environ["PATH"],
                "HOME": subprocess.os.environ["HOME"],
                "ANTHROPIC_AUTH_TOKEN": subprocess.os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
                "ANTHROPIC_BASE_URL": subprocess.os.environ.get("ANTHROPIC_BASE_URL", ""),
            }

            process = subprocess.Popen(
                cmd,
                cwd=str(KSEARCH_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # 等待完成（最长10分钟）
            start_time = time.time()
            timeout_seconds = 600

            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f"[TIMEOUT] Killing process after {timeout_seconds}s")
                    process.kill()
                    process.wait()
                    break
                time.sleep(10)
                print(f"[PROGRESS] {variant_id}: {int(elapsed)}s elapsed...")

            elapsed_total = time.time() - start_time
            print(f"[TIME] Completed in {int(elapsed_total)}s")

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return {
            "variant_id": variant_id,
            "strategy_chars": strategy_chars,
            "status": "execution_error",
            "rounds": "NA",
            "speedup": "NA",
            "score": "NA",
        }

    # 解析结果
    rounds = "NA"
    speedup = "NA"
    score = "NA"
    status = "completed"

    try:
        # 从world_model.json读取最终状态
        wm_file = output_dir / "multi_query_attention" / "world_model" / "world_model.json"
        if wm_file.exists():
            with open(wm_file) as f:
                wm = json.load(f)
                rounds = wm.get("computed_signals", {}).get("round_index", "NA")
                best_score = wm.get("best_score", None)
                if best_score and best_score != 1.0:
                    speedup = best_score
                    if rounds != "NA" and speedup != "NA":
                        score = (1.0 / int(rounds)) * float(speedup)

        # 从日志中查找
        with open(log_file) as f:
            log_content = f.read()
            # 查找最后一轮的加速比
            import re
            speedup_matches = re.findall(r"speedup[:\s]+([0-9.]+)", log_content)
            if speedup_matches:
                speedup = speedup_matches[-1]

            # 查找轮数
            round_matches = re.findall(r"round[:\s]+([0-9]+)", log_content)
            if round_matches:
                rounds = round_matches[-1]

            # 检查是否有错误
            if "ERROR" in log_content or "Traceback" in log_content:
                status = "partial_failure"

    except Exception as e:
        print(f"[ERROR] Failed to parse results: {e}")
        status = "parse_error"

    result = {
        "variant_id": variant_id,
        "strategy_chars": strategy_chars,
        "rounds": rounds,
        "speedup": speedup,
        "score": score,
        "status": status,
        "elapsed_seconds": int(elapsed_total),
    }

    # 写入CSV
    with open(results_csv, "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            result["variant_id"],
            result["strategy_chars"],
            result["rounds"],
            result["speedup"],
            result["score"],
            result["status"],
        ])

    print(f"[RESULT] {variant_id}: rounds={rounds}, speedup={speedup}, score={score}, status={status}")
    return result


def main():
    """运行所有实验"""
    print(f"[START] MQA Strategy Batch Experiment")
    print(f"[BASELINE] BASELINE_MS={BASELINE_MS}")
    print(f"[TOTAL] {len(VARIANTS)} variants to test")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 创建结果CSV
    results_csv = RESULTS_DIR / "experiment_results.csv"
    with open(results_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["variant_id", "strategy_chars", "rounds", "speedup", "score", "status"])

    # 运行所有实验
    all_results = []
    for variant_id in VARIANTS:
        result = run_experiment(variant_id, results_csv)
        all_results.append(result)

        # 休息一会儿
        time.sleep(5)

    # 总结
    print(f"\n[SUMMARY] All experiments completed!")
    print(f"[RESULTS] CSV saved to: {results_csv}")

    # 分析Top 5
    valid_results = [r for r in all_results if r["score"] != "NA" and r["status"] != "parse_error"]
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: float(x["score"]), reverse=True)
        print(f"\n[ANALYSIS] Top 5 strategies by score:")
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r['variant_id']}: score={r['score']}, rounds={r['rounds']}, speedup={r['speedup']}")

    # 保存完整结果JSON
    results_json = RESULTS_DIR / "experiment_results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[JSON] Full results saved to: {results_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())