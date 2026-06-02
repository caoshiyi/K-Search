#!/usr/bin/env python3
"""
MQA策略变体快速示范实验 - 测试3个代表性变体验证流程
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

# 配置
BASELINE_MS = 0.834548
KSEARCH_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = Path(os.environ.get("TASK_DIR", "/mnt/workspace/cv_agent/tile2asc/multi_query_attention"))
STRATEGIES_DIR = KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments"
RESULTS_DIR = KSEARCH_ROOT / ".ksearch-exp-mqa-quick"

# 示范变体（3个代表性策略）
SAMPLE_VARIANTS = [
    "S1-V1",  # 最小版 - tiling策略
    "S4-V6",  # 示例代码版 - compute策略
    "S8-V4",  # 使用场景版 - pipeline策略
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

    print(f"\n{'='*60}")
    print(f"[EXPERIMENT] Testing variant: {variant_id}")
    print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 计算策略长度
    with open(strategy_file) as f:
        strategy_data = json.load(f)
    strategy_chars = len(json.dumps(strategy_data["strategy_catalog"][0], ensure_ascii=False))
    print(f"[STRATEGY] Length: {strategy_chars} chars")

    # 输出目录
    output_dir = RESULTS_DIR / f"{variant_id}_natural_language"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行K-Search（减少参数以加快速度）
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
        "--ascendc-timeout-seconds", "300",  # 5分钟超时
        "--world-model",
        "--strategy-file", str(strategy_file),
        "--strategy-form", "natural_language",
        "--max-opt-rounds", "3",  # 只运行3轮加快速度
        "--artifacts-dir", str(output_dir),
        "--save-solutions",
    ]

    log_file = output_dir / "experiment_log.txt"
    elapsed_total = 0

    try:
        print(f"[CMD] Starting K-Search...")
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

            # 等待完成（最长5分钟）
            start_time = time.time()
            timeout_seconds = 300

            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f"\n[TIMEOUT] Killing process after {int(elapsed)}s")
                    process.kill()
                    process.wait()
                    break

                # 每30秒打印进度
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    print(f"[PROGRESS] {variant_id}: {int(elapsed)}s elapsed...")

                time.sleep(5)

            elapsed_total = time.time() - start_time
            print(f"\n[TIME] Process completed in {int(elapsed_total)}s")

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        return {
            "variant_id": variant_id,
            "strategy_chars": strategy_chars,
            "status": "execution_error",
            "rounds": "NA",
            "speedup": "NA",
            "score": "NA",
            "elapsed_seconds": int(elapsed_total),
        }

    # 解析结果
    rounds = "NA"
    speedup = "NA"
    score = "NA"
    status = "completed"

    print(f"[PARSE] Analyzing results...")
    try:
        # 从日志解析
        with open(log_file) as f:
            log_content = f.read()

            # 查找轮数
            round_matches = re.findall(r"round[:\s]+([0-9]+)", log_content, re.IGNORECASE)
            if round_matches:
                rounds = round_matches[-1]
                print(f"[FOUND] rounds = {rounds}")

            # 查找加速比
            speedup_matches = re.findall(r"speedup[:\s]+([0-9.]+)", log_content, re.IGNORECASE)
            if speedup_matches:
                speedup = speedup_matches[-1]
                print(f"[FOUND] speedup = {speedup}")

            # 查找latency
            latency_matches = re.findall(r"latency_ms[:\s]+([0-9.]+)", log_content, re.IGNORECASE)
            if latency_matches:
                latency = float(latency_matches[-1])
                speedup = BASELINE_MS / latency
                print(f"[FOUND] latency_ms = {latency}, calculated speedup = {speedup}")

            # 检查错误
            if "ERROR" in log_content.upper() or "Traceback" in log_content:
                status = "partial_failure"
                print(f"[WARN] Found errors in log")

        # 从world_model.json读取（如果存在）
        wm_file = output_dir / "multi_query_attention" / "world_model" / "world_model.json"
        if wm_file.exists():
            with open(wm_file) as f:
                wm = json.load(f)
                wm_rounds = wm.get("computed_signals", {}).get("round_index", None)
                wm_best_score = wm.get("best_score", None)

                if wm_rounds is not None and wm_rounds != 0:
                    rounds = wm_rounds

                if wm_best_score is not None and wm_best_score != 1.0:
                    speedup = wm_best_score

                print(f"[WM] round_index={wm_rounds}, best_score={wm_best_score}")

        # 计算得分
        if rounds != "NA" and speedup != "NA":
            try:
                score = (1.0 / int(rounds)) * float(speedup)
                print(f"[CALC] score = (1/{rounds}) * {speedup} = {score}")
            except:
                score = "NA"

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
            result["elapsed_seconds"],
        ])

    print(f"\n[RESULT] {variant_id}:")
    print(f"  - Rounds: {rounds}")
    print(f"  - Speedup: {speedup}")
    print(f"  - Score: {score}")
    print(f"  - Status: {status}")
    print(f"  - Elapsed: {int(elapsed_total)}s")

    return result


def main():
    """运行示范实验"""
    print(f"\n{'#'*60}")
    print(f"# MQA Strategy Quick Sample Experiment")
    print(f"# Baseline: {BASELINE_MS} ms")
    print(f"# Variants: {len(SAMPLE_VARIANTS)} (sample mode)")
    print(f"# {SAMPLE_VARIANTS}")
    print(f"{'#'*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 创建结果CSV
    results_csv = RESULTS_DIR / "quick_experiment_results.csv"
    with open(results_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["variant_id", "strategy_chars", "rounds", "speedup", "score", "status", "elapsed_seconds"])

    # 运行示范实验
    all_results = []
    for i, variant_id in enumerate(SAMPLE_VARIANTS, 1):
        print(f"\n[PROGRESS] Experiment {i}/{len(SAMPLE_VARIANTS)}")
        result = run_experiment(variant_id, results_csv)
        all_results.append(result)

        # 休息
        if i < len(SAMPLE_VARIANTS):
            print(f"\n[WAIT] Waiting 10s before next experiment...")
            time.sleep(10)

    # 总结
    print(f"\n{'='*60}")
    print(f"[SUMMARY] Quick sample experiment completed!")
    print(f"[RESULTS] CSV: {results_csv}")
    print(f"{'='*60}\n")

    # 分析结果
    valid_results = [r for r in all_results if r["score"] != "NA"]
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: float(x["score"]), reverse=True)
        print(f"[ANALYSIS] Results ranking:")
        for i, r in enumerate(sorted_results, 1):
            print(f"  {i}. {r['variant_id']}: score={r['score']:.4f}, rounds={r['rounds']}, speedup={r['speedup']}")
    else:
        print(f"[WARN] No valid results obtained")

    # 保存完整JSON
    results_json = RESULTS_DIR / "quick_experiment_results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[JSON] Full results: {results_json}")
    print(f"[DONE] Quick sample experiment finished!")

    return 0


if __name__ == "__main__":
    sys.exit(main())