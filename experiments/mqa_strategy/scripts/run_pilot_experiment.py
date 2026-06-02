#!/usr/bin/env python3
"""
run_pilot_experiment.py - 验证实验脚本

执行单个策略验证流程，确保能正确获得真实加速比
预计耗时：4-5小时

使用方法：
    python3 scripts/run_pilot_experiment.py

或者直接执行命令：
    python3 generate_kernels_and_eval.py \
        --kernel multi_query_attention \
        --strategy-file strategies/mqa_experiments/S1-V1.json \
        --strategy-form natural_language \
        --output-dir .ksearch-exp-mqa-pilot/S1-V1 \
        --max-rounds 3
"""

import os
import subprocess
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

KSEARCH_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ".ksearch-exp-mqa-pilot"
STRATEGY_FILE = str(KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments" / "S1-V1.json")
TIMEOUT_HOURS = 5
TASK_DIR = os.environ.get("TASK_DIR", "/mnt/workspace/cv_agent/tile2asc/multi_query_attention")
BASELINE_MS = 0.834548


def check_progress(output_dir: str) -> dict:
    """检查实验进度"""
    base_path = Path(output_dir) / "multi_query_attention"

    if not base_path.exists():
        return {"stage": "init", "rounds": 0, "completed": False}

    # 检查是否完成
    eval_reports = list(base_path.glob("eval/**/eval_report_*.json"))
    if eval_reports:
        return {"stage": "completed", "rounds": -1, "completed": True}

    # 检查snapshots数量
    snapshots = list(base_path.glob("snapshots/round_*"))
    rounds = len(snapshots)

    # 检查candidates
    candidates = list(base_path.glob("candidates/*"))

    if rounds > 0:
        stage = "optimizing"
    elif candidates:
        stage = "codegen"
    else:
        stage = "init"

    return {"stage": stage, "rounds": rounds, "completed": False}


def extract_real_speedup(output_dir: str) -> float | None:
    """从eval_report提取真实speedup"""
    eval_dir = Path(output_dir) / "multi_query_attention" / "eval" / "multi_query_attention"

    if not eval_dir.exists():
        print(f"❌ eval目录不存在：{eval_dir}")
        return None

    eval_reports = list(eval_dir.glob("eval_report_*.json"))

    if not eval_reports:
        print(f"❌ eval_report不存在：{output_dir}")
        return None

    report = json.loads(eval_reports[0].read_text())

    # 验证状态
    result = report["results"][0]["result"]
    status = result["status"]

    if status != "passed":
        print(f"❌ 实验失败：status={status}")
        return None

    # 提取真实值
    speedup = result.get("speedup_factor")
    latency_ms = result.get("latency_ms")

    if speedup is None:
        print(f"❌ speedup_factor为null")
        return None

    print(f"✅ 真实数据：speedup={speedup:.4f}, latency={latency_ms:.4f}ms")
    return float(speedup)


def run_pilot():
    """执行验证实验"""
    print("=" * 70)
    print("# Phase 1: 验证实验")
    print("# 目标：确保能正确获得真实加速比")
    print("# 预计耗时：4-5小时")
    print("=" * 70)

    # 清理旧数据
    print("\n[清理] 删除之前的错误数据...")
    for pattern in [".ksearch-exp-mqa-S*-series", ".ksearch-exp-mqa/remaining_*"]:
        for d in Path(".").glob(pattern):
            if d.exists():
                shutil.rmtree(d)
                print(f"  删除：{d}")

    # 确保策略文件存在
    if not Path(STRATEGY_FILE).exists():
        print(f"❌ 策略文件不存在：{STRATEGY_FILE}")
        return None

    # 启动实验
    print(f"\n[启动] 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[策略] {STRATEGY_FILE}")
    print(f"[输出] {OUTPUT_DIR}/S1-V1")

    cmd = [
        "python", "-u", "generate_kernels_and_eval.py",
        "--task-source", "ascendc",
        "--task-path", TASK_DIR,
        "--definition", "multi_query_attention",
        "--model-name", "glm-5.1",
        "--llm-provider", "claude-agent",
        "--language", "ascendc",
        "--target-gpu", "Ascend910B3",
        "--ascendc-build-cmd", "./ksearch_build.sh",
        "--ascendc-test-cmd", "./ksearch_test.sh",
        "--ascendc-bench-cmd", "./ksearch_bench.sh",
        "--ascendc-reference-latency-ms", str(BASELINE_MS),
        "--ascendc-timeout-seconds", "300",
        "--world-model",
        "--strategy-file", STRATEGY_FILE,
        "--strategy-form", "natural_language",
        "--max-opt-rounds", "3",
        "--artifacts-dir", f"{OUTPUT_DIR}/S1-V1",
        "--save-solutions",
    ]

    print(f"[命令] {' '.join(cmd[:10])}...")
    print()

    import os
    env = dict(os.environ)
    env["BASELINE_MS"] = str(BASELINE_MS)

    process = subprocess.Popen(cmd, env=env)

    # 监控进度
    start_time = time.time()
    for hour in range(TIMEOUT_HOURS):
        time.sleep(3600)  # 每小时检查
        elapsed = (time.time() - start_time) / 3600

        progress = check_progress(f"{OUTPUT_DIR}/S1-V1")
        print(f"[进度] {elapsed:.1f}h: stage={progress['stage']}, rounds={progress['rounds']}")

        if progress["completed"]:
            print(f"\n✅ 实验完成！耗时：{elapsed:.1f}小时")
            break

        # 检查进程状态
        if process.poll() is not None:
            print(f"\n进程已结束，返回码：{process.returncode}")
            break

    # 检查是否超时
    if process.poll() is None:
        print(f"\n⚠️ 实验超时（{TIMEOUT_HOURS}小时），继续等待...")
        # 不终止进程，让它继续运行

    # 提取结果
    print("\n[检查] 等待eval_report生成...")
    time.sleep(60)  # 等待文件写入完成

    speedup = extract_real_speedup(f"{OUTPUT_DIR}/S1-V1")

    if speedup is None:
        print("\n❌ 验证失败：无法获取真实加速比")
        print("请检查日志：")
        print(f"  - {OUTPUT_DIR}/S1-V1/multi_query_attention/")
        return None

    print(f"\n✅ 验证成功：真实speedup = {speedup:.4f}")

    # 保存结果
    result_file = Path(f"{OUTPUT_DIR}/pilot_result.json")
    result_data = {
        "strategy": "S1-V1",
        "speedup": speedup,
        "timestamp": datetime.now().isoformat(),
        "output_dir": f"{OUTPUT_DIR}/S1-V1"
    }
    result_file.write_text(json.dumps(result_data, indent=2))
    print(f"[保存] 结果保存到：{result_file}")

    return speedup


if __name__ == "__main__":
    speedup = run_pilot()
    if speedup:
        print(f"\n" + "=" * 70)
        print("验证完成，真实speedup={:.4f}".format(speedup))
        print("可以进入Phase 2批量实验")
        print("=" * 70)
    else:
        print("\n验证失败，请检查问题后重试")