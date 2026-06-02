#!/usr/bin/env python3
"""
check_experiment_progress.py - 检查实验进度

检查实验是否完成，提取真实加速比

使用方法：
    python3 scripts/check_experiment_progress.py .ksearch-exp-mqa-pilot/S1-V1
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def check_progress(output_dir: str) -> dict:
    """检查实验进度"""
    base_path = Path(output_dir) / "multi_query_attention"

    if not base_path.exists():
        return {
            "exists": False,
            "stage": "init",
            "rounds": 0,
            "completed": False,
            "message": "输出目录不存在"
        }

    # 检查是否完成
    eval_reports = list(base_path.glob("eval/**/eval_report_*.json"))

    if eval_reports:
        # 实验完成，提取结果
        report = json.loads(eval_reports[0].read_text())
        result = report["results"][0]["result"]

        return {
            "exists": True,
            "stage": "completed",
            "rounds": -1,
            "completed": True,
            "status": result["status"],
            "speedup": result.get("speedup_factor"),
            "latency_ms": result.get("latency_ms"),
            "solution": report["best"]["solution"],
            "message": "实验完成"
        }

    # 检查snapshots数量（判断round进度）
    snapshots = list(base_path.glob("snapshots/round_*"))
    rounds = len(snapshots)

    # 检查candidates
    candidates = list(base_path.glob("candidates/*"))
    candidates_count = len(candidates)

    # 检查world_model大小
    wm_path = base_path / "world_model" / "world_model.json"
    wm_size = wm_path.stat().st_size if wm_path.exists() else 0

    # 判断当前阶段
    if rounds > 0:
        stage = "optimizing"
    elif candidates_count > 0:
        stage = "codegen"
    elif wm_size > 1000:
        stage = "world_model_init"
    else:
        stage = "init"

    return {
        "exists": True,
        "stage": stage,
        "rounds": rounds,
        "candidates": candidates_count,
        "world_model_size": wm_size,
        "completed": False,
        "message": f"实验进行中：{stage}"
    }


def main():
    if len(sys.argv) < 2:
        print("用法：python3 scripts/check_experiment_progress.py <output_dir>")
        print("示例：python3 scripts/check_experiment_progress.py .ksearch-exp-mqa-pilot/S1-V1")
        sys.exit(1)

    output_dir = sys.argv[1]

    print(f"[检查] {output_dir}")
    print(f"[时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    progress = check_progress(output_dir)

    print("=" * 60)
    print("实验状态")
    print("=" * 60)

    for key, value in progress.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print()

    if progress["completed"]:
        print("✅ 实验完成！")

        if progress["status"] == "passed":
            print(f"真实speedup: {progress['speedup']:.4f}")
            print(f"latency_ms: {progress['latency_ms']:.4f}")
        else:
            print(f"⚠️ 实验状态：{progress['status']}")
    else:
        print(f"⏳ {progress['message']}")

    # 返回JSON格式（便于脚本解析）
    print()
    print("=" * 60)
    print("JSON输出")
    print("=" * 60)
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()