#!/usr/bin/env bash
# MQA策略变体批量实验脚本
#
# 用法:
#   ./scripts/run_mqa_strategy_experiments.sh [--baseline-ms <ms>] [--sample]
#
# --baseline-ms: baseline性能(ms)，必须提供或通过环境变量BASELINE_MS设置
# --sample: 仅运行示范性小规模实验（3-5个变体）而非完整实验
#
set -euo pipefail

# 解析参数
SAMPLE_MODE=false
BASELINE_MS_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-ms)
            BASELINE_MS_ARG="$2"
            shift 2
            ;;
        --sample)
            SAMPLE_MODE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# 设置baseline
if [[ -n "$BASELINE_MS_ARG" ]]; then
    export BASELINE_MS="$BASELINE_MS_ARG"
elif [[ -z "${BASELINE_MS:-}" ]]; then
    echo "ERROR: BASELINE_MS must be provided via --baseline-ms or environment variable"
    echo "To measure baseline:"
    echo "  cd <task_dir>"
    echo "  python3 <task_dir>/utils/run_perf.py --phase bench --task-dir . --case-type basic"
    echo "  export BASELINE_MS=<mean_us/1000>"
    exit 1
fi

KSEARCH_ROOT="${KSEARCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TASK_DIR="${TASK_DIR:-/mnt/workspace/cv_agent/tile2asc/multi_query_attention}"
STRATEGIES_DIR="$KSEARCH_ROOT/experiments/mqa_strategy/strategies/mqa_experiments"
RESULTS_DIR="$KSEARCH_ROOT/.ksearch-exp-mqa-strategies"

# 创建策略目录
mkdir -p "$STRATEGIES_DIR"
mkdir -p "$RESULTS_DIR"

# 定义实验变体（基于docs/mqa_strategy_experiment_plan.md）
if [[ "$SAMPLE_MODE" == "true" ]]; then
    # 示范模式：仅测试几个代表性变体
    VARIANTS=(
        "S1-V1:Enlarged Tile Sizes - Minimal"
        "S1-V5:Enlarged Tile Sizes - Anti Patterns"
        "S4-V6:Vectorized RowMuls/RowDivs - Code Examples"
        "S8-V1:Targeted SetWaitFlag - Minimal"
        "S10-V4:Multi-level KV Outer Tiling - Usage Scenarios"
    )
    echo "[EXPERIMENT] Running SAMPLE mode with ${#VARIANTS[@]} variants"
else
    # 完整模式：测试所有15个变体
    VARIANTS=(
        "S1-V1:Enlarged Tile Sizes - Minimal"
        "S1-V2:Enlarged Tile Sizes - DSL"
        "S1-V4:Enlarged Tile Sizes - Usage Scenarios"
        "S1-V5:Enlarged Tile Sizes - Anti Patterns"
        "S2-V1:WorkspaceQueue Pattern - Minimal"
        "S2-V3:WorkspaceQueue Pattern - Priorities"
        "S2-V7:WorkspaceQueue Pattern - Principles"
        "S4-V1:Vectorized RowMuls/RowDivs - Minimal"
        "S4-V5:Vectorized RowMuls/RowDivs - Anti Patterns"
        "S4-V6:Vectorized RowMuls/RowDivs - Code Examples"
        "S8-V1:Targeted SetWaitFlag - Minimal"
        "S8-V4:Targeted SetWaitFlag - Usage Scenarios"
        "S8-V5:Targeted SetWaitFlag - Anti Patterns"
        "S10-V1:Multi-level KV Outer Tiling - Minimal"
        "S10-V4:Multi-level KV Outer Tiling - Usage Scenarios"
        "S10-V7:Multi-level KV Outer Tiling - Principles"
    )
    echo "[EXPERIMENT] Running FULL mode with ${#VARIANTS[@]} variants"
fi

# 生成策略文件（从mqa_strategy_variants_test.json提取）
echo "[STAGE] Generating individual strategy files..."
python3 "$KSEARCH_ROOT/experiments/mqa_strategy/scripts/extract_strategy_variants.py" \
    --source "$KSEARCH_ROOT/experiments/mqa_strategy/strategies/mqa_strategy_variants_test.json" \
    --output-dir "$STRATEGIES_DIR"

# 运行实验并收集结果
RESULTS_CSV="$RESULTS_DIR/experiment_results.csv"
echo "variant_id,variant_name,chars,rounds,speedup,score,status" > "$RESULTS_CSV"

for variant_info in "${VARIANTS[@]}"; do
    variant_id="${variant_info%%:*}"
    variant_name="${variant_info##*:}"
    strategy_file="$STRATEGIES_DIR/${variant_id}.json"

    if [[ ! -f "$strategy_file" ]]; then
        echo "[ERROR] Strategy file not found: $strategy_file"
        echo "$variant_id,$variant_name,NA,NA,NA,NA,strategy_file_missing" >> "$RESULTS_CSV"
        continue
    fi

    echo "[EXPERIMENT] Testing variant: $variant_id - $variant_name"

    # 计算策略长度
    strategy_chars=$(wc -c < "$strategy_file")

    # 运行K-Search优化
    output_dir="$RESULTS_DIR/${variant_id}_natural_language"
    export KSEARCH_RUN_START=$(date -u +%Y%m%d_%H%M%S)

    # 调用generate_kernels_and_eval.py（通过ascendc_mqa_wm.sh的参数）
    cd "$KSEARCH_ROOT"
    timeout 3600 python -u generate_kernels_and_eval.py \
        --task-source ascendc \
        --task-path "$TASK_DIR" \
        --definition "multi_query_attention" \
        --model-name glm-5.1 \
        --llm-provider claude-agent \
        --language ascendc \
        --target-gpu Ascend910B3 \
        --ascendc-build-cmd "./ksearch_build.sh" \
        --ascendc-test-cmd "./ksearch_test.sh" \
        --ascendc-bench-cmd "./ksearch_bench.sh" \
        --ascendc-reference-latency-ms "$BASELINE_MS" \
        --ascendc-timeout-seconds 900 \
        --world-model \
        --strategy-file "$strategy_file" \
        --strategy-form natural_language \
        --max-opt-rounds 10 \
        --artifacts-dir "$output_dir" \
        --save-solutions \
        2>&1 | tee "$output_dir/experiment_log.txt" || true

    # 解析结果
    rounds=$(grep -o "round [0-9]+" "$output_dir/experiment_log.txt" | tail -1 | grep -o "[0-9]+" || echo "NA")
    speedup=$(grep -o "speedup.*[0-9.]+" "$output_dir/experiment_log.txt" | tail -1 | grep -o "[0-9.]+" || echo "NA")

    # 计算得分
    if [[ "$rounds" != "NA" && "$speedup" != "NA" ]]; then
        score=$(python3 -c "print( (1.0 / $rounds) * $speedup )")
        status="completed"
    else
        score="NA"
        status="failed"
    fi

    # 记录结果
    echo "$variant_id,$variant_name,$strategy_chars,$rounds,$speedup,$score,$status" >> "$RESULTS_CSV"
    echo "[RESULT] $variant_id: rounds=$rounds, speedup=$speedup, score=$score"
done

# 输出总结
echo ""
echo "[SUMMARY] Experiment results saved to: $RESULTS_CSV"
cat "$RESULTS_CSV"

# 排序并输出最佳策略
if command -v python3 >/dev/null 2>&1; then
    echo ""
    echo "[ANALYSIS] Top 5 strategies by score:"
    python3 -c "
import csv
with open('$RESULTS_CSV') as f:
    reader = csv.DictReader(f)
    results = [r for r in reader if r['score'] != 'NA']
    sorted_results = sorted(results, key=lambda x: float(x['score']), reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f'{i}. {r[\"variant_id\"]}: score={r[\"score\"]}, rounds={r[\"rounds\"]}, speedup={r[\"speedup\"]}')
"
fi

echo "[DONE] MQA strategy experiment completed!"