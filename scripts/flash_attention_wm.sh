#!/usr/bin/env bash
# flash_attention 性能优化脚本
#
# 功能：对 flash_attention 算子执行 K-Search 性能优化
#
# 前置条件：
#   1. Claude Agent SDK 已安装：uv pip install claude-agent-sdk
#   2. ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL 已设置
#   3. NPU 设备可用（使用设备 0）
#
# 用法：
#   ./scripts/flash_attention_wm.sh
#   BASELINE_MS=0.5 ./scripts/flash_attention_wm.sh  # 跳过 baseline 测量

set -euo pipefail

# --- Auto-load env from ~/.claude/settings.json if available -----------------
if command -v jq >/dev/null 2>&1 && [ -f "$HOME/.claude/settings.json" ]; then
    while IFS='=' read -r k v; do
        if [ -n "$k" ] && [ -z "${!k:-}" ]; then
            export "$k=$v"
        fi
    done < <(jq -r '.env | to_entries[] | .key + "=" + (.value|tostring)' "$HOME/.claude/settings.json")
fi

# --- Required env ------------------------------------------------------------
: "${ANTHROPIC_AUTH_TOKEN:?missing - export it or set in ~/.claude/settings.json}"
: "${ANTHROPIC_BASE_URL:?missing - export it or set in ~/.claude/settings.json}"

# --- Configurable ------------------------------------------------------------
KSEARCH_ROOT="${KSEARCH_ROOT:-/mnt/workspace/K-Search}"
WORKDIR="${WORKDIR:-/mnt/workspace/cv_agent_adv/agent_workdir}"
TASK_NAME="${TASK_NAME:-flash_attention}"
MODEL_NAME="${MODEL_NAME:-glm-5.1}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
TARGET_GPU="${TARGET_GPU:-Ascend910B3}"
TIMEOUT_S="${TIMEOUT_S:-900}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-fa}"

# Performance: warmup and repeat count
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-20}"
DEVICE_ID="${DEVICE_ID:-0}"

# API and agent settings
export API_TIMEOUT_MS=7200000
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-64000}"
export CLAUDE_AGENT_MAX_TURNS=50
export KSEARCH_USE_AGENT_TOOL_ALLOWLIST=0
export KSEARCH_KEEP_AGENTIC_WORKTREES=1

# Run timestamp for unique log directories
export KSEARCH_RUN_START="${KSEARCH_RUN_START:-$(date -u +%Y%m%d_%H%M%S)}"

# --- Measure baseline performance --------------------------------------------
measure_baseline() {
    echo "=== Phase 2: 测量 Baseline 性能 ==="

    cd "$WORKDIR"

    # Step 2.1: 编译 + 运行性能测试
    echo "[baseline] 编译并运行性能测试..."
    local OUTPUT
    OUTPUT=$(bash scripts/evaluate_performance.sh "$TASK_NAME" ascendc "$WARMUP" "$REPEAT" "$DEVICE_ID" 2>&1)

    # Step 2.2: 解析延迟值（输出单位是 us，转换为 ms）
    # 输出格式: mean=607.140us  min=590.620us  max=710.340us
    local BASELINE_US
    BASELINE_US=$(echo "$OUTPUT" | sed -n 's/.*mean=\([0-9.]\+\)us.*/\1/p')

    if [ -z "$BASELINE_US" ]; then
        echo "错误: 无法从性能测试输出中解析延迟值"
        echo "输出内容:"
        echo "$OUTPUT"
        echo ""
        echo "请手动测量 baseline 并设置: BASELINE_MS=<值> ./scripts/flash_attention_wm.sh"
        return 1
    fi

    # 转换 us -> ms
    BASELINE_MS=$(echo "$BASELINE_US" | awk '{printf "%.3f", $1/1000}')

    echo "[baseline] 基准性能: ${BASELINE_MS}ms (来自 ${BASELINE_US}us)"
    export BASELINE_MS
}

# --- Run K-Search optimization ------------------------------------------------
run_ksearch_optimization() {
    echo "=== Phase 3: 执行 K-Search 优化 (${MAX_ROUNDS} 轮) ==="

    cd "$KSEARCH_ROOT"

    # 构建命令参数 - 直接调用 cv_agent_adv 的脚本
    local BUILD_CMD="bash $WORKDIR/scripts/evaluate_ascendc.sh $TASK_NAME basic"
    local TEST_CMD="ASCENDC_SKIP_BUILD=1 bash $WORKDIR/scripts/evaluate_ascendc.sh $TASK_NAME basic"
    local BENCH_CMD="bash $WORKDIR/scripts/evaluate_performance.sh $TASK_NAME ascendc $WARMUP $REPEAT $DEVICE_ID"

    python -u generate_kernels_and_eval.py \
        --task-source ascendc \
        --task-path "$WORKDIR/$TASK_NAME" \
        --definition "$TASK_NAME" \
        --model-name "$MODEL_NAME" \
        --llm-provider claude-agent \
        --language ascendc \
        --target-gpu "$TARGET_GPU" \
        --ascendc-build-cmd "$BUILD_CMD" \
        --ascendc-test-cmd "$TEST_CMD" \
        --ascendc-bench-cmd "$BENCH_CMD" \
        --ascendc-reference-latency-ms "$BASELINE_MS" \
        --ascendc-timeout-seconds "$TIMEOUT_S" \
        --world-model \
        --strategy-file "$KSEARCH_ROOT/strategies/flash_attention_round_design_strategies/catalog.json" \
        --max-opt-rounds "$MAX_ROUNDS" \
        --artifacts-dir "$ARTIFACTS_DIR" \
        --save-solutions
}

# --- Main execution -----------------------------------------------------------
main() {
    echo "=== flash_attention 性能优化脚本 ==="
    echo "K-Search 根目录: $KSEARCH_ROOT"
    echo "目标算子目录:   $WORKDIR/$TASK_NAME"
    echo "优化轮数:       $MAX_ROUNDS"
    echo "输出目录:       $ARTIFACTS_DIR"
    echo "设备 ID:        $DEVICE_ID"
    echo "运行时间戳:     ${KSEARCH_RUN_START}"
    echo ""

    # Phase 1: 环境检查（API Token）
    echo "=== Phase 1: 环境检查 ==="
    echo "ANTHROPIC_AUTH_TOKEN: 已设置"
    echo "ANTHROPIC_BASE_URL: ${ANTHROPIC_BASE_URL}"

    # Phase 2: Baseline 测量（如果未提供）
    if [ -z "${BASELINE_MS:-}" ]; then
        measure_baseline || exit 1
    else
        echo "=== Phase 2: 使用预设 Baseline (${BASELINE_MS}ms) ==="
    fi

    # Phase 3: 执行优化
    run_ksearch_optimization

    # Phase 4: 输出结果位置
    echo ""
    echo "=== Phase 4: 完成 ==="
    echo "优化结果保存在: ${KSEARCH_ROOT}/${ARTIFACTS_DIR}"
}

main "$@"
