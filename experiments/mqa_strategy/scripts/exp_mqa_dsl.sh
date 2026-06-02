#!/usr/bin/env bash
# K-Search strategy experiment: DSL form
#
# Controlled variables identical to exp_mqa_natural_language.sh
# Independent variable: strategy form = dsl
#
set -euo pipefail

# --- Auto-load env ---
if command -v jq >/dev/null 2>&1 && [ -f "$HOME/.claude/settings.json" ]; then
    while IFS='=' read -r k v; do
        if [ -n "$k" ] && [ -z "${!k:-}" ]; then
            export "$k=$v"
        fi
    done < <(jq -r '.env | to_entries[] | .key + "=" + (.value|tostring)' "$HOME/.claude/settings.json")
fi

: "${ANTHROPIC_AUTH_TOKEN:?missing}"
: "${ANTHROPIC_BASE_URL:?missing}"
: "${BASELINE_MS:?run baseline first}"

# --- Fixed experiment parameters (identical to nl/sp experiments) ---
KSEARCH_ROOT="${KSEARCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TASK_DIR="${TASK_DIR:-/mnt/workspace/cv_agent/tile2asc/multi_query_attention}"
MODEL_NAME="glm-5.1"
MAX_ROUNDS=12
TARGET_GPU="Ascend910B3"
TIMEOUT_S=900
STAGNATION_WINDOW=5
STRATEGY_CATALOG="${KSEARCH_ROOT}/strategies/mqa_strategies_catalog.json"

# --- Independent variable: strategy form ------------------------------------
STRATEGY_FORM="dsl"

# --- Experiment-specific output dir ------------------------------------------
EXPERIMENT_ID="mqa_strat_dsl_$(date -u +%Y%m%d_%H%M%S)"
ARTIFACTS_DIR=".ksearch-exp-${EXPERIMENT_ID}"

export API_TIMEOUT_MS=7200000
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-64000}"
export CLAUDE_AGENT_MAX_TURNS=50
export KSEARCH_RUN_START="${KSEARCH_RUN_START:-$(date -u +%Y%m%d_%H%M%S)}"

cd "$KSEARCH_ROOT"

echo "============================================"
echo "Strategy Experiment: DSL form"
echo "============================================"
echo "  Task:          $(basename "$TASK_DIR")"
echo "  Model:         $MODEL_NAME"
echo "  Max Rounds:    $MAX_ROUNDS"
echo "  Strategy Form: $STRATEGY_FORM"
echo "  Baseline:      ${BASELINE_MS}ms"
echo "  Artifacts Dir: $ARTIFACTS_DIR"
echo "============================================"

python -u generate_kernels_and_eval.py \
    --task-source ascendc \
    --task-path "$TASK_DIR" \
    --definition "$(basename "$TASK_DIR")" \
    --model-name "$MODEL_NAME" \
    --llm-provider claude-agent \
    --language ascendc \
    --target-gpu "$TARGET_GPU" \
    --ascendc-build-cmd  "./ksearch_build.sh" \
    --ascendc-test-cmd   "./ksearch_test.sh" \
    --ascendc-bench-cmd  "./ksearch_bench.sh" \
    --ascendc-reference-latency-ms "$BASELINE_MS" \
    --ascendc-timeout-seconds "$TIMEOUT_S" \
    --world-model \
    --wm-stagnation-window "$STAGNATION_WINDOW" \
    --strategy-file "$STRATEGY_CATALOG" \
    --strategy-form "$STRATEGY_FORM" \
    --max-opt-rounds "$MAX_ROUNDS" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --save-solutions

echo "Experiment $EXPERIMENT_ID completed. Results in $ARTIFACTS_DIR"