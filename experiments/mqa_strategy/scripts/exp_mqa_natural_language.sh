#!/usr/bin/env bash
# K-Search strategy experiment: NATURAL_LANGUAGE form
#
# This script runs MQA optimization with the supported v2 strategy injection
# model: natural-language markdown referenced by a concise JSON catalog.
#
# Controlled variables:
#   - Same MQA task directory
#   - Same model (glm-5.1)
#   - Same max rounds (12)
#   - Same target GPU (Ascend910B3)
#   - Same timeout (900s)
#   - Same stagnation window (5)
#   - Same v2 strategy catalog file
#   - Same baseline reference
#
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
: "${ANTHROPIC_AUTH_TOKEN:?missing - export it or set .env.ANTHROPIC_AUTH_TOKEN in ~/.claude/settings.json}"
: "${ANTHROPIC_BASE_URL:?missing - export .env.ANTHROPIC_BASE_URL in ~/.claude/settings.json}"
: "${BASELINE_MS:?run baseline first and export BASELINE_MS}"

# --- Fixed experiment parameters --------------------------------------------
KSEARCH_ROOT="${KSEARCH_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
TASK_DIR="${TASK_DIR:-/mnt/workspace/cv_agent/tile2asc/multi_query_attention}"
MODEL_NAME="glm-5.1"
MAX_ROUNDS=12
TARGET_GPU="Ascend910B3"
TIMEOUT_S=900
STAGNATION_WINDOW=5
STRATEGY_CATALOG="${KSEARCH_ROOT}/strategies/mqa_strategies_catalog.json"

# --- Experiment-specific output dir ------------------------------------------
EXPERIMENT_ID="mqa_strat_nl_$(date -u +%Y%m%d_%H%M%S)"
ARTIFACTS_DIR=".ksearch-exp-${EXPERIMENT_ID}"

# Force a long HTTP timeout for SDK sessions
export API_TIMEOUT_MS=7200000
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-64000}"
export CLAUDE_AGENT_MAX_TURNS=50
export KSEARCH_RUN_START="${KSEARCH_RUN_START:-$(date -u +%Y%m%d_%H%M%S)}"

cd "$KSEARCH_ROOT"

echo "============================================"
echo "Strategy Experiment: NATURAL_LANGUAGE form"
echo "============================================"
echo "  Task:          $(basename "$TASK_DIR")"
echo "  Model:         $MODEL_NAME"
echo "  Max Rounds:    $MAX_ROUNDS"
echo "  Strategy Form: natural_language"
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
    --max-opt-rounds "$MAX_ROUNDS" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --save-solutions

echo "Experiment $EXPERIMENT_ID completed. Results in $ARTIFACTS_DIR"
