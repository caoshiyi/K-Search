#!/usr/bin/env bash
# K-Search optimization run for flash_attention on Ascend 910B.
#
# Pre-conditions:
#   1. Claude Agent SDK installed:   uv pip install claude-agent-sdk
#   2. flash_attention baseline measured;   export BASELINE_MS=<float>  (ascendc kernel mean_us/1000 from
#      utils/run_perf.py --mode profiler — use the ascendc column, NOT the PyTorch base column)
#   3. ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL exported, or present in
#      ~/.claude/settings.json (auto-loaded below if jq is installed).
set -euo pipefail

# --- Auto-load env from ~/.claude/settings.json if available -----------------
if command -v jq >/dev/null 2>&1 && [ -f "$HOME/.claude/settings.json" ]; then
    while IFS='=' read -r k v; do
        # Only export keys that are not already set, so caller can override.
        if [ -n "$k" ] && [ -z "${!k:-}" ]; then
            export "$k=$v"
        fi
    done < <(jq -r '.env | to_entries[] | .key + "=" + (.value|tostring)' "$HOME/.claude/settings.json")
fi

# --- Required env ------------------------------------------------------------
: "${ANTHROPIC_AUTH_TOKEN:?missing - export it or set .env.ANTHROPIC_AUTH_TOKEN in ~/.claude/settings.json}"
: "${ANTHROPIC_BASE_URL:?missing - export it or set .env.ANTHROPIC_BASE_URL in ~/.claude/settings.json}"
: "${BASELINE_MS:?run baseline first and export BASELINE_MS (mean_us/1000 from utils/run_perf.py)}"

# --- Configurable ------------------------------------------------------------
KSEARCH_ROOT="${KSEARCH_ROOT:-/mnt/workspace/K-Search}"
TASK_DIR="${TASK_DIR:-/mnt/workspace/cv_agent/tile2asc/flash_attention}"
MODEL_NAME="${MODEL_NAME:-glm-5.1}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
TARGET_GPU="${TARGET_GPU:-Ascend910B3}"
TIMEOUT_S="${TIMEOUT_S:-900}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-fa}"

# Accuracy: which case types to validate each round (comma-separated).
export KSEARCH_TEST_CASE_TYPES="${KSEARCH_TEST_CASE_TYPES:-basic,general}"
# Performance: bench case type, warmup, and repeat count.
export KSEARCH_BENCH_CASE_TYPE="${KSEARCH_BENCH_CASE_TYPE:-basic}"
export KSEARCH_WARMUP="${KSEARCH_WARMUP:-10}"
export KSEARCH_REPEAT="${KSEARCH_REPEAT:-50}"
# NPU device ID (check available devices with `npu-smi info`).
export KSEARCH_DEVICE_ID="${KSEARCH_DEVICE_ID:-3}"

# Force a long HTTP timeout for SDK sessions (overrides any pre-set value).
export API_TIMEOUT_MS=7200000
# Raise output cap for code-generation responses (default in SDK is 32000).
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-64000}"
# Cap max turns per session so the agent doesn't spin indefinitely.
export CLAUDE_AGENT_MAX_TURNS=50

# Measured baseline: 0.394ms (basic case, warmup=5, repeat=10)
export BASELINE_MS="${BASELINE_MS:-0.394}"

cd "$KSEARCH_ROOT"

# Record task start time so LLM log dirs include a unique run-level timestamp.
export KSEARCH_RUN_START="${KSEARCH_RUN_START:-$(date -u +%Y%m%d_%H%M%S)}"

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
    --max-opt-rounds "$MAX_ROUNDS" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --save-solutions