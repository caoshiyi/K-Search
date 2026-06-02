#!/usr/bin/env bash
# Run all 4 strategy-form experiments sequentially with controlled variables.
#
# Experiments:
#   1. Baseline LLM (no strategy injection)
#   2. Natural Language strategy form
#   3. Structured Params strategy form
#   4. DSL strategy form
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BASELINE_MS="${BASELINE_MS:?run baseline first and export BASELINE_MS}"

LOG_DIR="/tmp/ksearch_exp_logs"
mkdir -p "$LOG_DIR"

echo "Starting 4 sequential strategy-form experiments..."

# Experiment 1: Baseline LLM
echo ""
echo ">>> Experiment 1: Baseline LLM (pure LLM WM) <<<"
export BASELINE_MS
bash "$SCRIPT_DIR/exp_mqa_baseline_llm.sh" > "$LOG_DIR/exp_baseline_llm.log" 2>&1
echo "Experiment 1 completed."

# Experiment 2: Natural Language
echo ""
echo ">>> Experiment 2: Natural Language strategy form <<<"
bash "$SCRIPT_DIR/exp_mqa_natural_language.sh" > "$LOG_DIR/exp_natural_language.log" 2>&1
echo "Experiment 2 completed."

# Experiment 3: Structured Params
echo ""
echo ">>> Experiment 3: Structured Params strategy form <<<"
bash "$SCRIPT_DIR/exp_mqa_structured_params.sh" > "$LOG_DIR/exp_structured_params.log" 2>&1
echo "Experiment 3 completed."

# Experiment 4: DSL
echo ""
echo ">>> Experiment 4: DSL strategy form <<<"
bash "$SCRIPT_DIR/exp_mqa_dsl.sh" > "$LOG_DIR/exp_dsl.log" 2>&1
echo "Experiment 4 completed."

echo ""
echo "All 4 experiments completed. Collecting results..."
python3 "$SCRIPT_DIR/collect_experiment_results.py"

echo "Done. Check logs in $LOG_DIR/"