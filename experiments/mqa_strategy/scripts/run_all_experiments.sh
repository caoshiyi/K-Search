#!/usr/bin/env bash
# Run currently supported strategy experiments sequentially with controlled variables.
#
# Experiments:
#   1. Baseline LLM (no strategy injection)
#   2. Natural Language strategy form
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BASELINE_MS="${BASELINE_MS:?run baseline first and export BASELINE_MS}"

LOG_DIR="/tmp/ksearch_exp_logs"
mkdir -p "$LOG_DIR"

echo "Starting supported strategy experiments..."

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

echo ""
echo "Supported experiments completed. Collecting results..."
python3 "$SCRIPT_DIR/collect_experiment_results.py"

echo "Done. Check logs in $LOG_DIR/"
