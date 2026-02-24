#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="<PATH_TO_KSEARCH_REPO>"

OE_CONFIG="${REPO_ROOT}/baselines/openevolve/config_moe.yaml"
EVALUATOR_CONFIG="${REPO_ROOT}/baselines/openevolve/flashinfer_evaluator_config_moe.yaml"
INITIAL_PROGRAM="${REPO_ROOT}/baselines/openevolve/initial_empty.txt"

# Require non-interactive credentials from env (avoid W&B prompt/hang)
export OPENAI_API_KEY="DUMMY_OPENAI_API_KEY"
WANDB_PROJECT="${WANDB_PROJECT:-YOUR_WANDB_PROJECT}"

cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/openevolve_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME_DEFAULT="oe_moe_${TS}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WANDB_RUN_NAME_DEFAULT}}"
LOG_FILE="${LOG_DIR}/${WANDB_RUN_NAME}.log"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
python3 -u "${REPO_ROOT}/baselines/openevolve/run_evolve.py" \
  --config "${OE_CONFIG}" \
  --evaluator-config "${EVALUATOR_CONFIG}" \
  --initial-program "${INITIAL_PROGRAM}" \
  --final-eval-all-workloads \
  --enable-wandb --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${WANDB_RUN_NAME}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
