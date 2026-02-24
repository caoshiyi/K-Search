#!/usr/bin/env bash
set -euo pipefail

# OpenEvolve runner for MLA prefill (mirrors `run_evolve_mla_decode.sh`).

REPO_ROOT="<PATH_TO_KSEARCH_REPO>"

OE_CONFIG="${REPO_ROOT}/examples/openevolve/config_mla_prefill.yaml"
EVALUATOR_CONFIG="${REPO_ROOT}/examples/openevolve/flashinfer_evaluator_config_mla_prefill.yaml"
INITIAL_PROGRAM="${REPO_ROOT}/examples/openevolve/initial_empty.txt"

# Credentials: prefer OPENAI_API_KEY, fall back to LLM_API_KEY (like your cluster YAMLs)
export OPENAI_API_KEY="DUMMY_OPENAI_API_KEY"

WANDB_PROJECT="${WANDB_PROJECT:-YOUR_WANDB_PROJECT}"

cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/openevolve_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME_DEFAULT="oe_mla_prefill_${TS}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WANDB_RUN_NAME_DEFAULT}}"
LOG_FILE="${LOG_DIR}/${WANDB_RUN_NAME}.log"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
python3 -u "${REPO_ROOT}/examples/openevolve/run_evolve.py" \
  --config "${OE_CONFIG}" \
  --evaluator-config "${EVALUATOR_CONFIG}" \
  --initial-program "${INITIAL_PROGRAM}" \
  --final-eval-all-workloads \
  --enable-wandb --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${WANDB_RUN_NAME}" \
  "$@" 2>&1 | tee "${LOG_FILE}"

