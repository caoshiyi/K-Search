#!/usr/bin/env bash
set -euo pipefail

# ShinkaEvolve analogue of `baselines/openevolve/run_evolve_moe.sh`.
# It pins:
#  - kernel definition
#  - evaluation workload UUIDs
#  - benchmark knobs
#
# by pointing to `flashinfer_evaluator_config_moe.yaml`.
#
# Notes:
# - This script expects ShinkaEvolve to be installed (providing the `shinka` package / CLI).
# - LLM credentials are not hardcoded; set them via env if your ShinkaEvolve install requires them.

REPO_ROOT="<PATH_TO_KSEARCH_REPO>"

EVALUATOR_CONFIG="${REPO_ROOT}/baselines/shinkaevolve/flashinfer_evaluator_config_moe.yaml"
INITIAL_PROGRAM="${REPO_ROOT}/baselines/shinkaevolve/flashinfer_initial.py"
EVAL_PROGRAM="${REPO_ROOT}/baselines/shinkaevolve/evaluate_flashinfer.py"
OE_CONFIG="${REPO_ROOT}/baselines/shinkaevolve/config_moe.yaml"

cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/shinkaevolve_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME_DEFAULT="shinka_moe_${TS}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_DEFAULT}}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
export OPENAI_API_KEY="DUMMY_OPENAI_API_KEY"

# ShinkaEvolve's Gemini client reads GEMINI_API_KEY (not OPENAI_API_KEY).
# For consistency with the OpenEvolve scripts in this repo (which export OPENAI_API_KEY),
# map it automatically if GEMINI_API_KEY isn't already set.
export GEMINI_API_KEY="${GEMINI_API_KEY:-${OPENAI_API_KEY:-}}"

# Match OpenEvolve script naming defaults
WANDB_PROJECT="${WANDB_PROJECT:-YOUR_WANDB_PROJECT}"
WANDB_RUN_NAME_DEFAULT="shinka_moe_${TS}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WANDB_RUN_NAME_DEFAULT}}"

# Mirror OpenEvolve's config for BOTH:
# - the spec prompt (definition/algorithm requirements)
# - LLM knobs (temperature/max_tokens)
# via `--openevolve-config`.

# Default to remote Ollama embeddings (OpenAI-compatible /v1 API).
# Assumes Ollama is already running and reachable; this script will not start/pull/stop anything.
# Override by setting EMBEDDING_MODEL directly, or set OLLAMA_BASE_URL / OLLAMA_EMBED_MODEL.
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://YOUR_OLLAMA_HOST:11434/v1}"
# Default embedding model name on your Ollama:
#   ollama pull qwen3-embedding:8b-fp16
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-qwen3-embedding:8b-fp16}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-local-${OLLAMA_EMBED_MODEL}-${OLLAMA_BASE_URL}}"

# Optional: sanity-check that Ollama is reachable (does not start anything).
CHECK_OLLAMA="${CHECK_OLLAMA:-1}"
OLLAMA_TAGS_URL="${OLLAMA_TAGS_URL:-${OLLAMA_BASE_URL%/v1}/api/tags}"
if [[ "${CHECK_OLLAMA}" == "1" ]]; then
  if ! curl -fsS "${OLLAMA_TAGS_URL}" >/dev/null 2>&1; then
    echo "[run_evolve_moe] ERROR: Ollama not reachable at ${OLLAMA_TAGS_URL}" | tee -a "${LOG_FILE}"
    echo "[run_evolve_moe] Set OLLAMA_BASE_URL (ending in /v1) or disable check with CHECK_OLLAMA=0" | tee -a "${LOG_FILE}"
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
python3 -u "${REPO_ROOT}/baselines/shinkaevolve/run_evolve.py" \
  --initial-program "${INITIAL_PROGRAM}" \
  --eval-program "${EVAL_PROGRAM}" \
  --evaluator-config "${EVALUATOR_CONFIG}" \
  --openevolve-config "${OE_CONFIG}" \
  --final-eval-all-workloads \
  --enable-wandb --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-log-file "${LOG_FILE}" \
  --num-generations "${NUM_GENERATIONS:-20}" \
  --max-parallel-jobs "${MAX_PARALLEL_JOBS:-1}" \
  --db-path "${REPO_ROOT}/shinka_output/${RUN_NAME}" \
  ${EMBEDDING_MODEL:+--embedding-model "${EMBEDDING_MODEL}"} \
  --num-runs "${NUM_RUNS:-1}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
