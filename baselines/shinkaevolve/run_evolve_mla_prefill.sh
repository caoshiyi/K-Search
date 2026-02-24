#!/usr/bin/env bash
set -euo pipefail

# ShinkaEvolve analogue of `examples/openevolve/run_evolve_mla_prefill.sh`.
#
# This mirrors the OpenEvolve approach:
# - evaluator-config YAML selects definition + feedback workloads + bench knobs
# - OpenEvolve-style config YAML provides the full spec prompt + LLM knobs

REPO_ROOT="<PATH_TO_KSEARCH_REPO>"
OE_CONFIG="${REPO_ROOT}/examples/shinkaevolve/config_mla_prefill.yaml"
EVALUATOR_CONFIG="${REPO_ROOT}/examples/shinkaevolve/flashinfer_evaluator_config_mla_prefill.yaml"
INITIAL_PROGRAM="${REPO_ROOT}/examples/shinkaevolve/flashinfer_initial.py"
EVAL_PROGRAM="${REPO_ROOT}/examples/shinkaevolve/evaluate_flashinfer.py"

cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/shinkaevolve_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME_DEFAULT="shinka_mla_prefill_${TS}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_DEFAULT}}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

export OPENAI_API_KEY="DUMMY_OPENAI_API_KEY"

# ShinkaEvolve's Gemini adapter expects GEMINI_API_KEY; map OPENAI_API_KEY for convenience.
export GEMINI_API_KEY="${GEMINI_API_KEY:-${OPENAI_API_KEY:-}}"

# Match OpenEvolve script naming defaults
WANDB_PROJECT="${WANDB_PROJECT:-YOUR_WANDB_PROJECT}"
WANDB_RUN_NAME_DEFAULT="shinka_mla_prefill_${TS}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WANDB_RUN_NAME_DEFAULT}}"

# Default to remote Ollama embeddings (OpenAI-compatible /v1 API).
# Assumes Ollama is already running and reachable; this script will not start/pull/stop anything.
# Override by setting EMBEDDING_MODEL directly, or set OLLAMA_BASE_URL.
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://YOUR_OLLAMA_HOST:11434/v1}"
# Default Ollama embedding model name.
# You confirmed this is pullable and supports embeddings on your setup:
#   ollama pull qwen3-embedding:8b-fp16
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-qwen3-embedding:8b-fp16}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-local-${OLLAMA_EMBED_MODEL}-${OLLAMA_BASE_URL}}"

CUDA_VISIBLE_DEVICES_DEFAULT="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Optional: sanity-check that Ollama is reachable (does not start anything).
CHECK_OLLAMA="${CHECK_OLLAMA:-1}"
OLLAMA_TAGS_URL="${OLLAMA_TAGS_URL:-${OLLAMA_BASE_URL%/v1}/api/tags}"
if [[ "${CHECK_OLLAMA}" == "1" ]]; then
  if ! curl -fsS "${OLLAMA_TAGS_URL}" >/dev/null 2>&1; then
    echo "[run_evolve_mla_prefill] ERROR: Ollama not reachable at ${OLLAMA_TAGS_URL}" | tee -a "${LOG_FILE}"
    echo "[run_evolve_mla_prefill] Set OLLAMA_BASE_URL (ending in /v1) or disable check with CHECK_OLLAMA=0" | tee -a "${LOG_FILE}"
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_DEFAULT}" \
python3 -u "${REPO_ROOT}/examples/shinkaevolve/run_evolve.py" \
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
