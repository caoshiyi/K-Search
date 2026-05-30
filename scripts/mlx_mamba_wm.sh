#!/usr/bin/env bash
set -euo pipefail

# MLX Mamba selective scan forward launcher for k-search (World Model generator).
#
# Environment variables (common):
# - KSEARCH_ROOT: path to k-search repo (default: repo root inferred from this script)
# - MODEL_NAME: LLM model name
# - LLM_API_KEY or API_KEY: OpenAI-compatible API key (required for LLM_PROVIDER=openai)
# - BASE_URL: OpenAI-compatible base url (optional)
# - LLM_PROVIDER: openai|claude-agent (default: openai)
# - ANTHROPIC_API_KEY: Claude Agent SDK key when LLM_PROVIDER=claude-agent
#
# Task/generation:
# - LANGUAGE: mlx (default: mlx)
# - MAX_OPT_ROUNDS: (default: 50)
# - ARTIFACTS_DIR: base output dir (default: .ksearch)
#
# World model:
# - WM: 1 to enable world-model prompting (default: 1)
# - WM_STAGNATION_WINDOW: (default: 5)
#
# Optional W&B:
# - WANDB: 1 to enable (default: 0)
# - WANDB_PROJECT, RUN_NAME, WANDB_API_KEY

KSEARCH_ROOT="${KSEARCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

MODEL_NAME="${MODEL_NAME:-gpt-5.2}"
LLM_PROVIDER="${LLM_PROVIDER:-openai}"
API_KEY="${API_KEY:-${LLM_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://us.api.openai.com/v1}"

LANGUAGE="${LANGUAGE:-mlx}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-50}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch}"

WM="${WM:-1}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-5}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ksearch-mlx}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${LANGUAGE}-mlx-mamba-wm-opt${MAX_OPT_ROUNDS}}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "ERROR: MODEL_NAME is required" >&2
  exit 2
fi
if [[ "${LLM_PROVIDER}" == "openai" && -z "${API_KEY}" ]]; then
  echo "ERROR: API key is required for LLM_PROVIDER=openai (set LLM_API_KEY or API_KEY)" >&2
  exit 2
fi

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

WM_ARGS=()
if [[ "${WM}" == "1" ]]; then
  WM_ARGS+=(--world-model --wm-stagnation-window "${WM_STAGNATION_WINDOW}")
fi

WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  export WANDB_API_KEY="${WANDB_API_KEY:-}"
  WANDB_ARGS+=(--wandb --run-name "${RUN_NAME}" --wandb-project "${WANDB_PROJECT}")
fi

python3 -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" \
  --task-source mlx \
  --definition "mlx_mamba_selective_scan_fwd" \
  --model-name "${MODEL_NAME}" \
  --llm-provider "${LLM_PROVIDER}" \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --language "${LANGUAGE}" \
  --max-opt-rounds "${MAX_OPT_ROUNDS}" \
  --warmup-runs 3 \
  --iterations 10 \
  --save-solutions \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  "${WM_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
