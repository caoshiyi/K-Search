#!/usr/bin/env bash
# K-Search launcher for any provider targeting Blackwell (B200) GPUs.
#
# Required environment variables:
#   KSEARCH_ROOT    - path to K-Search repo root
#   DATASET_ROOT    - path to flashinfer-bench dataset
#   API keys in .env.local (auto-loaded) or environment
#
# Usage examples:
#   # Claude Sonnet on Blackwell
#   PROVIDER=anthropic MODEL_NAME=claude-sonnet-4-6 bash scripts/flashinfer_claude_blackwell.sh
#
#   # Claude Opus on Blackwell
#   PROVIDER=anthropic MODEL_NAME=claude-opus-4-6 bash scripts/flashinfer_claude_blackwell.sh
#
#   # GPT-5.2 on Blackwell
#   PROVIDER=openai MODEL_NAME=gpt-5.2 bash scripts/flashinfer_claude_blackwell.sh
#
#   # Gemini on Blackwell
#   PROVIDER=google bash scripts/flashinfer_claude_blackwell.sh
#
#   # DeepSeek on Blackwell
#   PROVIDER=deepseek bash scripts/flashinfer_claude_blackwell.sh
#
#   # Any model via OpenRouter
#   PROVIDER=openrouter MODEL_NAME=anthropic/claude-sonnet-4 bash scripts/flashinfer_claude_blackwell.sh
#
# Optional overrides:
#   DEFINITION          - kernel definition (default: mla_paged_decode_h16_ckv512_kpe64_ps1)
#   LANGUAGE            - cuda | triton (default: cuda)
#   TARGET_GPU          - target GPU arch (default: B200)
#   MAX_OPT_ROUNDS      - optimization rounds (default: 20)

set -euo pipefail

KSEARCH_ROOT="${KSEARCH_ROOT:-}"
DATASET_ROOT="${DATASET_ROOT:-}"

PROVIDER="${PROVIDER:-anthropic}"
MODEL_NAME="${MODEL_NAME:-}"

DEFINITION="${DEFINITION:-mla_paged_decode_h16_ckv512_kpe64_ps1}"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-B200}"

BASELINE_SOLUTION="${BASELINE_SOLUTION:-flashinfer_wrapper_03f7b0}"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"

MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-20}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-7}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-output}"
ENV_FILE="${ENV_FILE:-${KSEARCH_ROOT}/.env.local}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ksearch-blackwell}"
RUN_NAME="${RUN_NAME:-${PROVIDER}-${MODEL_NAME:-default}-${LANGUAGE}-wm-${DEFINITION}-${TARGET_GPU}-opt${MAX_OPT_ROUNDS}}"

if [[ -z "${KSEARCH_ROOT}" ]]; then
  echo "ERROR: KSEARCH_ROOT is required" >&2
  exit 2
fi
if [[ -z "${DATASET_ROOT}" ]]; then
  echo "ERROR: DATASET_ROOT is required" >&2
  exit 2
fi

CONT_ARGS=()
if [[ -n "${CONTINUE_FROM_SOLUTION}" ]]; then
  CONT_ARGS+=(--continue-from-solution "${CONTINUE_FROM_SOLUTION}")
fi

MODEL_ARGS=()
if [[ -n "${MODEL_NAME}" ]]; then
  MODEL_ARGS+=(--model-name "${MODEL_NAME}")
fi

WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  export WANDB_API_KEY="${WANDB_API_KEY:-}"
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}" --run-name "${RUN_NAME}")
fi

python3 -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" \
  --env-file "${ENV_FILE}" \
  --local "${DATASET_ROOT}" \
  --task-source flashinfer \
  --task-path "${DATASET_ROOT}" \
  --definition "${DEFINITION}" \
  --provider "${PROVIDER}" \
  "${MODEL_ARGS[@]}" \
  --language "${LANGUAGE}" \
  --target-gpu "${TARGET_GPU}" \
  --world-model \
  --wm-stagnation-window "${WM_STAGNATION_WINDOW}" \
  --max-opt-rounds "${MAX_OPT_ROUNDS}" \
  --parallel-workloads \
  --save-solutions \
  --use-isolated-runner \
  --baseline-solution "${BASELINE_SOLUTION}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  "${CONT_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
