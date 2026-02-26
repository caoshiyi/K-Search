# GPUMode TriMul launcher for k-search (World Model generator).
#
# Environment variables (common):
# - KSEARCH_ROOT: path to k-search repo (default: /mnt/cluster_storage/k-search)
# - MODEL_NAME: LLM model name (required unless passed via env)
# - LLM_API_KEY or API_KEY: OpenAI-compatible API key (required)
# - BASE_URL: OpenAI-compatible base url (optional)
#
# Environment variables (task/generation):
# - LANGUAGE: triton|python|cuda (default: triton)
# - TARGET_GPU: e.g. H100 (default: H100)
# - MAX_OPT_ROUNDS: (default: 5)
# - ARTIFACTS_DIR: base output dir (default: .ksearch-output)
# - CONTINUE_FROM_SOLUTION: optional solution name or path to a persisted solution .json
#   (if set, resumes optimization from that solution)
#
# World model options:
# - WM: 1 to enable world-model prompting (default: 1)
# - WM_STAGNATION_WINDOW: end an action cycle after this many consecutive non-improving rounds (default: 5)
#
# Environment variables (gpumode):
# - GPUMODE_MODE: benchmark|test|profile|leaderboard (default: benchmark)
# - GPUMODE_KEEP_TMP: 1 to keep tmp dirs (default: 0)
# - GPUMODE_TASK_DIR: override task dir (default: vendored trimul task)
#
# Optional W&B:
# - WANDB: 1 to enable (default: 0)
# - WANDB_PROJECT, RUN_NAME

KSEARCH_ROOT="${KSEARCH_ROOT:-}"

# MODEL_NAME="${MODEL_NAME:-gemini-3-pro-preview}"
# BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/}"
# API_KEY="${API_KEY:-${LLM_API_KEY:-}}"
MODEL_NAME="${MODEL_NAME:-gpt-5.2}"
API_KEY="${API_KEY:-${LLM_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://us.api.openai.com/v1}"

LANGUAGE="${LANGUAGE:-triton}"
TARGET_GPU="${TARGET_GPU:-H100}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-300}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-output-gpumode}"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-gpt-5.2_gpumode_trimul_triton_r182_20260210_173942_b89cf625}"

WM="${WM:-1}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-5}"
WORLD_MODEL_JSON="${WORLD_MODEL_JSON:-}"

GPUMODE_MODE="${GPUMODE_MODE:-leaderboard}"
GPUMODE_KEEP_TMP="${GPUMODE_KEEP_TMP:-0}"
GPUMODE_TASK_DIR="${GPUMODE_TASK_DIR:-$KSEARCH_ROOT/k_search/tasks/gpu_mode/trimul}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-test}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${LANGUAGE}-gpumode-trimul-wm-opt${MAX_OPT_ROUNDS}}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "ERROR: MODEL_NAME is required" >&2
  exit 2
fi
if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: API key is required (set LLM_API_KEY or API_KEY)" >&2
  exit 2
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"

CONT_ARGS=()
if [[ -n "${CONTINUE_FROM_SOLUTION}" ]]; then
  CONT_ARGS+=(--continue-from-solution "${CONTINUE_FROM_SOLUTION}")
fi

WM_ARGS=()
if [[ "${WM}" == "1" ]]; then
  WM_ARGS+=(--world-model --wm-stagnation-window "${WM_STAGNATION_WINDOW}")
fi

sudo -E env "PATH=$PATH" python3 -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" \
  --task-source gpumode \
  --model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --language "${LANGUAGE}" \
  --target-gpu "${TARGET_GPU}" \
  --max-opt-rounds "${MAX_OPT_ROUNDS}" \
  --save-solutions \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --gpumode-mode "${GPUMODE_MODE}" \
  --gpumode-task-dir "${GPUMODE_TASK_DIR}" \
  --world-model \
  --wm-stagnation-window "${WM_STAGNATION_WINDOW}" \
  --wandb \
  --run-name "${RUN_NAME}" \
  --wandb-project "${WANDB_PROJECT}"


