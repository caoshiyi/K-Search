# KernelBench launcher for k-search (World Model generator).
#
# Environment variables (common):
# - KSEARCH_ROOT: path to k-search repo (default: repo root)
# - MODEL_NAME: LLM model name (default: gpt-5.2)
# - LLM_API_KEY or API_KEY: OpenAI-compatible API key (required)
# - BASE_URL: OpenAI-compatible base url (optional)
#
# Environment variables (task/generation):
# - KSEARCH_LANGUAGE: triton|cuda (default: triton)
# - TARGET_GPU: e.g. A100-80GB, H100 (default: H100)
# - MAX_OPT_ROUNDS: (default: 50)
# - ARTIFACTS_DIR: base output dir (default: .ksearch-output-kernelbench)
# - CONTINUE_FROM_SOLUTION: optional solution name or path to a persisted solution .json
#   (if set, resumes optimization from that solution)
#
# World model options:
# - WM: 1 to enable world-model prompting (default: 1)
# - WM_STAGNATION_WINDOW: end an action cycle after this many consecutive non-improving rounds (default: 5)
#
# Environment variables (kernelbench):
# - LEVEL: KernelBench level [1, 2, or 3] (default: 1)
# - PROBLEM_ID: Problem ID within the level (default: 1)
# - EVAL_MODE: local|modal (default: local)
# - NUM_CORRECT_TRIALS: Number of correctness trials (default: 5)
# - NUM_PERF_TRIALS: Number of performance trials (default: 100)
#
# Optional W&B:
# - WANDB: 1 to enable (default: 0)
# - WANDB_PROJECT, RUN_NAME

KSEARCH_ROOT="${KSEARCH_ROOT:-.}"

# Model configuration
MODEL_NAME="${MODEL_NAME:-gpt-5.2}"
API_KEY="${API_KEY:-${LLM_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"

# Generation configuration
KSEARCH_LANGUAGE="${KSEARCH_LANGUAGE:-triton}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-50}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-output-kernelbench}"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"

# World model configuration
WM="${WM:-1}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-5}"
WORLD_MODEL_JSON="${WORLD_MODEL_JSON:-}"

# KernelBench configuration
LEVEL="${LEVEL:-1}"
PROBLEM_ID="${PROBLEM_ID:-1}"
EVAL_MODE="${EVAL_MODE:-local}"
TARGET_GPU="${TARGET_GPU:-H100}"
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-100}"

# W&B configuration
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-kernelbench}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-kernelbench-l${LEVEL}-p${PROBLEM_ID}-wm-opt${MAX_OPT_ROUNDS}}"

# Validation
if [[ -z "${MODEL_NAME}" ]]; then
  echo "ERROR: MODEL_NAME is required" >&2
  exit 2
fi
if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: API key is required (set LLM_API_KEY or API_KEY)" >&2
  exit 2
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Build continuation arguments
CONT_ARGS=()
if [[ -n "${CONTINUE_FROM_SOLUTION}" ]]; then
  CONT_ARGS+=(--continue-from-solution "${CONTINUE_FROM_SOLUTION}")
fi

# Build world model arguments
WM_ARGS=()
if [[ "${WM}" == "1" ]]; then
  WM_ARGS+=(--world-model --wm-stagnation-window "${WM_STAGNATION_WINDOW}")
fi

# Build W&B arguments
WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--wandb --run-name "${RUN_NAME}" --wandb-project "${WANDB_PROJECT}")
fi

echo "=========================================="
echo "KernelBench K-Search Launcher"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "KernelBench Level: ${LEVEL}, Problem ID: ${PROBLEM_ID}"
echo "Eval Mode: ${EVAL_MODE}"
echo "GPU: ${TARGET_GPU}"
echo "Language: ${KSEARCH_LANGUAGE}"
echo "Max Opt Rounds: ${MAX_OPT_ROUNDS}"
echo "World Model: ${WM}"
echo "Artifacts Dir: ${ARTIFACTS_DIR}"
echo "=========================================="

# Run the generation and evaluation
env "PYTHONPATH=$KSEARCH_ROOT" python -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" \
  --task-source kernelbench \
  --model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --language "${KSEARCH_LANGUAGE}" \
  --target-gpu "${TARGET_GPU}" \
  --max-opt-rounds "${MAX_OPT_ROUNDS}" \
  --save-solutions \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --kernelbench-level "${LEVEL}" \
  --kernelbench-problem-id "${PROBLEM_ID}" \
  --kernelbench-eval-mode "${EVAL_MODE}" \
  --kernelbench-num-correct-trials "${NUM_CORRECT_TRIALS}" \
  --kernelbench-num-perf-trials "${NUM_PERF_TRIALS}" \
  ${CONT_ARGS[@]+"${CONT_ARGS[@]}"} \
  ${WM_ARGS[@]+"${WM_ARGS[@]}"} \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}
