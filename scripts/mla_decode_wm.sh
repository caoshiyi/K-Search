
KSEARCH_ROOT="${KSEARCH_ROOT:-}"
DATASET_ROOT="${DATASET_ROOT:-}"

MODEL_NAME="${MODEL_NAME:-gemini-3-pro-preview}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/}"

DEFINITION="${DEFINITION:-mla_paged_decode_h16_ckv512_kpe64_ps1}"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-H100}"

BASELINE_SOLUTION="${BASELINE_SOLUTION:-flashinfer_wrapper_03f7b0}"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-gemini-3-pro-preview_mla_paged_decode_h16_ckv512_kpe64_ps1_mma}"

MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-20}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-7}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-output}"

WANDB_PROJECT="${WANDB_PROJECT:-test}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${LANGUAGE}-wm-${DEFINITION}-seed-opt${MAX_OPT_ROUNDS}}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
sudo -E env "PATH=$PATH" python3 -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" \
  --local "${DATASET_ROOT}" \
  --task-source flashinfer \
  --task-path "${DATASET_ROOT}" \
  --definition "${DEFINITION}" \
  --model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --language "${LANGUAGE}" \
  --target-gpu "${TARGET_GPU}" \
  --world-model \
  --wm-stagnation-window "${WM_STAGNATION_WINDOW}" \
  --max-opt-rounds "${MAX_OPT_ROUNDS}" \
  --parallel-workloads \
  --continue-from-solution "${CONTINUE_FROM_SOLUTION}" \
  --save-solutions \
  --use-isolated-runner \
  --baseline-solution "${BASELINE_SOLUTION}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --run-name "${RUN_NAME}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --feedback-workloads \
    bd2dae14-7bae-4edb-964f-2163accf506e \
    84221f45-78f8-4d44-84f6-998153d2c1fa \
    d0da33e2-2d94-42b5-be8a-09111f9f2649 \
    e417264f-195d-4204-89fa-3ebdb539f1cf \
    939f995a-1ab2-4d19-8d94-50f07e73542d



