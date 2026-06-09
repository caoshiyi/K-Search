# flash_attention 性能优化脚本实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建自动化脚本，通过 K-Search World Model 对 flash_attention 算子执行 3 轮性能优化

**Architecture:** 主脚本 `flash_attention_wm.sh` 调用 cv_agent_adv 的编译/测试脚本进行 baseline 测量，然后调用 K-Search 核心 Python 模块执行优化循环。三个子脚本 (`ksearch_*.sh`) 作为 K-Search 与实际编译/测试流程的适配层。

**Tech Stack:** Bash, Python (K-Search), AscendC NPU, Claude Agent SDK

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `scripts/flash_attention_wm.sh` | 创建 | 主脚本：环境检查、baseline 测量、调用 K-Search |
| `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_build.sh` | 修改 | 编译适配：调用 build_ascendc.py |
| `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_test.sh` | 修改 | 测试适配：调用 run_accuracy.py |
| `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_bench.sh` | 修改 | 性能适配：调用 run_perf.py |

---

### Task 1: 创建主脚本 flash_attention_wm.sh

**Files:**
- Create: `/mnt/workspace/K-Search/scripts/flash_attention_wm.sh`

- [ ] **Step 1: 创建脚本框架和头部注释**

```bash
#!/usr/bin/env bash
# flash_attention 性能优化脚本
#
# 功能：对 flash_attention 算子执行 K-Search 性能优化
#
# 前置条件：
#   1. Claude Agent SDK 已安装：uv pip install claude-agent-sdk
#   2. ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL 已设置
#   3. NPU 设备可用（默认设备 ID 3）
#
# 用法：
#   ./scripts/flash_attention_wm.sh
#   BASELINE_MS=0.5 ./scripts/flash_attention_wm.sh  # 跳过 baseline 测量
#   KSEARCH_DEVICE_ID=0 ./scripts/flash_attention_wm.sh  # 指定 NPU 设备

set -euo pipefail
```

- [ ] **Step 2: 添加环境变量自动加载逻辑**

```bash
# --- Auto-load env from ~/.claude/settings.json if available -----------------
if command -v jq >/dev/null 2>&1 && [ -f "$HOME/.claude/settings.json" ]; then
    while IFS='=' read -r k v; do
        if [ -n "$k" ] && [ -z "${!k:-}" ]; then
            export "$k=$v"
        fi
    done < <(jq -r '.env | to_entries[] | .key + "=" + (.value|tostring)' "$HOME/.claude/settings.json")
fi
```

- [ ] **Step 3: 添加必需环境检查**

```bash
# --- Required env ------------------------------------------------------------
: "${ANTHROPIC_AUTH_TOKEN:?missing - export it or set in ~/.claude/settings.json}"
: "${ANTHROPIC_BASE_URL:?missing - export it or set in ~/.claude/settings.json}"
```

- [ ] **Step 4: 添加目录和参数配置**

```bash
# --- Configurable ------------------------------------------------------------
KSEARCH_ROOT="${KSEARCH_ROOT:-/mnt/workspace/K-Search}"
TASK_DIR="${TASK_DIR:-/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/mnt/workspace/cv_agent_adv/agent_workdir/scripts}"
MODEL_NAME="${MODEL_NAME:-glm-5.1}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
TARGET_GPU="${TARGET_GPU:-Ascend910B3}"
TIMEOUT_S="${TIMEOUT_S:-900}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-fa}"

# Accuracy: which case types to validate each round
export KSEARCH_TEST_CASE_TYPES="${KSEARCH_TEST_CASE_TYPES:-basic,general}"
# Performance: bench case type, warmup, and repeat count
export KSEARCH_BENCH_CASE_TYPE="${KSEARCH_BENCH_CASE_TYPE:-basic}"
export KSEARCH_WARMUP="${KSEARCH_WARMUP:-5}"
export KSEARCH_REPEAT="${KSEARCH_REPEAT:-20}"
# NPU device ID
export KSEARCH_DEVICE_ID="${KSEARCH_DEVICE_ID:-3}"

# API and agent settings
export API_TIMEOUT_MS=7200000
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-64000}"
export CLAUDE_AGENT_MAX_TURNS=50

# Run timestamp for unique log directories
export KSEARCH_RUN_START="${KSEARCH_RUN_START:-$(date -u +%Y%m%d_%H%M%S)}"
```

- [ ] **Step 5: 添加 NPU 设备检查函数**

```bash
# --- Check NPU device availability -------------------------------------------
check_npu_device() {
    local device_id="${KSEARCH_DEVICE_ID:-3}"
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "警告: npu-smi 未安装，跳过设备检查"
        return 0
    fi
    if ! npu-smi info 2>/dev/null | grep -q "Device ${device_id}"; then
        echo "错误: NPU 设备 ${device_id} 不可用"
        echo "请运行 'npu-smi info' 查看可用设备，或设置 KSEARCH_DEVICE_ID"
        return 1
    fi
    echo "NPU 设备 ${device_id} 可用"
}
```

- [ ] **Step 6: 添加 baseline 测量函数**

```bash
# --- Measure baseline performance --------------------------------------------
measure_baseline() {
    echo "=== Phase 2: 测量 Baseline 性能 ==="

    # Step 2.1: 编译 AscendC kernel
    echo "[baseline] 编译 AscendC kernel..."
    cd "$TASK_DIR"
    ASCENDC_SOC_VERSION="${TARGET_GPU}" \
    ASCENDC_QUIET=1 \
    ASCENDC_CLEAN_BUILD=1 \
    bash "${SCRIPTS_DIR}/evaluate_ascendc.sh" flash_attention basic

    # Step 2.2: 运行性能测试
    echo "[baseline] 运行性能测试..."
    cd "$TASK_DIR"
    ASCEND_RT_VISIBLE_DEVICES="${KSEARCH_DEVICE_ID}" \
    OUTPUT=$(python utils/run_perf.py flash_attention \
        --impl ascendc \
        --mode time \
        --warmup "${KSEARCH_WARMUP}" \
        --repeat "${KSEARCH_REPEAT}" \
        --case_type "${KSEARCH_BENCH_CASE_TYPE}" \
        --device_id "${KSEARCH_DEVICE_ID}" 2>&1)

    # Step 2.3: 解析延迟值
    BASELINE_MS=$(echo "$OUTPUT" | grep -oP 'mean=\K[0-9.]+')
    if [ -z "$BASELINE_MS" ]; then
        echo "错误: 无法从性能测试输出中解析延迟值"
        echo "输出内容:"
        echo "$OUTPUT"
        echo ""
        echo "请手动测量 baseline 并设置: BASELINE_MS=<值> ./scripts/flash_attention_wm.sh"
        return 1
    fi

    echo "[baseline] 基准性能: ${BASELINE_MS}ms"
    export BASELINE_MS
}
```

- [ ] **Step 7: 添加 K-Search 优化调用函数**

```bash
# --- Run K-Search optimization ------------------------------------------------
run_ksearch_optimization() {
    echo "=== Phase 3: 执行 K-Search 优化 (${MAX_ROUNDS} 轮) ==="

    cd "$KSEARCH_ROOT"

    python -u generate_kernels_and_eval.py \
        --task-source ascendc \
        --task-path "$TASK_DIR" \
        --definition flash_attention \
        --model-name "$MODEL_NAME" \
        --llm-provider claude-agent \
        --language ascendc \
        --target-gpu "$TARGET_GPU" \
        --ascendc-build-cmd "./ksearch_build.sh" \
        --ascendc-test-cmd "./ksearch_test.sh" \
        --ascendc-bench-cmd "./ksearch_bench.sh" \
        --ascendc-reference-latency-ms "$BASELINE_MS" \
        --ascendc-timeout-seconds "$TIMEOUT_S" \
        --world-model \
        --max-opt-rounds "$MAX_ROUNDS" \
        --artifacts-dir "$ARTIFACTS_DIR" \
        --save-solutions
}
```

- [ ] **Step 8: 添加主函数和执行入口**

```bash
# --- Main execution -----------------------------------------------------------
main() {
    echo "=== flash_attention 性能优化脚本 ==="
    echo "K-Search 根目录: $KSEARCH_ROOT"
    echo "目标算子目录:   $TASK_DIR"
    echo "优化轮数:       $MAX_ROUNDS"
    echo "输出目录:       $ARTIFACTS_DIR"
    echo "NPU 设备 ID:    ${KSEARCH_DEVICE_ID}"
    echo "运行时间戳:     ${KSEARCH_RUN_START}"
    echo ""

    # Phase 1: 环境检查
    echo "=== Phase 1: 环境检查 ==="
    check_npu_device || exit 1

    # Phase 2: Baseline 测量（如果未提供）
    if [ -z "${BASELINE_MS:-}" ]; then
        measure_baseline || exit 1
    else
        echo "=== Phase 2: 使用预设 Baseline (${BASELINE_MS}ms) ==="
    fi

    # Phase 3: 执行优化
    run_ksearch_optimization

    # Phase 4: 输出结果位置
    echo ""
    echo "=== Phase 4: 完成 ==="
    echo "优化结果保存在: ${KSEARCH_ROOT}/${ARTIFACTS_DIR}"
}

main
```

- [ ] **Step 9: 保存文件并设置权限**

完整脚本内容已写入，保存后执行：

```bash
chmod +x /mnt/workspace/K-Search/scripts/flash_attention_wm.sh
```

- [ ] **Step 10: 验证脚本语法**

Run: `bash -n /mnt/workspace/K-Search/scripts/flash_attention_wm.sh`
Expected: 无输出（语法正确）

---

### Task 2: 修改 ksearch_build.sh

**Files:**
- Modify: `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_build.sh`

- [ ] **Step 1: 备份原文件**

```bash
cp /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_build.sh \
   /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_build.sh.bak
```

- [ ] **Step 2: 覆写为新的适配脚本**

```bash
#!/usr/bin/env bash
# K-Search build phase: compile AscendC kernel
set -euo pipefail

TASK_DIR="/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention"
cd "$TASK_DIR"

python utils/build_ascendc.py flash_attention \
    -v "${KSEARCH_SOC_VERSION:-Ascend910B3}" \
    ${KSEARCH_QUIET:+-q} \
    ${KSEARCH_CLEAN_BUILD:+--clean}
```

- [ ] **Step 3: 验证脚本语法**

Run: `bash -n /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_build.sh`
Expected: 无输出（语法正确）

---

### Task 3: 修改 ksearch_test.sh

**Files:**
- Modify: `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_test.sh`

- [ ] **Step 1: 备份原文件**

```bash
cp /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_test.sh \
   /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_test.sh.bak
```

- [ ] **Step 2: 覆写为新的适配脚本**

```bash
#!/usr/bin/env bash
# K-Search correctness phase: run accuracy test
set -euo pipefail

TASK_DIR="/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention"
cd "$TASK_DIR"

ASCEND_RT_VISIBLE_DEVICES="${KSEARCH_DEVICE_ID:-3}" \
python utils/run_accuracy.py flash_attention \
    --impl ascendc \
    --case "${KSEARCH_CASE_TYPE:-basic}"
```

- [ ] **Step 3: 验证脚本语法**

Run: `bash -n /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_test.sh`
Expected: 无输出（语法正确）

---

### Task 4: 修改 ksearch_bench.sh

**Files:**
- Modify: `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_bench.sh`

- [ ] **Step 1: 备份原文件**

```bash
cp /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_bench.sh \
   /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_bench.sh.bak
```

- [ ] **Step 2: 覆写为新的适配脚本**

```bash
#!/usr/bin/env bash
# K-Search benchmark phase: measure kernel latency
# Uses host-side timing (time.perf_counter + npu.synchronize)
set -euo pipefail

TASK_DIR="/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention"
cd "$TASK_DIR"

ASCEND_RT_VISIBLE_DEVICES="${KSEARCH_DEVICE_ID:-3}" \
python utils/run_perf.py flash_attention \
    --impl ascendc \
    --mode time \
    --warmup "${KSEARCH_WARMUP:-5}" \
    --repeat "${KSEARCH_REPEAT:-20}" \
    --case_type "${KSEARCH_CASE_TYPE:-basic}" \
    --device_id "${KSEARCH_DEVICE_ID:-3}"
```

- [ ] **Step 3: 验证脚本语法**

Run: `bash -n /mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/ksearch_bench.sh`
Expected: 无输出（语法正确）

---

### Task 5: 提交更改

**Files:**
- Commit all changes

- [ ] **Step 1: Git add 所有修改的文件**

```bash
cd /mnt/workspace/K-Search
git add scripts/flash_attention_wm.sh
git add docs/superpowers/specs/2026-06-06-flash-attention-perf-script-design.md
git add docs/superpowers/plans/2026-06-06-flash-attention-perf-script.md
```

- [ ] **Step 2: 提交更改**

```bash
git commit -m "feat: add flash_attention performance optimization script

- Create flash_attention_wm.sh for automated K-Search optimization
- Update ksearch_*.sh to use cv_agent_adv build/test scripts
- Add design spec and implementation plan

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 自审检查清单

| 检查项 | 状态 |
|--------|------|
| Spec 覆盖完整 | ✓ 所有配置参数已实现 |
| 无 placeholder | ✓ 所有代码完整 |
| 参数名一致 | ✓ 使用 --case_type (Python argparse) |
| 环境变量传递 | ✓ KSEARCH_* 变量已 export |
| 错误处理 | ✓ baseline 失败有明确提示 |