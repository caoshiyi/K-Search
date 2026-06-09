# flash_attention 性能优化脚本设计

## 概述

开发一个自动化脚本，通过 K-Search 的 World Model 策略对 `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention` 算子执行性能优化。

## 目标

- 自动测量 baseline 性能
- 执行 3 轮性能优化循环
- 验证精度正确性
- 输出优化结果到 `.ksearch-fa/` 目录

## 技术设计

### 1. 脚本定位

**文件名**: `scripts/flash_attention_wm.sh`

**位置**: `/mnt/workspace/K-Search/scripts/flash_attention_wm.sh`

**调用关系**:
```
flash_attention_wm.sh
    ├── /mnt/workspace/cv_agent_adv/agent_workdir/scripts/evaluate_ascendc.sh
    ├── /mnt/workspace/cv_agent_adv/agent_workdir/scripts/evaluate_performance.sh
    └── generate_kernels_and_eval.py (K-Search 核心)
```

### 2. 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TASK_DIR | `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention` | 目标算子目录 |
| SCRIPTS_DIR | `/mnt/workspace/cv_agent_adv/agent_workdir/scripts` | 编译/测试脚本目录 |
| MODEL_NAME | `glm-5.1` | LLM 模型 |
| MAX_ROUNDS | `3` | 优化轮数 |
| TARGET_GPU | `Ascend910B3` | 目标硬件 |
| TIMEOUT_S | `900` | 每轮超时 (15分钟) |
| ARTIFACTS_DIR | `.ksearch-fa` | 输出目录 |
| TEST_CASE_TYPES | `basic,general` | 精度验证 case 类型 |
| BENCH_CASE_TYPE | `basic` | 性能测试 case 类型 |
| WARMUP | `5` | 性能测试预热次数 |
| REPEAT | `20` | 性能测试重复次数 |
| DEVICE_ID | `3` | NPU 设备 ID |

### 3. 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        执行流程                                   │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: 环境检查                                               │
│  ├── 检查 ANTHROPIC_AUTH_TOKEN                                   │
│  ├── 检查 ANTHROPIC_BASE_URL                                    │
│  └── 检查 NPU 设备 (npu-smi info)                                │
│                                                                  │
│  Phase 2: Baseline 测量 (如果未提供 BASELINE_MS)                 │
│  ├── 编译 AscendC kernel (evaluate_ascendc.sh)                  │
│  └── 运行性能测试 (evaluate_performance.sh)                       │
│      └── 解析输出获取 latency_ms 值                               │
│                                                                  │
│  Phase 3: 执行 K-Search 优化                                     │
│  └── generate_kernels_and_eval.py                               │
│      --task-source ascendc                                       │
│      --task-path $TASK_DIR                                       │
│      --definition flash_attention                                │
│      --model-name glm-5.1                                        │
│      --llm-provider claude-agent                                 │
│      --language ascendc                                          │
│      --target-gpu Ascend910B3                                    │
│      --ascendc-build-cmd ./ksearch_build.sh                      │
│      --ascendc-test-cmd ./ksearch_test.sh                        │
│      --ascendc-bench-cmd ./ksearch_bench.sh                      │
│      --ascendc-reference-latency-ms $BASELINE_MS                 │
│      --ascendc-timeout-seconds 900                               │
│      --world-model                                               │
│      --max-opt-rounds 3                                          │
│      --artifacts-dir .ksearch-fa                                 │
│      --save-solutions                                            │
│                                                                  │
│  Phase 4: 输出结果                                               │
│  └── 结果保存到 .ksearch-fa/                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4. 目标脚本修改

修改 `/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention/` 下的三个脚本：

#### 4.1 ksearch_build.sh

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

#### 4.2 ksearch_test.sh

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

#### 4.3 ksearch_bench.sh

```bash
#!/usr/bin/env bash
# K-Search benchmark phase: print latency_ms=<float>
set -euo pipefail

TASK_DIR="/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention"
cd "$TASK_DIR"

ASCEND_RT_VISIBLE_DEVICES="${KSEARCH_DEVICE_ID:-3}" \
python utils/run_perf.py flash_attention \
    --impl ascendc \
    --case-type "${KSEARCH_CASE_TYPE:-basic}" \
    --warmup "${KSEARCH_WARMUP:-5}" \
    --repeat "${KSEARCH_REPEAT:-20}"
```

### 5. 环境变量传递

K-Search 通过环境变量向子脚本传递配置：

| 环境变量 | 来源 | 用途 |
|----------|------|------|
| KSEARCH_SOC_VERSION | 主脚本 | SoC 版本 |
| KSEARCH_DEVICE_ID | 主脚本 | NPU 设备 ID |
| KSEARCH_CASE_TYPE | 主脚本 | 测试 case 类型 |
| KSEARCH_WARMUP | 主脚本 | 预热次数 |
| KSEARCH_REPEAT | 主脚本 | 重复次数 |
| KSEARCH_QUIET | 主脚本 | 静默编译 |
| KSEARCH_CLEAN_BUILD | 主脚本 | 清理编译 |

### 6. 输出目录结构

```
.ksearch-fa/
├── solutions/
│   └── flash_attention/
│       ├── solution_1.json      # 优化方案 1
│       ├── solution_2.json      # 优化方案 2
│       └── solution_3.json      # 优化方案 3
├── reports/
│   └── evaluation_report.json   # 评估报告
└── logs/
    └── run_YYYYMMDD_HHMMSS.log # 运行日志
```

### 7. 错误处理

| 错误场景 | 处理方式 |
|----------|----------|
| API Token 缺失 | 打印错误信息并退出 |
| NPU 设备不可用 | 提示运行 `npu-smi info` 检查 |
| Baseline 测量失败 | 打印详细错误，建议手动设置 BASELINE_MS |
| 编译失败 | 保留错误日志路径 |
| 优化超时 | 记录当前轮次进度 |

### 8. Baseline 解析逻辑

`run_perf.py` 输出格式示例：
```
  [ascendc] 采集到 20 次执行记录
    mean=0.394ms  min=0.380ms  max=0.420ms
```

解析方式：
```bash
BASELINE_MS=$(echo "$OUTPUT" | grep -oP 'mean=\K[0-9.]+')
```

CSV 输出格式（备用方案）：
- 输出文件：`prof_results/perf_flash_attention_ascendc_time.csv`
- 字段：`mean_ms`

### 8. 使用方式

```bash
# 基本用法 (自动测量 baseline)
cd /mnt/workspace/K-Search
./scripts/flash_attention_wm.sh

# 跳过 baseline 测量
BASELINE_MS=0.394 ./scripts/flash_attention_wm.sh

# 指定 NPU 设备
KSEARCH_DEVICE_ID=0 ./scripts/flash_attention_wm.sh
```

## 实现清单

1. 创建 `scripts/flash_attention_wm.sh` 主脚本
2. 修改 `flash_attention/ksearch_build.sh`
3. 修改 `flash_attention/ksearch_test.sh`
4. 修改 `flash_attention/ksearch_bench.sh`
5. 测试脚本执行

## 风险与限制

- 需要确认 `evaluate_performance.sh` 输出格式包含 `latency_ms=` 字段
- 如果 `run_perf.py` 参数与预期不符，需要调整 `ksearch_bench.sh`
- 优化依赖 LLM API 稳定性