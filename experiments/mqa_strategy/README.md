# MQA Strategy Form Experiment

Multi-Query Attention (MQA) 算子优化策略形式对比实验的完整归档。

## 目录结构

```
experiments/mqa_strategy/
  ├── README.md          ← 本文件
  ├── docs/              ← 实验报告和分析文档
  ├── scripts/           ← 实验脚本（路径已修复，支持任意部署位置）
  └── strategies/        ← 策略变体 JSON 文件
      ├── mqa_experiments/        ← 25 个单策略变体文件
      └── mqa_strategy_variants_test.json  ← 完整策略目录
```

## 快速开始

### 前置条件

1. K-Search 已安装并可运行 `generate_kernels_and_eval.py`
2. 已测量 baseline 延迟：`export BASELINE_MS=<mean_us/1000>`
3. Claude Agent SDK 已安装：`pip install claude-agent-sdk`
4. `ANTHROPIC_AUTH_TOKEN` 和 `ANTHROPIC_BASE_URL` 已设置

### 运行单个策略形式实验

```bash
# Baseline（纯 LLM WM，无策略注入）
bash experiments/mqa_strategy/scripts/exp_mqa_baseline_llm.sh

# Natural Language 策略形式
bash experiments/mqa_strategy/scripts/exp_mqa_natural_language.sh

# Structured Params 策略形式
bash experiments/mqa_strategy/scripts/exp_mqa_structured_params.sh

# DSL 策略形式
bash experiments/mqa_strategy/scripts/exp_mqa_dsl.sh
```

### 运行全部 4 个对比实验

```bash
bash experiments/mqa_strategy/scripts/run_all_experiments.sh
```

### 策略变体对照实验

```bash
# S1 系列（tiling 策略对照）
python3 experiments/mqa_strategy/scripts/run_s1_series_comparison.py

# S4 系列（compute 策略对照）
python3 experiments/mqa_strategy/scripts/run_s4_series_comparison.py

# S6 系列（memory 策略对照）
python3 experiments/mqa_strategy/scripts/run_s6_series_comparison.py

# S10 系列（多级 tiling 策略对照）
python3 experiments/mqa_strategy/scripts/run_s10_series_comparison.py
```

### 批量运行所有变体

```bash
python3 experiments/mqa_strategy/scripts/run_mqa_strategy_experiments.sh --baseline-ms <ms>
# 或小规模示范
python3 experiments/mqa_strategy/scripts/run_mqa_strategy_experiments.sh --baseline-ms <ms> --sample
```

## 路径设计

所有脚本中的路径已修复为**相对推导**，不再硬编码绝对路径：

| 路径 | Shell 脚本 | Python 脚本 |
|------|-----------|-------------|
| KSEARCH_ROOT | `${KSEARCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}` | `Path(__file__).resolve().parent.parent` |
| TASK_DIR | `${TASK_DIR:-/mnt/workspace/cv_agent/tile2asc/multi_query_attention}` | `os.environ.get("TASK_DIR", "...")` |
| STRATEGIES_DIR | `$KSEARCH_ROOT/experiments/mqa_strategy/strategies/mqa_experiments` | `KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments"` |

如果将 K-Search 部署到不同位置，只需：
- Shell 脚本：设置 `export KSEARCH_ROOT=<new_path>` 和 `export TASK_DIR=<new_task_path>`
- Python 脚本：设置 `export TASK_DIR=<new_task_path>`（KSEARCH_ROOT 通过 `__file__` 自动推导）

## 策略目录说明

`strategies/mqa_strategies_catalog.json`（在 K-Search 根目录下）包含完整的 12 个策略定义（S1-S12），而 `strategies/mqa_experiments/` 下是每个策略变体的独立 JSON 文件，由 `extract_strategy_variants.py` 从 `mqa_strategy_variants_test.json` 生成。

## 文档说明

`docs/` 目录下的实验报告文档中可能包含**当时运行环境的绝对路径**（如 `/mnt/workspace/...`），这些是事实性引用，记录了实际运行时的路径和结果位置，不做修改。

## 实验输出目录

实验运行后产生的输出目录（`.ksearch-exp-*`）在 K-Search 根目录下，已被 `.gitignore` 排除。可通过 `collect_experiment_results.py` 收集汇总。