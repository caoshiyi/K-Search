# MQA Strategy Experiment Archive

Multi-Query Attention (MQA) 算子优化策略实验归档。

当前 K-Search 只支持 `natural_language` 策略形式，并要求 JSON catalog
使用 v2 schema：JSON 保存摘要和 `markdown_ref`，完整策略正文保存在
markdown 文件中。旧的 `structured_params` 和 `dsl` form 对比脚本只作为历史
实验材料保留，不能直接用于当前 CLI。

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

### 运行当前支持的策略实验

```bash
# Baseline（纯 LLM WM，无策略注入）
bash experiments/mqa_strategy/scripts/exp_mqa_baseline_llm.sh

# Natural Language markdown 策略注入
bash experiments/mqa_strategy/scripts/exp_mqa_natural_language.sh
```

### 历史 form 对比实验

`exp_mqa_structured_params.sh`、`exp_mqa_dsl.sh` 和旧的 `run_all_experiments.sh`
记录了早期 strategy-form 对比实验。当前 CLI 会拒绝
`--strategy-form structured_params` 和 `--strategy-form dsl`。

### 历史策略变体对照实验

以下脚本依赖旧的 inline JSON 变体文件，属于历史归档。当前 CLI 使用 v2
catalog 后，需要先把对应单策略 JSON 迁移为 `summary + markdown_ref` 才能重跑。

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

### 历史批量运行所有变体

```bash
python3 experiments/mqa_strategy/scripts/run_mqa_strategy_experiments.sh --baseline-ms <ms>
# 或小规模示范
python3 experiments/mqa_strategy/scripts/run_mqa_strategy_experiments.sh --baseline-ms <ms> --sample
```

## 路径设计

当前支持的 shell 脚本使用相对路径推导，不再硬编码 K-Search 根目录：

| 路径 | Shell 脚本 |
|------|-----------|
| KSEARCH_ROOT | `${KSEARCH_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}` |
| TASK_DIR | `${TASK_DIR:-/mnt/workspace/cv_agent/tile2asc/multi_query_attention}` |
| STRATEGY_CATALOG | `$KSEARCH_ROOT/strategies/mqa_strategies_catalog.json` |

如果将 K-Search 部署到不同位置，只需：
- Shell 脚本：设置 `export KSEARCH_ROOT=<new_path>` 和 `export TASK_DIR=<new_task_path>`

## 策略目录说明

`strategies/mqa_strategies_catalog.json`（在 K-Search 根目录下）是当前可用的
v2 catalog，包含 12 个策略的摘要、元数据和 markdown 引用。
完整策略正文位于 `strategies/mqa_strategies/*.md`。

`experiments/mqa_strategy/strategies/` 下的旧 JSON 变体属于历史实验归档，
仍包含 inline `natural_language`、`structured_params` 或 `dsl` 字段；如需重跑，
需要先迁移到 v2 catalog + markdown_ref schema。

## 文档说明

`docs/` 目录下的实验报告文档中可能包含**当时运行环境的绝对路径**（如 `/mnt/workspace/...`），这些是事实性引用，记录了实际运行时的路径和结果位置，不做修改。

## 实验输出目录

实验运行后产生的输出目录（`.ksearch-exp-*`）在 K-Search 根目录下，已被 `.gitignore` 排除。可通过 `collect_experiment_results.py` 收集汇总。
