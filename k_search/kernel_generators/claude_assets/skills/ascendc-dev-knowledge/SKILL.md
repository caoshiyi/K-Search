---
name: ascendc-dev-knowledge
description: "用于查询 AscendC API、算子开发概念、Tiling、Host 侧数据结构、错误码、日志和故障定位资料。"
---

# AscendC 开发知识 Skill

本 skill 是 AscendC 相关资料的唯一入口。查询 API、编程概念、错误码、日志或故障定位资料时，必须先使用本文件的检索规则，再按需读取 `references/` 中的少量文件。

> 在 K-Search 中子代理**禁用 Bash**，检索一律使用 **Grep / Glob / Read 工具**（不是 shell 命令）。
> `references/` 以符号链接方式共享到候选 worktree（只读）。若链接缺失，记录缺口即可，不要尝试下载或重建。

## 核心规则

- 禁止一次性全文读取 `references/`。
- 禁止跳过索引直接扫描全部文档。
- 用 Grep/Glob 工具按关键词定位，不用 shell。

## 资料结构

知识库数据位于本 skill 的 `references/` 目录：

| 目录 | 用途 |
|------|------|
| `references/api_reference_docs/` | Kernel 侧 API、数据搬运、向量计算、矩阵计算、同步控制、基础数据结构 |
| `references/basic_knowledge_docs/` | AscendC 编程模型、硬件概念、算子实践、Tiling 策略、调试方法 |
| `references/basic_data_api_docs/` | Host 侧基础数据结构、`TilingContext`、`InferShape`、`OpDef` 注册 |
| `references/troubleshooting_docs/` | 错误码、AI Core Error、OOM、进程卡住、进程中断、故障定位工具 |
| `references/log_reference_docs/` | 日志级别、日志配置、日志问题定位 |
| `references/910D_knowledge_extra/` | 910D / 351x 专用补充资料、迁移资料、RegBase、MicroAPI, **910B禁止读取** |

## 检索流程

1. 先判断问题类型：API 查询、编程概念、Host 侧接口、错误码、日志、910D 专用问题。
2. 用 Read 工具读取对应目录的索引：
   - `references/api_reference_docs/INDEX.md`、`references/basic-knowledge.md` 等
   - 顶层索引：`references/api-reference.md`、`references/basic-data-api.md`、`references/troubleshooting.md`、`references/log-reference.md`
3. 根据索引定位最相关文件，单次最多 Read 3 个具体文档。
4. 如果 3 个文档仍不能定位问题，先总结已查资料和缺口，再决定下一轮关键词，不要扩大成全文扫描。

## 常用检索（用 Grep 工具，path 指向对应 references 子目录）

| 找什么 | Grep pattern | path |
|--------|--------------|------|
| Kernel 侧 API | `DataCopy\|Sqrt\|Mmad\|SetFlag\|WaitFlag` | `references/api_reference_docs` |
| 编程概念/算子实践 | `双缓冲\|Tiling\|流水\|向量化\|搬运` | `references/basic_knowledge_docs` |
| Host 侧接口 | `GetInputShape\|InferShape\|TilingContext\|OpDef` | `references/basic_data_api_docs` |
| 错误码/故障现象 | `EZ9999\|AI Core Error\|OOM\|卡住\|超时` | `references/troubleshooting_docs` |
| 日志资料 | `日志级别\|plog\|trace\|debug` | `references/log_reference_docs` |
| 910D/351x 专用（910B 禁读） | `910D\|351x\|RegBase\|MicroAPI\|迁移` | `references/910D_knowledge_extra` |

## 使用边界

- 优化已有 AscendC 算子：本 skill 是唯一 API/概念资料入口。
- 若本地 references 缺少目标资料，直接记录缺口；不要在算子生成过程中构建、下载或更新知识库。
