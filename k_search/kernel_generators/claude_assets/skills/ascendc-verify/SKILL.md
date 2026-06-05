---
name: ascendc-verify
description: |
  Use when: 需要对 AscendC 算子的验证结果进行分类（编译报错、内存越界、同步卡死、精度错误）并据此确定定位方向。
---

# AscendC 验证结果分类速查

> 在 K-Search 中，编译与精度/性能验证由框架（Python）在子代理流水结束后统一执行。
> 子代理**不自行运行验证命令、不使用 Bash**。本 skill 用于：拿到框架回传的验证日志后，
> 对失败结果分类，确定下一步的代码定位方向。

## 结果分类与处理

| 结果 | 判定依据 | 定位方向 | 后续操作 |
|:---|:---|:---|:---|
| **编译报错** | 编译阶段报错退出，不进入精度测试 | 根据报错关键词（undeclared identifier / no matching function / template deduction failed）定位 API 或类型问题 | 查 `ascendc-dev-knowledge` 确认 API 签名和用法 |
| **内存越界** | AI Core Error / Segfault / OOM / core dump / 异常退出（非 0 非 124） | DataCopy 长度或 offset 超出 buffer 容量、workspace 申请量不足、输出 shape 不对 | 主要检查偏移计算，查 `ascendc-hardware` 确认 UB/L1/L0 容量限制，检查 tiling 分块参数 |
| **同步卡死** | exit code 124（超时）或日志显示超时 | CrossCoreSetFlag/WaitFlag 不配对、EnQue 后缺 DeQue、DeQue 后缺 FreeTensor、条件分支提前 return 跳过信号、循环死锁 | 查 `ascendc-sync-guide` 检查同步配对和队列配对 |
| **精度错误** | 正常完成但输出 FAIL + 误差值（max_diff / mean_diff） | scale/offset 计算错误、尾块 mask 处理、dtype 分发、非连续输入、未初始化 | **必须用 `ascendc-dumptensor` 的方法论分段定位**：先建立 CPU Golden 心智模型，从输入开始沿数据流逐段比对，禁止不插桩只读代码猜 |
| **PASS** | 全部用例通过 | — | 在 REVIEW_NOTES.md / CODE_MAP.md 中记录通过状态 |

## 错误关键词速查

- `undeclared identifier` / `no matching function` → API 名或签名错误，查 dev-knowledge
- `template deduction failed` → 模板参数（dtype / 维度常量）不匹配
- `AI Core Error` / `core dump` → 越界或非法访问，查偏移与 buffer 容量
- exit code 124 / timeout → 同步死锁或死循环，查 sync-guide
- `FAIL` + `max_diff` → 精度问题，走 dumptensor 方法论
