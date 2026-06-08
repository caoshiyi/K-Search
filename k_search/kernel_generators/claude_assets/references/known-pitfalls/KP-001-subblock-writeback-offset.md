# KP-001：分段 / 子块写回偏移必须用单一全局行坐标系

- 状态：adopted（已在两个独立任务复现：cv_agent_adv AC-012 / K-Search flash_attention round1）
- 适用阶段：AscendC
- 主要受益 agent：designer / codegen / bug-fixer

## 适用场景

任何把一个 task 的行（Q 行 / head 行 / M 行）在多个 AIV subblock 之间二次切分，
或在一个 subblock 内再按 chunk 分块循环处理，且最终有一个写回函数
（如 `FinalizeOutputChunk` / writeback）按行偏移寻址 `outGm` / `workspace` 的实现。

典型结构：写回相关的坐标同时存在多个量——
`rowStart_`（subblock 基址）、`chunk` / `startRow`（subblock 内偏移）、
`curQRowStart_` / `bx*BLOCK_M`（task 基址）。

## 典型现象

- 某一半行（常见是第二个 AIV subblock，`subBlockIdx_==1`）结果整段错误或为 0。
- 上游 BMM1 / softmax / BMM2 / merge 的中间结果都正确，唯独最终 `outGm` 错。
- 只在多 subblock / 多 chunk / padded 场景出现；单块或小 shape 可能恰好不触发。
- 精度表现为「约一半行 mismatch」（如 mismatch ~50%），而非全错或全对。

## 根本原因

写回函数的行入参语义被搞混：调用方传了「subblock 内局部偏移」，
而函数体期望「task 内全局行偏移」（或反之），导致少加 / 多加一次 `rowStart_`。

放大此坑的两个诱因：
1. **误导性参数名**：写回函数形参名叫 `startRow`，但历史调用方实际传的是含 subblock
   基址的 `rowOffset`——名字与语义不符。后续实现者为「让名字自洽」把调用改成传局部
   `startRow`，名字对齐了，语义改坏了。
2. **文档命名不一致**：详设里 state cache 寻址用全局行 `mGlobal`，输出寻址却换写成
   `startRow`，同一份文档出现两套坐标名，实现者照字面实现即错。

## 修复 / 预防清单

1. **单一坐标系**：全文 / 全实现只定义一个 task 内全局行变量
   （如 `globalRow = rowStart_ + chunk`），**所有** GM / workspace / state 偏移公式
   一律用它表达，禁止在不同位置切换 `startRow` / `rowOffset` / `mGlobal` 等近义名。
2. **形参名反映语义而非来源**：写回函数入参命名为 `globalRow`，不要叫 `startRow`。
   一旦叫 `globalRow`，调用端传局部偏移就会显得明显可疑，自带自检。
3. **`globalRowStart` 只算一次**：从一种坐标系计算一次，不要叠加两个等价偏移。
4. **越界裁剪前先验证**：在 `qSeqLen` clamp 之前 dump / 断言 `globalRowStart` 与逻辑行 end。
5. **定位口诀**：若「写回前中间结果正确、写回后约一半行错」，优先查这里。

## 反例 / 不适用

- 没有 subblock / chunk 二次切分，写回始终按完整连续 block 处理时不适用。
- 若结果非「半段错」而是整体小数值偏差，应先查上游数学 / 布局 / 精度链，不是本坑。

## 证据来源

- cv_agent_adv AC-012（2026-05-25）：`FinalizeOutputChunk()` 把 `rowStart_` 计了两次
  （多加），heads 8-15 全部未写回；修复后恢复。
- K-Search flash_attention round1（盲实现复核）：`FinalizeOutputChunk(oNewUb, startRow, ...)`
  少加 `rowStart_`，第二个 subblock 的 Q 后 256 行写错地址，mismatch 57%；
  改为传 `rowOffset` 后 PASS（REL 9.77e-04），性能 2402us→1555us（1.54x）。
  同一根因、相反方向，两次独立复现 → adopted。
