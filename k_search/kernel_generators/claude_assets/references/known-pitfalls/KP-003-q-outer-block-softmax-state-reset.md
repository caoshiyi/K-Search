# KP-003：多 Q outer block 任务的 softmax 状态缓存必须在每块开始时重置

- 状态：tentative（已在 K-Search flash_attention round1 v0 验证，待采纳进基线）
- 适用阶段：AscendC / 多级分块优化
- 主要受益 agent：designer / codegen / reviewer / bug-fixer

## 适用场景

实现 QKV 两级分块（或任意多级分块）时，当 `qSeqLen > mBaseSize` 导致任务被分成多个 Q outer block（`qOuterBlocks > 1`），且 softmax 使用 flash attention 模式（online-softmax with state cache）。

典型结构：
- Vec kernel class 持有 softmax 状态缓存：`maxCacheUb_`、`sumCacheUb_`（per-row max/sum）
- `ComputeVec1(slot, isFirst, kvRows)` 使用 slot-indexed state cache
- 多个 Q outer block 串行处理，每个 block 有独立的 KV 外层循环

## 典型现象

- 精度测试显示 **恰好 50% mismatch**（或 `(qOuterBlocks-1)/qOuterBlocks * 100%`）
- 误差分布呈明显 **bimodal**：前 `mBaseSize` 行正确，后续行全部错误
- 误差集中在后半段输出（第二个及后续 Q outer block）
- REL 误差 ~4.5e-01（显著错误），而非微小精度偏差

## 根本原因

Vec class 的 softmax 状态缓存（`maxCacheUb_`、`sumCacheUb_`）在跨 Q outer block 时**未重置**：

1. Q outer block 0 处理完成，softmax 状态缓存保留了该 block 最后一个 KV iteration 的状态
2. Q outer block 1 开始，`vec1Loop` 重置为 0（正确），但 **softmax state cache 未清空**
3. 当 Q outer block 1 执行 KV iteration 1 时，`isFirst=false`，代码从 `prevSlot` 读取状态
4. 读到的是 **Q outer block 0 的最终状态**，而非 Q outer block 1 的 KV iteration 0 状态
5. Flash attention correction 公式应用了错误的 prev_max/prev_sum，导致后半数据全部错误

### 误导性代码结构

```cpp
// Vec kernel 顶层
FlashAttentionVec vecKernel_;
vecKernel_.InitBuffers(&pipe);  // state cache 初始化一次

// Kernel 顶层任务循环
for each task (bz, qHead, qOuterIdx):
    vec1Loop = 0;  // 重置循环计数器
    // ❌ 但 softmax state cache 未重置！
    
    for s2 in 0 .. s2OuterBlocks:
        vec.ComputeVec1(slot, isFirst=(s2==0), kvRows)
```

`isFirst=(s2==0)` 只追踪当前 Q outer block 内的 KV iteration，无法区分"这是第一个 Q outer block 的第一个 iteration"还是"这是后续 Q outer block 的第一个 iteration"。

## 修复 / 预防清单

### 方法 A：添加 `isFirstQBlock` 参数

```cpp
// Kernel 顶层
bool isFirstQBlock = (qOuterIdx == 0);
for s2 in 0 .. s2OuterBlocks:
    vec.ComputeVec1(slot, isFirst=(s2==0), kvRows, isFirstQBlock)

// Vec class
ComputeVec1(slot, isFirst, kvRows, isFirstQBlock):
    if (isFirstQBlock && isFirst) {
        // 使用默认初始状态
        inMaxTensor = softmaxMaxDefaultUb_;  // -INFINITY
        inSumTensor = softmaxSumDefaultUb_;  // 0
    } else if (isFirstQBlock) {
        // Q outer block 0 的后续 KV iteration：从 prevSlot 读
        // ... 正常逻辑
    } else {
        // Q outer block > 0：不应该使用 prevSlot 的旧状态
        // 但如果是 KV iteration 0，也应该用初始状态
        // 这里有语义歧义，需要重构
    }
```

### 方法 B：添加 `ResetSoftmaxState()` 方法（推荐）

```cpp
// Vec class 新增
public:
    __aicore__ inline void ResetSoftmaxState() {
        // 清空当前 softmax 状态缓存
        Duplicate(maxCacheUb_, SOFTMAX_NEG_INF, stateStride_);
        Duplicate(sumCacheUb_, 0.0f, stateStride_);
    }

// Kernel 顶层任务循环
for each task (bz, qHead, qOuterIdx):
    if AIV:
        vecKernel_.ResetSoftmaxState();  // ✅ 每个新 Q outer block 重置
    
    for s2 in 0 .. s2OuterBlocks:
        vec.ComputeVec1(slot, isFirst=(s2==0), kvRows)
```

### 调用时机

状态重置必须在以下时机触发：
1. **每个 Q outer block 任务开始时**（`qOuterIdx` 变化）
2. **在新任务处理第一个 KV 外层块之前**

### 注意事项

- 该问题只在 `qOuterBlocks > 1` 时触发（即 `qSeqLen > mBaseSize`）
- 对于 `qOuterBlocks = 1`，重置是安全的但非必须
- 重置必须覆盖所有 softmax 状态：`maxCache`、`sumCache`（如果有 `expCache` 也需重置）
- 该问题与 KP-001（subblock writeback offset）症状相似（约一半行错），但根因不同

## 快速诊断口诀

若出现以下特征组合，高度怀疑 KP-003：
1. 多级分块实现 + `qSeqLen > mBaseSize`
2. mismatch 比例 = `(qOuterBlocks-1)/qOuterBlocks * 100%`（典型 50%）
3. 误差分布 bimodal（前半正确、后半错误）
4. REL 误差 ~4e-01 或更高（明显数值错误）

验证方法：
- 在 kernel 任务循环开始处添加 `vec.ResetSoftmaxState()` 调用
- 若精度从 FAIL 变 PASS，确诊为 KP-003

## 反例 / 不适用

- 单级固定 128 分块（`qOuterBlocks=1`）不触发
- 非 flash attention 模式（标准 softmax，不缓存 max/sum）不适用
- 若误差为微小精度偏差而非明显数值错误，应查精度链而非本坑

## 关联 Anti-Pattern

- AP-002 (Q Outer Block State Cache Pollution) - 本坑的 formal anti-pattern 记录