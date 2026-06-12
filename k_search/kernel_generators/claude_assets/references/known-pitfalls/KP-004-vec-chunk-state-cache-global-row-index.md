# KP-004：Vec chunk 循环中 softmax state cache 索引必须使用全局行偏移

- 状态：adopted（已在 K-Search flash_attention round1 验证并修复）
- 适用阶段：AscendC / 两级分块 + chunk 循环
- 主要受益 agent：codegen / reviewer / bug-fixer

## 适用场景

实现 QKV 两级分块时，Vec 侧 ComputeVec1 使用行 chunk 循环（`vec1ChunkRows=16` 行 per chunk），且 softmax 使用 flash attention 模式（online-softmax with state cache）。

典型结构：
- `ComputeVec1(slot, isFirst, kvRows)` 内部有 chunk 循环：`for chunkRow in 0..qRows step vec1ChunkRows`
- softmax state cache（`maxCacheUb_`、`sumCacheUb_`）按 slot + row 索引
- `isFirst=false` 时需要从 `prevSlot` 读取上一 KV iteration 的状态

## 典型现象

- 精度测试显示 **约 41% mismatch**
- 误差分布：几乎所有 chunk（除第一个 chunk）都有错误
- REL 误差 ~2.46e-01（显著错误），而非微小精度偏差
- bug-fixer 修复后精度 PASS（mismatch 0.00%）

## 根本原因

ComputeVec1 的 softmax state cache 索引**缺少全局行偏移**：

```cpp
// ❌ 错误代码（Round 1）
uint32_t prevStateBase = prevSlot * stateStride_;
// softmax 状态读取
SoftmaxFlashV2<...>(..., maxCacheUb_[prevStateBase], sumCacheUb_[prevStateBase], ...);
```

问题分析：
1. `prevStateBase = prevSlot * stateStride_` 只给出 slot 的基址偏移
2. chunk 循环中，每个 chunk 处理不同的行（`chunkRow` 从 0 到 qRows）
3. 所有 chunk 都读取 `prevStateBase`（即 slot 的第 0-15 行状态）
4. 第二个及后续 chunk 应读取 `prevStateBase + chunkRow` 对应行的状态
5. 读取错误状态导致 online-softmax rescaling 计算错误，产生精度失败

### 正确实现

```cpp
// ✅ 正确代码（Round 2 bug-fixer）
uint32_t prevStateBase = prevSlot * stateStride_;
uint32_t mGlobal = vecStartM_ + chunkRow;  // task 内全局行索引
// softmax 状态读取（加上 mGlobal 偏移）
SoftmaxFlashV2<...>(..., maxCacheUb_[prevStateBase + mGlobal], 
                    sumCacheUb_[prevStateBase + mGlobal], ...);
```

## 修复 / 预防清单

### 1. 状态缓存索引公式

状态缓存索引必须使用 **slot 基址 + 全局行偏移**：

```
stateIndex = slot * stateStride_ + globalRow
```

其中：
- `slot`：当前 KV iteration 的 slot 编号
- `stateStride_`：每个 slot 的状态缓存行数（= AlignUp(mBaseSize, 8)）
- `globalRow`：task 内全局行索引（= vecStartM_ + chunkRow）

### 2. 一致性检查

确保 ComputeVec1 和 ComputeVec2 使用相同的索引公式：
- ComputeVec1 读取 prevState：`prevStateBase + mGlobal`
- ComputeVec1 写入当前 state：`stateBase + mGlobal`
- ComputeVec2 读取 state：`slot * stateStride_ + globalRow`

### 3. Chunk 循环注意事项

- Chunk 循环变量 `chunkRow` 是局部偏移，必须加上 `vecStartM_` 才是全局行
- `vecStartM_` 是 task 内 Q outer block 的起始行（= qOuterIdx * mBaseSize + subBlockIdx * vecDealM_）
- `mGlobal = vecStartM_ + chunkRow` 正确表达 task 内全局行

### 4. 对齐验证

- `stateStride_` 是 32B 对齐的（AlignUp(mBaseSize*4, 32)/4），是 8 的倍数
- `vecStartM_` 是 vecDealM_ 的倍数，vecDealM_ 是 C0=16 的倍数
- `chunkRow` 是 vec1ChunkRows 的倍数，vec1ChunkRows >= 8
- 因此 `prevStateBase + mGlobal` 总是 8 的倍数，满足 float LocalTensor 的 32B 对齐要求

## 快速诊断口诀

若出现以下特征组合，高度怀疑 KP-004：
1. 两级分块 + Vec chunk 循环实现
2. mismatch 比例约 40%（或 `(qRows-vec1ChunkRows)/qRows * 100%`）
3. REL 误差 ~2e-01 或更高（明显数值错误）
4. bug-fixer 在 state cache 索引处添加 mGlobal 后精度 PASS

验证方法：
- 检查 ComputeVec1 的 `isFirst=false` 分支
- 查找 softmax state cache 读取位置
- 确认索引是否包含全局行偏移

## 与 KP-003 的区别

| 特征 | KP-003 | KP-004 |
|------|---------|---------|
| mismatch 比例 | ~50%（后半段错） | ~41%（除首 chunk 全错） |
| 根因 | 跨 Q outer block 未重置状态 | chunk 内索引缺少全局行偏移 |
| 修复方法 | 添加 ResetSoftmaxState() | 添加 mGlobal 到索引公式 |
| 触发条件 | qOuterBlocks > 1 | vec1ChunkRows < qRows |

## 反例 / 不适用

- Vec 不使用 chunk 循环（一次性处理所有行）时不触发
- 非 flash attention 模式（不缓存 softmax state）不适用
- 若 state cache 按完整 slot 索引而非 slot+row，不适用

## 关联文档

- KP-001：subblock writeback offset（同样涉及全局行坐标系）
- KP-003：Q outer block softmax state reset
- K-Search flash_attention round1 transcript.md（Round 1 精度失败，Round 2 bug-fixer 修复）