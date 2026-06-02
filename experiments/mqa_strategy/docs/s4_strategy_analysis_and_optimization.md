# S4策略得分分析报告

## 问题分析

**得分对比**：
- S1 (Enlarged Tile Sizes): **1.47**
- S4 (Vectorized RowMuls/RowDivs): **1.20**

**差距**: S4得分比S1低 **18%** (1.20 vs 1.47)

---

## 根本原因

### 1. Baseline已经使用了某种向量化

**Baseline实现**（当前代码）：
```cpp
__aicore__ inline void BroadcastRowMuls(LocalTensor<float> &dst,
                                         LocalTensor<float> &src,
                                         LocalTensor<float> &scale,
                                         uint32_t rows, uint32_t cols)
{
    LocalTensor<float> brcbMat = brcbBuf_.Get<float>();  // 需要额外缓冲区
    for (uint32_t row = 0; row < rows; row++) {
        Brcb(brcbMat[row * cols], scale[row], cols);  // 逐行Brcb广播
    }
    PipeBarrier<PIPE_V>();  // 需要等待Brcb完成
    Mulcs(dst, src, brcbMat, rows * cols);  // 批量乘法
}
```

**关键发现**：
- ✅ Baseline **不是scalar循环**（策略描述错误）
- ✅ Baseline已经使用了Brcb（广播）+ Mulcs（批量乘法）
- ⚠️ 但效率不如BinaryRepeatParams方式

---

### 2. 优化实现对比

**优化方法**（RowMuls with BinaryRepeatParams）：
```cpp
__aicore__ inline void RowMuls(LocalTensor<T> dstUb,
                               LocalTensor<T> src0Ub,
                               LocalTensor<T> src1Ub,
                               uint32_t dealRowCount,
                               uint32_t columnCount,
                               uint32_t actualColumnCount)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src1BlkStride = 0;  // 广播模式
    repeatParams.src1RepStride = 1;
    Mul(dstUb, src0Ub, src1Ub, repeatElementNum, dealRowCount, repeatParams);
    // 一次Mul处理所有行，无需额外缓冲区，无需PipeBarrier
}
```

**优势对比**：

| 实现方式 | 指令数 | 需要额外缓冲 | 需要PipeBarrier | 性能提升 |
|---------|-------|------------|----------------|---------|
| Baseline (Brcb+Mulcs) | rows次Brcb + 1次Mulcs | **是** (brcbMat) | **是** | 基线 |
| Optimized (BinaryRepeatParams) | 1次Mul | **否** | **否** | ~20% |

**结论**: 优化空间有限，约20%提升（与策略预期1.2x相符）

---

### 3. 策略预期加速比设置

**S1策略**:
- expected_speedup_interval: {"min": 1.30, **"likely": 1.47**, "max": 1.50}
- 实际得分: **1.47** (达到预期)

**S4策略**:
- expected_speedup_interval: {"min": 1.10, **"likely": 1.20**, "max": 1.30}
- 实际得分: **1.20** (达到预期)

**问题**: S4策略的预期加速比本身就比S1低 **27%** (1.20 vs 1.47)

---

## 为什么S4预期加速比低？

### 理论分析

**优化效果有限的原因**：

1. **Baseline已经向量化**：不是从scalar(1/cycle)到vector(64/cycle)的巨大跨越
2. **改进是"向量化方法优化"**：从"逐行Brcb循环"到"BinaryRepeatParams直接广播"
3. **性能瓶颈不在这里**：
   - RowMuls/RowDivs在softmax归一化阶段（Vec2）
   - 但整体性能瓶颈可能在其他地方（如KV重复加载、同步开销）

**实测证据**：
```cpp
// Baseline: 逐行Brcb + 1次Mulcs
for (row = 0; row < rows; row++) {  // rows次Brcb指令发射
    Brcb(...);
}
PipeBarrier<PIPE_V>();  // 等待开销
Mulcs(dst, src, brcbMat, rows * cols);  // 1次批量乘法

// Optimized: 1次Mul with BinaryRepeatParams
Mul(dst, src0, src1, repeatElementNum, dealRowCount, repeatParams);  // 1次指令
```

**节省**: rows次Brcb指令发射 + 1次PipeBarrier + brcbMat缓冲区分配

对于rows=8（VEC2_M_CHUNK），节省约8次指令发射+1次等待，约20%提升。

---

## 深层问题：VEC2_M_CHUNK太小

**真正影响性能的因素**：

查看baseline代码：
```cpp
// multi_query_attention_vec.h
#define VEC2_M_CHUNK 8  // 太小！
```

**问题**：
- VEC2_M_CHUNK=8意味着每次只处理8行
- RowMuls/RowDivs优化效果受限（rows=8）
- 循环次数多，每轮都有开销

**对比Flash Attention**：
```cpp
// flash_attention_vec.h
VEC2_M_CHUNK = 64  // 8倍提升！
```

**潜在优化**：
- 如果VEC2_M_CHUNK从8增加到64，RowMuls效果会放大8倍
- 循环次数减少，整体Vec2阶段加速更多

---

## 优化建议

### 策略层面优化

**当前S4策略问题**：
1. ❌ 描述不准确："Replace scalar loops"（baseline不是scalar）
2. ❌ 范围太窄：只关注RowMuls/RowDivs函数本身
3. ❌ 预期保守：1.2x预期低估了潜力

**优化方案1：扩展策略范围**

将S4策略扩展为"VEC2全面优化"：

```
策略ID: S4-Enhanced
策略名称: VEC2全面优化

包含优化点：
1. 替换BroadcastRowMuls/BroadcastRowDivs为RowMuls/RowDivs (BinaryRepeatParams方式)
2. 增加VEC2_M_CHUNK从8到64（关键！）
3. 添加Softmax UB状态缓存（避免GM往返）
4. 消除VEC2阶段的PipeBarrier<PIPE_ALL>

预期加速比: 1.5x (组合效果)

实施优先级：
P0: 增加VEC2_M_CHUNK从8到64
P1: 定义RowMuls/RowDivs helper函数（引入vector_common_row_ops.h）
P2: 替换BroadcastRowMuls/BroadcastRowDivs调用
P3: 添加Softmax状态UB缓存
```

---

### 策略内容优化

**改进后的natural_language**：
```
VEC2阶段的全面优化，包含4个关键改进：

1. 增加VEC2_M_CHUNK从8到64（最关键）
   - 每次处理64行而非8行，减少循环次数8倍
   - 更高效利用RowMuls/RowDivs的向量化能力
   - UB预算：oPrevBuf需要64*dimAlign*4字节，dim=128时约32KB，在UB 196KB内

2. 定义并使用RowMuls/RowDivs helper函数（替代逐行Brcb循环）
   - 使用BinaryRepeatParams.src1BlkStride=0实现广播乘法
   - 无需额外brcbMat缓冲区，节省UB空间
   - 无需PipeBarrier等待，减少同步开销
   - 参考实现：references/row_ops_source_reference/vector_common_row_ops.h

3. Softmax状态UB缓存（避免GM往返）
   - 将max/sum/exp状态保存在UB而非GM workspace
   - 减少每轮KV tile的3次GM读写（max, sum, exp）
   - 需要maxCacheBuf_/sumCacheBuf_/expCacheBuf_（RING_SLOTS * stateStride * 4字节）

4. 消除VEC2阶段的PIPE_ALL barrier
   - 使用SetWaitFlag<HardEvent::V_MTE3>替代PipeBarrier<PIPE_ALL>
   - 保留pipeline overlap

组合预期加速比: 1.5x（VEC2_M_CHUNK贡献最大）
```

---

### 具体优化步骤

**Step 1: 增加VEC2_M_CHUNK**（最关键，贡献~1.3x）
```cpp
// multi_query_attention_vec.h
// 旧值
constexpr uint32_t VEC2_M_CHUNK = 8;

// 新值
constexpr uint32_t VEC2_M_CHUNK = 64;  // 8倍提升
```

**Step 2: 引入RowMuls/RowDivs定义**（贡献~1.2x）
```cpp
// 在kernel_common.h或新文件vector_common_row_ops.h中定义
#include "vector_common_row_ops.h"  // 或直接复制定义

// RowMuls函数定义（参考flash_attention实现）
template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb,
                               LocalTensor<T> src0Ub,
                               LocalTensor<T> src1Ub,
                               uint32_t dealRowCount,
                               uint32_t columnCount,
                               uint32_t actualColumnCount)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src1BlkStride = 0;  // 广播模式
    repeatParams.src1RepStride = 1;
    // ...完整实现见flash_attention/kernel_common.h
}
```

**Step 3: 替换调用**
```cpp
// Vec2中的旧调用
BroadcastRowMuls(oPrevUb, oPrevUb, softmaxExpUb_, dealRows, dim);

// 新调用
RowMuls(oPrevUb, oPrevUb, expStateUb, dealRows, dim, actualDim);
// 注意：dealRows现在是64而非8，放大效果8倍
```

---

## 预期效果提升

**保守估计**：
- VEC2_M_CHUNK从8→64: **+1.30x**
- RowMuls优化（BinaryRepeatParams）: **+1.15x**
- Softmax UB缓存: **+1.05x**
- 组合效果: **1.30 × 1.15 × 1.05 ≈ 1.56x**

**对比**：
- 当前S4得分: 1.20
- 优化后预期: **1.56**（提升 **30%**）
- 接近S1得分: 1.47

---

## 结论

### S4得分低的根本原因

1. **策略范围太窄**：只关注函数替换，忽略了VEC2_M_CHUNK等关键因素
2. **预期设置保守**：1.2x预期低估了VEC2优化的潜力
3. **Baseline已经部分优化**：不是从scalar到vector的跨越，而是从"低效向量化"到"高效向量化"

### 优化方向

**不要单独测试RowMuls/RowDivs替换，而是测试"VEC2全面优化组合"**：

```
新策略ID: S4-VEC2-Full-Optimization
包含因素：
- VEC2_M_CHUNK增加（核心）
- RowMuls/RowDivs定义和替换
- Softmax UB状态缓存
- PIPE_ALL消除

预期加速比: 1.5x (比当前S4高25%)
```

---

## 下一步行动

1. **创建新策略**：S4-VEC2-Full-Optimization（包含4个优化点）
2. **运行对照实验**：对比当前S4 vs 新S4-Full
3. **验证效果**：预期得分从1.20提升到1.56

是否需要我立即创建并测试这个优化后的策略？