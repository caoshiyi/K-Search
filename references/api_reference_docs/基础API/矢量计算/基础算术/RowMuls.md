# RowMuls

> ⚠️ **重要提示**: RowMuls **不是** AscendC 框架内置 API，而是**用户自定义 `__aicore__ inline` 函数**。
> 它使用 AscendC 框架的 `Mul`（高维切分模式）+ `BinaryRepeatParams` 组合实现行级广播缩放。
> 使用前需要自行定义此函数（或引入包含此函数的头文件）。

#### 功能说明
逐行向量乘法：将每行的元素与 scale 张量中对应行的缩放因子相乘。

计算公式：
```
dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
```

等价于标量循环 `GetValue + Muls` 的向量化版本，但通过 `BinaryRepeatParams.src1RepStride=1, src1BlkStride=0` 实现每行使用同一个 BRCB_BLOCK（8个fp32元素）做缩放，避免标量逐元素处理。

#### 函数原型

```
template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstUb | 输出 | 目标张量 [dealRowCount, columnCount] |
| src0Ub | 输入 | 源张量 [dealRowCount, columnCount] |
| src1Ub | 输入 | 缩放因子张量 [dealRowCount, FP32_BLOCK_ELEMENT_NUM(=8)]，每行一个 BRCB_BLOCK |
| dealRowCount | 输入 | 处理行数（repeat times，必须 < 256） |
| columnCount | 输入 | 对齐列数（含padding） |
| actualColumnCount | 输入 | 实际有效列数 |

#### 底层实现：Mul + BinaryRepeatParams

RowMuls 的核心实现依赖以下 AscendC 框架 API：

1. **`Mul`**（高维切分模式）：`Mul(dst, src0, src1, mask, repeatTimes, repeatParams)`
2. **`BinaryRepeatParams`**：控制 src0/src1/dst 的 block stride 和 repeat stride

关键参数配置：
```cpp
// 列方式处理（dLoop <= dealRowCount 时）
BinaryRepeatParams repeatParams;
repeatParams.src0BlkStride = 1;
repeatParams.src1BlkStride = 0;    // src1 在 block 内连续读取
repeatParams.dstBlkStride = 1;
repeatParams.src0RepStride = columnCount / blockElementNum;  // 每行跳跃
repeatParams.src1RepStride = 1;    // src1 每次repeat前进1个block（=8个fp32元素）
repeatParams.dstRepStride = columnCount / blockElementNum;   // 每行跳跃

Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
```

#### 完整源码参考

源码来自华为 CANN omni-ops 项目 `vector_common.h`：
```
K-Search/references/row_ops_source_reference/vector_common_row_ops.h
```

#### Anti-Pattern（已知误用模式）

1. **Do NOT use Duplicate(scalar,dimAlign)+Mul(dst,src,buf,dimAlign) 作为替代**
   - Reason: Duplicate仅填充dim个有效元素为scalar值，padding区域(dimAlign-dim)保留垃圾值不清零。Mul操作整个dimAlign宽度包括padding区域的垃圾值。且Duplicate和Mul在同一PIPE_V上执行，无PipeBarrier分隔，存在RAW冒险。额外赋值步骤引入精度误差累积。
   - 正确替代：使用 RowMuls 函数（Mul + BinaryRepeatParams.src1RepStride=1 组合）。

2. **Do NOT use LocalTensor[] 下标运算符作为 float 标量参数传入 Muls**
   - Reason: LocalTensor[] 返回子张量视图 LocalTensor<T>，不是标量 T。使用 GetValue(tensor, index) 提取标量值。

3. **Do NOT use Brcb(dst,src,count) 只传3个参数替代行级缩放**
   - Reason: AscendC Brcb 需要4个参数: Brcb(dst,src,repeatTimes,dstStride)。行级广播缩放的正确实现是 RowMuls 函数（Mul + BinaryRepeatParams），不是 Brcb+Mulcs 组合。

4. **Do NOT 直接调用 RowMuls 而不先定义该函数**
   - Reason: RowMuls 不是 AscendC 框架内置 API。调用前必须：要么在项目中定义该函数（参考 vector_common_row_ops.h 源码），要么引入包含该函数的头文件。否则会编译失败（undeclared）。

#### 调用示例

```cpp
// Step 1: 定义或引入 RowMuls 函数（从 vector_common_row_ops.h）
// Step 2: 准备 scale tensor（每行缩放因子在 offset row * BRCB_NUM 处）
LocalTensor<float> expStateUb = expCacheUb_[slot * stateStride_ + startRow * BRCB_NUM];
// Step 3: 调用 RowMuls
RowMuls(oPrevUb_, oPrevUb_, expStateUb, dealRows, dim, actualDim);
```