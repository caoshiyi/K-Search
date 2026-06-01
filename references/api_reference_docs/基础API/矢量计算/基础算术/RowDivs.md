# RowDivs

> ⚠️ **重要提示**: RowDivs **不是** AscendC 框架内置 API，而是**用户自定义 `__aicore__ inline` 函数**。
> 它使用 AscendC 框架的 `Div`（高维切分模式）+ `BinaryRepeatParams` 组合实现行级广播除法。
> 使用前需要自行定义此函数（或引入包含此函数的头文件）。

#### 功能说明
逐行向量除法：将每行的元素与 scale 张量中对应行的缩放因子相除。与 RowMuls 对称，用于 softmax 最终归一化等场景。

计算公式：
```
dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
```

#### 函数原型

```
template <typename T>
__aicore__ inline void RowDivs(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstUb | 输出 | 目标张量 [dealRowCount, columnCount] |
| src0Ub | 输入 | 源张量 [dealRowCount, columnCount] |
| src1Ub | 输入 | 除法因子张量 [dealRowCount, FP32_BLOCK_ELEMENT_NUM(=8)]，每行一个 BRCB_BLOCK |
| dealRowCount | 输入 | 处理行数 |
| columnCount | 输入 | 对齐列数（含padding） |
| actualColumnCount | 输入 | 实际有效列数 |

#### 底层实现：Div + BinaryRepeatParams

关键参数配置：
```cpp
BinaryRepeatParams repeatParamsDiv;
repeatParamsDiv.src0BlkStride = 1;
repeatParamsDiv.src1BlkStride = 0;    // src1 在 block 内连续读取
repeatParamsDiv.dstBlkStride = 1;
repeatParamsDiv.src0RepStride = columnCount / blockNum;
repeatParamsDiv.src1RepStride = 1;    // src1 每次repeat前进1个block
repeatParamsDiv.dstRepStride = columnCount / blockNum;

Div(dstUb[offset], src0Ub[offset], src1Ub, repeatNum, dealRowCount, repeatParamsDiv);
```

#### 完整源码参考

源码来自华为 CANN omni-ops 项目 `vector_common.h`：
```
K-Search/references/row_ops_source_reference/vector_common_row_ops.h
```

#### Anti-Pattern

1. **Do NOT 直接调用 RowDivs 而不先定义该函数**
   - Reason: RowDivs 不是 AscendC 框架内置 API。调用前必须定义或引入包含该函数的头文件。

#### 调用示例

```cpp
// softmax final normalization using RowDivs
LocalTensor<float> sumStateUb = sumCacheUb_[slot * stateStride_ + startRow * BRCB_NUM];
RowDivs(oNewUb, oNewUb, sumStateUb, dealRows, dim, actualDim);
```