# GetBatchTensorC

**页面ID:** atlasascendc_api_07_0637  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0637.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

调用一次GetBatchTensorC，会获取C矩阵片，该接口可以与IterateNBatch异步接口配合使用。用于在调用IterateNBatch迭代计算后，获取一片std::max(batchA, batchB) * singleCoreM * singleCoreN大小的矩阵分片。

#### 函数原型

```
template <bool sync = true>
__aicore__ inline GlobalTensor<DstT> GetBatchTensorC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

```
template <bool sync = true>
__aicore__ inline void GetBatchTensorC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| sync | 当前仅支持异步模式，即该参数只支持取值为false。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| batchA | 输入 | 左矩阵的batch数。 |
| batchB | 输入 | 右矩阵的batch数。 |
| enSequentialWrite | 输入 | 该参数预留，开发者无需关注。 |
| c | 输入 | C矩阵放置于Local Memory的地址，用于保存矩阵分片。 |

#### 返回值说明

GlobalTensor<DstT>，返回计算的矩阵分片。

#### 约束说明

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- C矩阵片输出到Local Memory，且单核计算的N方向大小singleCoreN非32字节对齐的场景，C矩阵的CubeFormat仅支持ND_ALIGN格式，输出C矩阵片时，自动将singleCoreN方向上的数据补齐至32字节。

#### 调用示例

```
// 计算需要多Batch计算循环次数
int for_extent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
mm1.SetTensorA(gm_a[0], isTransposeAIn);
mm1.SetTensorB(gm_b[0], isTransposeBIn);
if (tiling.isBias) {
    mm1.SetBias(gm_bias[0]);
}
// 多batch Matmul计算    
mm1.template IterateNBatch<false>(for_extent, batchA, batchB, false);
...other compute
for (int i = 0; i < for_extent; ++i) {   
    mm1.template GetBatchTensorC<false>(ubCmatrix); 
    ...other compute
}
```
