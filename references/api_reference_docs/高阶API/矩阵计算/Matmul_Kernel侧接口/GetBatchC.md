# GetBatchC

**页面ID:** atlasascendc_api_07_0668  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0668.html

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

GetBatchTensorC接口与该接口的功能相同，建议使用GetBatchTensorC。

调用一次GetBatchC，会获取C矩阵分片，该接口可以与IterateNBatch异步接口配合使用。用于在调用IterateNBatch迭代计算后，获取一片std::max(batchA, batchB) * singleCoreM * singleCoreN大小的矩阵分片。

#### 函数原型

```
template <bool sync = true>
__aicore__ inline GlobalTensor<DstT> GetBatchC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

```
template <bool sync = true>
__aicore__ inline void GetBatchC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| sync | 通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false，默认为同步模式。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| batchA | 输入 | 左矩阵的batch数 |
| batchB | 输入 | 右矩阵的batch数 |
| enSequentialWrite | 输入 | 输出是否连续存放数据，默认false（非连续写模式） |
| c | 输入 | C矩阵，用于保存矩阵分片。类型为LocalTensor。 |

#### 返回值说明

GlobalTensor<DstT>，返回计算的矩阵分片。

#### 约束说明

当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

#### 调用示例

```
// 计算需要多Batch计算循环次数
int g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
int for_exent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
mm1.SetTensorA(gm_a[0], isTransposeAIn);
mm1.SetTensorB(gm_b[0], isTransposeBIn);
if (tiling.isBias) {
    mm1.SetBias(gm_bias[0]);
}
// 多batch Matmul计算    
mm1.template IterateNBatch<false>(for_exent, batchA, batchB, false);
// ...other compute
for (int i = 0; i < for_exent ; ++i) {
    mm1.template GetBatchC<false>(ubCmatrix, batchA, batchB); 
    // ...other compute
}
```
