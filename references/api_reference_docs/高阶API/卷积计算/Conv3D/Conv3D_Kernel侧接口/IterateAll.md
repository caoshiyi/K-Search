# IterateAll

**页面ID:** atlasascendc_api_07_10077  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10077.html

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

通过设置结果矩阵Output在GM上的首地址，本接口一次性计算singleCo * singleDo * singleM大小的数据块，并写到结果矩阵Output中。

本接口提供单核内卷积计算能力，singleCo为多核切分后单个核内的输出通道大小；singleDo为多核切分后单个核内的Dout大小；singleM为多核切分后单个核内的M大小。singleCo、singleDo和singleM的大小通过SetSingleOutputShape接口设置。

#### 函数原型

```
__aicore__ inline void IterateAll(const AscendC::GlobalTensor<OutputT>& output, bool enPartialSum = false)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| output | 输入 | Output在GM上的地址。类型为GlobalTensor。结果矩阵Output支持的数据类型为：half、bfloat16_t。 |
| enPartialSum | 输入 | 预留参数。 |

#### 约束说明

- IterateAll接口仅支持处理单batch数据，在多batch计算场景中，需要通过batch次循环调用IterateAll接口完成计算。

```
for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
    conv3dApi.SetInput(inputGm[batchIter * inputOneBatchSize]);
    conv3dApi.IterateAll(outputGm[batchIter * outputOneBatchSize]);
    conv3dApi.End();
}
```

- IterateAll接口必须在初始化接口及输入输出配置接口之后进行调用，完成Conv3D计算，调用顺序如下。

```
Init(...);
... // 输入输出配置
IterateAll(...);
End();
```

#### 调用示例

```
TPipe pipe;
conv3dApi.Init(&tiling);
conv3dApi.SetWeight(weightGm);
if (biasFlag) {
    conv3dApi.SetBias(biasGm);
}
conv3dApi.SetInputStartPosition(diIdxStart, mIdxStart);
conv3dApi.SetSingleOutputShape(singleCoreCout, singleCoreDout, singleCoreM);
for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
    conv3dApi.SetInput(inputGm[batchIter * inputOneBatchSize]);
    conv3dApi.IterateAll(outputGm[batchIter * outputOneBatchSize]);
    conv3dApi.End();
}
```
