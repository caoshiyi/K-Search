# SetSingleShape

**页面ID:** atlasascendc_api_07_0899  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0899.html

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

设置Conv3DBackpropFilter在单核上计算的形状。用于运行时修改shape。

#### 函数原型

```
__aicore__ inline void SetSingleShape(uint64_t singleCoreM, uint64_t singleCoreN, uint64_t singleCoreK)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleCoreM | 输入 | 单核上M的大小，单位为元素个数。 |
| singleCoreN | 输入 | 单核上N的大小，单位为元素个数。 |
| singleCoreK | 输入 | 单核上K的大小，单位为元素个数。 |

#### 约束说明

无

#### 调用示例

```
const Conv3DBackpropFilterTilingData* tilingData;
// ...初始化tilingData
ConvBackpropApi::Conv3DBackpropFilter <inputType, weightSizeType, gradOutputType, gradWeightType > gradWeight_;
....
gradWeight_.Init(&(tilingData->dwTiling));
gradWeight_.SetInput(inputGm_[offsetB_]);
gradWeight_.SetGradOutput(gradOutputGm_[offsetA_]);
gradWeight_.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);// 设置单核计算量
....
```
