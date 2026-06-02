# SetInput

**页面ID:** atlasascendc_api_07_0897  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0897.html

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

设置特征矩阵Input。

#### 函数原型

```
__aicore__ inline void SetInput(const AscendC::GlobalTensor<SrcT> &input)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| input | 输入 | Input在Global Memory上的首地址。类型为GlobalTensor。特征矩阵Input支持的数据类型SrcT为：half、bfloat16_t。 |

#### 约束说明

无

#### 调用示例

```
const Conv3DBackpropFilterTilingData* tilingData;
// ...初始化tilingData
ConvBackpropApi::Conv3DBackpropFilter <inputType, weightSizeType, gradOutputType, gradWeightType > gradWeight_;
gradWeight_.Init(&(tilingData->dwTiling));
// 设置Input
gradWeight_.SetInput(inputGm_[offsetB_]);
...
```
