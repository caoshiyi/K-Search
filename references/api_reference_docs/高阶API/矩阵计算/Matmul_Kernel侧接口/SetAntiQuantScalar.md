# SetAntiQuantScalar

**页面ID:** atlasascendc_api_07_0655  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0655.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

在Matmul计算时支持A矩阵half类型输入，B矩阵int8类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从GM搬出到L1时，会执行伪量化操作，将B矩阵转化为half类型。本节的伪量化接口提供对B矩阵的所有数据采用同一量化系数进行伪量化的功能。

请在Iterate或者IterateAll之前调用该接口。

#### 函数原型

```
__aicore__ inline void SetAntiQuantScalar(const SrcT offsetScalar, const SrcT scaleScalar)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offsetScalar | 输入 | 伪量化系数，用于加法。SrcT为A_TYPE中对应的数据类型。 |
| scaleScalar | 输入 | 伪量化系数，用于乘法。SrcT为A_TYPE中对应的数据类型。 |

#### 约束说明

无
