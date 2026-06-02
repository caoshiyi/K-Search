# SetAntiQuantVector

**页面ID:** atlasascendc_api_07_0656  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0656.html

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

在Matmul计算时支持A矩阵half类型输入，B矩阵int8类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从GM搬出到L1时，会执行伪量化操作，将B矩阵转化为half类型。本节的伪量化接口提供一个量化参数向量，该向量的shape为[1, N]，N值为Matmul矩阵计算时M/N/K中的N值。对B矩阵的每一列都采用该向量中对应列的伪量化系数进行伪量化。

请在Iterate或者IterateAll之前调用该接口。

#### 函数原型

```
__aicore__ inline void SetAntiQuantVector(const LocalTensor<SrcT> &offsetTensor, const LocalTensor<SrcT> &scaleTensor)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offsetTensor | 输入 | 伪量化运算时的参数向量，用于加。SrcT为A_TYPE中对应的数据类型。 |
| scaleTensor | 输入 | 伪量化运算时的参数向量，用于乘。SrcT为A_TYPE中对应的数据类型。 |

#### 约束说明

无
