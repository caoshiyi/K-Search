# SetQuantVector

**页面ID:** atlasascendc_api_07_0650  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0650.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

本接口提供对输出矩阵采用向量进行量化或反量化的功能，即对于输入shape为[1, N]的参数向量，N值为Matmul矩阵计算时M/N/K中的N值，对输出矩阵的每一列都采用该向量中对应列的系数进行量化或反量化。量化、反量化的详细内容请参考量化场景。

Matmul反量化场景：在Matmul计算时，左、右矩阵的输入为int8_t或int4b_t类型，输出为half类型；或者左、右矩阵的输入为int8_t类型，输出为int8_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行反量化操作，将最终结果反量化为对应的half或int8_t类型。

Matmul量化场景：在Matmul计算时，左、右矩阵的输入为half或bfloat16_t类型，输出为int8_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行量化操作，将最终结果量化为int8_t类型。

#### 函数原型

```
__aicore__ inline void SetQuantVector(const GlobalTensor<uint64_t>& quantTensor)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| quantTensor | 输入 | 量化或反量化运算时的参数向量。 |

#### 约束说明

需与SetDequantType保持一致。

本接口必须在Iterate或者IterateAll前调用。

#### 调用示例

```
GlobalTensor gmQuant;
...
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetQuantVector(gmQuant);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
