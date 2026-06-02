# SetQuantScalar

**页面ID:** atlasascendc_api_07_0649  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0649.html

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

本接口提供对输出矩阵的所有值采用同一系数进行量化或反量化的功能，即整个C矩阵对应一个量化参数，量化参数的shape为[1]。量化、反量化的详细内容请参考量化场景。

Matmul反量化场景：在Matmul计算时，左、右矩阵的输入为int8_t或int4b_t类型，输出为half类型；或者左、右矩阵的输入为int8_t类型，输出为int8_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行反量化操作，将最终结果反量化为对应的half或int8_t类型。

Matmul量化场景：在Matmul计算时，左、右矩阵的输入为half或bfloat16_t类型，输出为int8_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行量化操作，将最终结果量化为int8_t类型。

#### 函数原型

```
__aicore__ inline void SetQuantScalar(const uint64_t quantScalar)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| quantScalar | 输入 | 量化或反量化系数。 |

#### 约束说明

需与SetDequantType保持一致。

本接口必须在Iterate或者IterateAll前调用。

#### 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
float tmp = 0.1;  // 输出gm时会乘以0.1
uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp)); // 浮点值量化或反量化系数转换为uint64_t类型进行设置
mm.SetQuantScalar(ans);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
