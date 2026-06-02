# SetTensorB

**页面ID:** atlasascendc_api_07_0632  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0632.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置矩阵乘的右矩阵B。

#### 函数原型

```
__aicore__ inline void SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false)
```

```
__aicore__ inline void SetTensorB(const LocalTensor<SrcBT>& rightMatrix, bool isTransposeB = false)
```

```
__aicore__ inline void SetTensorB(SrcBT bScalar)
```

Atlas 推理系列产品AI Core不支持SetTensorB(SrcBT bScalar)接口原型。

Atlas 200I/500 A2 推理产品，不支持SetTensorB(SrcBT bScalar)接口原型。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| SrcBT | 操作数的数据类型。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gm | 输入 | B矩阵。类型为GlobalTensor。SrcBT参数表示B矩阵的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float/bfloat16_t/int8_t/int4b_t Atlas 推理系列产品AI Core，支持的数据类型为：half/float/int8_t Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float/bfloat16_t/int8_t/int4b_t Atlas 200I/500 A2 推理产品，支持的数据类型为：half/float/bfloat16_t/int8_t |
| rightMatrix | 输入 | B矩阵。类型为LocalTensor，支持的TPosition为TSCM/VECOUT。SrcBT参数表示B矩阵的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float/bfloat16_t/int8_t/int4b_t Atlas 推理系列产品AI Core，支持的数据类型为：half/float/int8_t Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float/bfloat16_t/int8_t/int4b_t Atlas 200I/500 A2 推理产品，支持的数据类型为：half/float/bfloat16_t/int8_t 若设置TSCM首地址，默认矩阵可全载，已经位于TSCM，Iterate接口无需再进行GM->A1/B1搬运。 |
| bScalar | 输入 | B矩阵中设置的值。支持传入标量数据，标量数据会被扩展为一个形状为[1, K]的tensor参与矩阵乘计算，tensor的数值均为该标量值。例如，开发者可以通过将bScalar设置为1来实现矩阵A在K方向的reduce sum操作。SrcBT参数表示B矩阵的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float Atlas 推理系列产品AI Core不支持该参数。 Atlas 200I/500 A2 推理产品不支持该参数。 |
| isTransposeB | 输入 | B矩阵是否需要转置。 **注意：** - 若B矩阵MatmulType的ISTRANS参数设置为true，该参数可以为true也可以为false，即运行时可以转置和非转置交替使用；- 若B矩阵MatmulType的ISTRANS参数设置为false，该参数只能设置为false，若强行设置为true，精度会有异常；- 对于非half、非bfloat16_t输入类型的场景，为了确保Tiling侧与Kernel侧L1 Buffer空间计算大小保持一致及结果精度正确，该参数取值必须与Kernel侧定义B矩阵MatmulType的ISTRANS参数以及Tiling侧SetBType()接口的isTrans参数保持一致，即上述三个参数必须同时设置为true或同时设置为false。 |

#### 约束说明

传入的TensorB地址空间大小需要保证不小于singleK * singleN。

#### 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);    // 设置右矩阵B
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
