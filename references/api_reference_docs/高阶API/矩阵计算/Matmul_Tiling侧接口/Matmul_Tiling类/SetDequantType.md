# SetDequantType

**页面ID:** atlasascendc_api_07_0698  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0698.html

---

#### 功能说明

该接口用于设置量化或反量化的模式。

Matmul反量化场景：在Matmul计算时，左、右矩阵的输入为int8_t或int4b_t类型，输出为half类型；或者左、右矩阵的输入为int8_t类型，输出为int8_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行反量化操作，将最终结果反量化为对应的half或int8_t类型。

Matmul量化场景：在Matmul计算时，左、右矩阵的输入为half或bfloat16_t类型，输出为int8_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行量化操作，将最终结果量化为int8_t类型。

量化或反量化时有两种模式： 一种是同一系数的量化/反量化模式，一种是向量的量化/反量化模式。

- 同一系数的量化或反量化模式：对输出矩阵的所有值采用同一系数进行量化或反量化。
- 向量的量化或反量化模式：提供一个参数向量，对输出矩阵的每一列都采用该向量中对应列的系数进行量化或反量化。

#### 函数原型

```
int32_t SetDequantType(DequantType dequantType)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| 设置量化或反量化时的模式。DequantType类型，该类型的定义如下。 ``` enum class DequantType {     SCALAR = 0,     TENSOR = 1, }; ```  参数的取值及含义如下： - SCALAR：表示同一系数的量化或反量化模式。- TENSOR：表示向量的量化或反量化模式。 |  |  |

#### 返回值说明

-1表示设置失败；0表示设置成功。

#### 约束说明

本接口支持的同一系数的量化/反量化模式、向量的量化/反量化模式分别与Kernel侧接口SetQuantScalar和SetQuantVector对应，本接口设置的量化/反量化模式必须与Kernel侧使用的接口保持一致。

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);   
tiling.SetShape(M, N, K);   
tiling.SetOrgShape(M, N, K);
tiling.EnableBias(true);
tiling.SetDequantType(DequantType::SCALAR);  // 设置同一系数的量化/反量化模式
// tiling.SetDequantType(DequantType::TENSOR);  // 设置向量的量化/反量化模式
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
