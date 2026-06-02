# SetAType

**页面ID:** atlasascendc_api_07_0674  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0674.html

---

#### 功能说明

设置A矩阵的位置，数据格式，数据类型，是否转置等信息，这些信息需要和kernel侧的设置保持一致。

#### 函数原型

```
int32_t SetAType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pos | 输入 | A矩阵所在的buffer位置，可设置为：TPosition::GM, TPosition::VECOUT, TPosition::TSCM。 |
| type | 输入 | A矩阵的数据格式，可设置为：CubeFormat::ND，CubeFormat::NZ，CubeFormat::VECTOR |
| dataType | 输入 | A矩阵的数据类型，可设置为：DataType::DT_FLOAT/DataType::DT_FLOAT16/DataType::DT_BFLOAT16/DataType::DT_INT8/DataType::DT_INT4。 |
| isTrans | 输入 | A矩阵是否转置。 参数取值： - true：A矩阵转置；- false：A矩阵不转置。 Atlas 推理系列产品AI Core，A矩阵为DataType::DT_INT8数据类型时不支持转置，即不支持该参数设置为true。 |

#### 返回值说明

-1表示设置失败；0表示设置成功。

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
// 设置A矩阵，buffer位置为GM，数据格式为ND，数据类型为bfloat16，默认不转置
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);  
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetOrgShape(1024, 1024, 1024);  
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
