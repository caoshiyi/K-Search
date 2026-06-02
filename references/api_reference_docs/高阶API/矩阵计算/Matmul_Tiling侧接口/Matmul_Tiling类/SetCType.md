# SetCType

**页面ID:** atlasascendc_api_07_0676  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0676.html

---

#### 功能说明

设置C矩阵的位置，数据格式，数据类型等信息，这些信息需要和kernel侧的设置保持一致。

#### 函数原型

```
int32_t SetCType(TPosition pos, CubeFormat type, DataType dataType)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pos | 输入 | C矩阵所在的buffer位置，可设置为：TPosition::GM, TPosition::VECIN。 |
| type | 输入 | C矩阵的数据格式，可设置为：CubeFormat::ND，CubeFormat::NZ,  CubeFormat::ND_ALIGN。 |
| dataType | 输入 | C矩阵的数据类型，可设置为：DataType::DT_FLOAT/DataType::DT_FLOAT16/DataType::DT_BFLOAT16 /DataType::DT_INT8/DataType::DT_INT32。 |

#### 返回值说明

-1表示设置失败；0表示设置成功。

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16); 
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);  
// 设置C矩阵，buffer位置为GM，数据格式为ND，数据类型为float，默认不转置
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling.SetBiasType(AscendC::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetOrgShape(1024, 1024, 1024);   
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
