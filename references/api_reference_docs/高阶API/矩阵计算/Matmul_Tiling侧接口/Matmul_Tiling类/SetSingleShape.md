# SetSingleShape

**页面ID:** atlasascendc_api_07_0678  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0678.html

---

#### 功能说明

设置Matmul单核计算的形状singleMIn，singleNIn，singleKIn，单位为元素。

#### 函数原型

```
int32_t SetSingleShape(int32_t singleMIn = -1, int32_t singleNIn = -1, int32_t singleKIn = -1)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleMIn | 输入 | 设置的singleMIn大小，单位为元素，默认值为-1。-1表示不设置指定的singleMIn，该值由tiling函数自行计算。 |
| singleNIn | 输入 | 设置的singleNIn大小，单位为元素，默认值为-1。-1表示不设置指定的singleNIn，该值由tiling函数自行计算。 |
| singleKIn | 输入 | 设置的singleKIn大小，单位为元素，默认值为-1。-1表示不设置指定的singleKIn，该值由tiling函数自行计算。 |

#### 返回值说明

-1表示设置失败；0表示设置成功。

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
tiling.SetDim(1);
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);  // 设置Matmul单次计算的形状 
tiling.SetSingleShape(1024, 1024, 1024);  // 设置单核计算的形状
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
