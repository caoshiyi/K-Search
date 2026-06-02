# SetSingleRange

**页面ID:** atlasascendc_api_07_0694  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0694.html

---

#### 功能说明

设置singleCoreM/singleCoreN/singleCoreK的最大值与最小值。

#### 函数原型

```
int32_t SetSingleRange(int32_t maxM = -1, int32_t maxN = -1, int32_t maxK = -1, int32_t minM = -1, int32_t minN = -1, int32_t minK = -1)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| maxM | 输入 | 设置最大的singleCoreM值，默认值为-1，表示不设置指定的singleCoreM最大值，该值由Tiling函数自行计算。 |
| maxN | 输入 | 设置最大的singleCoreN值，默认值为-1，表示不设置指定的singleCoreN最大值，该值由Tiling函数自行计算。 |
| maxK | 输入 | 设置最大的singleCoreK值，默认值为-1，表示不设置指定的singleCoreK最大值，该值由Tiling函数自行计算。 |
| minM | 输入 | 设置最小的singleCoreM值，默认值为-1，表示不设置指定的singleCoreM最小值，该值由Tiling函数自行计算。 |
| minN | 输入 | 设置最小的singleCoreN值，默认值为-1，表示不设置指定的singleCoreN最小值，该值由Tiling函数自行计算。 |
| minK | 输入 | 设置最小的singleCoreK值，默认值为-1，表示不设置指定的singleCoreK最小值，该值由Tiling函数自行计算。 |

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
tiling.SetShape(1024, 1024, 1024);   
tiling.SetSingleRange(1024, 1024, 1024, 1024, 1024, 1024);  // 设置singleCoreM/singleCoreN/singleCoreK的最大值与最小值
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
