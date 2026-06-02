# SetAlignSplit

**页面ID:** atlasascendc_api_07_0697  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0697.html

---

#### 功能说明

多核切分时， 设置singleCoreM/singleCoreN/singleCoreK的对齐值。比如设置singleCoreM的对齐值为64（单位为元素），切分出的singleCoreM为64的倍数。

#### 函数原型

```
int32_t SetAlignSplit(int32_t alignM, int32_t alignN, int32_t alignK)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| alignM | 输入 | singleCoreM的对齐值。若传入-1或0，表示不设置指定的singleCoreM的对齐值，该值由Tiling函数自行计算。 |
| alignN | 输入 | singleCoreN的对齐值。若传入-1或0，表示不设置指定的singleCoreN的对齐值，该值由Tiling函数自行计算。 |
| alignK | 输入 | singleCoreK的对齐值。若传入-1或0，表示不设置指定的singleCoreK的对齐值，该值由Tiling函数自行计算。 |

#### 返回值说明

-1表示设置失败； 0表示设置成功。

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
tiling.SetAlignSplit(-1, 64, -1);  // 设置singleCoreM/singleCoreN/singleCoreK的对齐值
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
