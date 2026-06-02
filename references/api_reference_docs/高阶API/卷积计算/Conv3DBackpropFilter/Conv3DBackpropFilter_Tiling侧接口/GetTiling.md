# GetTiling

**页面ID:** atlasascendc_api_07_0908  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0908.html

---

#### 功能说明

获取Tiling参数。

#### 函数原型

```
int64_t GetTiling(optiling::Conv3DBackpropFilterTilingData& tiling)
```

```
int64_t GetTiling(AscendC::tiling::Conv3DBackpropFilterTilingData& tiling)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tiling | 输出 | TConv3DBpFilterTiling的Tiling结构体，用于存储最终的Tiling结果。TConv3DBpFilterTiling结构介绍请参考TConv3DBpFilterTiling结构体。 |

#### 返回值说明

如果返回值不为-1，则代表Tiling计算成功，用户可以使用该Tiling结构的值。如果返回值为-1，则代表Tiling计算失败，该Tiling结果无法使用。

#### 约束说明

无

#### 调用示例

```
#include "tiling/conv_backprop/conv3d_bp_filter_tiling.h"

optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetWeightType(ConvCommonApi::TPosition::GM,
                                   ConvCommonApi::ConvFormat::FRACTAL_Z_3D,
                                   ConvCommonApi::ConvDtype::FLOAT32);
conv3dBpDwTiling.SetInputType(ConvCommonApi::TPosition::GM,
                                 ConvCommonApi::ConvFormat::NDC1HWC0,
                                 ConvCommonApi::ConvDtype::FLOAT16);
conv3dBpDwTiling.SetGradOutputType(ConvCommonApi::TPosition::GM,
                                   ConvCommonApi::ConvFormat::NDC1HWC0,
                                   ConvCommonApi::ConvDtype::FLOAT16);
conv3dBpDwTiling.SetInputShape(n, c, d, h, w);
conv3dBpDwTiling.SetGradOutputShape(n, c, d, h, w);
conv3dBpDwTiling.SetWeightShape(cout, cin, d, h, w);
conv3dBpDwTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
conv3dBpDwTiling.SetStride(strideD, strideH, strideW);
conv3dBpDwTiling.SetDilation(dilationD, dilationH, dilationW);
int ret = conv3dBpDwTiling.GetTiling(tilingData);    // 获取tiling参数
// 使用AscendC::tiling命名空间中的Tiling结构体获取tiling参数
Ascend C::tiling::Conv3DBackpropFilterTilingData tilingDataNotOp;
ret = conv3dBpDwTiling.GetTiling(tilingDataNotOp);
```
