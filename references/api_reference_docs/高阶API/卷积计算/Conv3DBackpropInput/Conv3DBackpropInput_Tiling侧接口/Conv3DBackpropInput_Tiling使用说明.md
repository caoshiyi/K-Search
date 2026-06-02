# Conv3DBackpropInput Tiling使用说明

**页面ID:** atlasascendc_api_07_0930  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0930.html

---

Ascend C提供一组Conv3DBackpropInput Tiling API，方便用户获取Conv3DBackpropInput Kernel计算时所需的Tiling参数。用户只需要传入Input/GradOutput/Weight的Position位置、Format格式和DType数据类型及相关参数等信息，调用API接口，即可获取Init中TConv3DBackpropInputTiling结构体中的相关参数。

Conv3DBackpropInput Tiling API提供一个GetTiling接口获取Tiling参数，获取Tiling参数的流程如下：

1. 创建一个单核Tiling对象。
2. 设置Input、GradOutput、Weight的参数类型信息以及Shape信息，如果存在Padding、Stride参数，通过SetPadding、SetStride接口设置。
3. 调用GetTiling接口，获取Tiling信息。

使用Conv3DBackpropInput Tiling接口获取Tiling参数的样例如下：

```
#include "tiling/conv_backprop/conv3d_bp_input_tiling.h"

optiling::Conv3DBackpropInputTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetWeightType(Convolution3DBackprop::TPosition::GM,
                                   Convolution3DBackprop::ConvFormat::FRACTAL_Z_3D,
                                   Convolution3DBackprop::ConvDtype::FLOAT32);
conv3DBpDxTiling.SetGradOutputType(Convolution3DBackprop::TPosition::GM,
                                   Convolution3DBackprop::ConvFormat::NDC1HWC0,
                                   Convolution3DBackprop::ConvDtype::FLOAT16);
conv3DBpDxTiling.SetInputType(Convolution3DBackprop::TPosition::CO1,
                                 Convolution3DBackprop::ConvFormat::NDC1HWC0,
                                 Convolution3DBackprop::ConvDtype::FLOAT16);
conv3DBpDxTiling.SetInputShape(orgN, orgCi, orgDi, orgHi, orgWi);
conv3DBpDxTiling.SetGradOutputShape(orgCo, orgDo, orgHo, orgWo);
conv3DBpDxTiling.SetWeightShape(orgKd, orgKh, orgKw);
conv3DBpDxTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
conv3DBpDxTiling.SetStride(strideD, strideH, strideW);
conv3DBpDxTiling.SetDilation(dilationD, dilationH, dilationW);
int ret = conv3DBpDxTiling.GetTiling(tilingData);    // if ret = -1, get tiling failed
```

#### 需要包含的头文件

```
#include "lib/conv_backprop/conv3d_bp_input_tiling.h"
```
