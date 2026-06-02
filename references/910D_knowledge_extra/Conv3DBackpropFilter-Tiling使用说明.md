# Conv3DBackpropFilter Tiling使用说明<a name="ZH-CN_TOPIC_0000002554423419"></a>

Ascend C提供一组Conv3DBackpropFilter Tiling API，方便用户获取Conv3DBackpropFilter Kernel计算时所需的Tiling参数。用户只需要传入Input/GradOutput/GradWeight的Position位置、Format格式和DType数据类型及相关参数等信息，调用API接口，即可获取[Init](Init-141.md)中TConv3DBpFilterTiling结构体中的相关参数。

Conv3DBackpropFilter Tiling API提供一个GetTiling接口获取Tiling参数，获取Tiling参数的流程如下：

1.  创建一个单核Tiling对象。
2.  设置Input、GradOutput、GradWeight的参数类型信息以及Shape信息，如果存在Padding、Stride参数，通过[SetPadding](SetPadding-156.md)、[SetStride](SetStride-157.md)接口设置。
3.  调用[GetTiling](GetTiling-149.md)接口，获取Tiling信息。

使用Conv3DBackpropFilter Tiling接口获取Tiling参数的样例如下：

```
#include "tiling/conv_backprop/conv3d_bp_filter_tiling.h"

optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);

conv3dBpDwTiling.SetWeightType(ConvCommonApi::TPosition::CO1,
                                   ConvCommonApi::ConvFormat::FRACTAL_Z_3D,
                                   ConvCommonApi::ConvDtype::FLOAT32);
conv3dBpDwTiling.SetInputType(ConvCommonApi::TPosition::GM,
                                 ConvCommonApi::ConvFormat::NDC1HWC0,
                                 ConvCommonApi::ConvDtype::FLOAT16);
conv3dBpDwTiling.SetGradOutputType(ConvCommonApi::TPosition::GM,
                                   ConvCommonApi::ConvFormat::NDC1HWC0,
                                   ConvCommonApi::ConvDtype::FLOAT16);
conv3dBpDwTiling.SetGradOutputShape(n, c, d, h, w);
conv3dBpDwTiling.SetInputShape(n, c, d, h, w);
conv3dBpDwTiling.SetWeightShape(cout, cin, d, h, w);
conv3dBpDwTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
conv3dBpDwTiling.SetStride(strideD, strideH, strideW);
conv3dBpDwTiling.SetDilation(dilationD, dilationH, dilationW);
int ret = conv3dBpDwTiling.GetTiling(tilingData);    // 如果 ret = -1, 获取tiling 结果失败
```

## 需要包含的头文件<a name="section1682364117469"></a>

```
#include "lib/conv_backprop/conv3d_bp_filter_tiling.h"
```

