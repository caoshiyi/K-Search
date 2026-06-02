# Conv3D Tiling使用说明

**页面ID:** atlasascendc_api_07_10080  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10080.html

---

Ascend C提供一组Conv3D Tiling API，方便用户获取Conv3D正向算子Kernel计算时所需的Tiling参数。用户只需要传入Input/Weight/Bias/Output的Position位置、Format格式和DType数据类型及相关参数等信息，调用API接口，即可获取Init中TConv3DApiTiling结构体中的相关参数。

Conv3D Tiling API提供Conv3D单核Tiling接口，用于Conv3D单核计算场景，获取Tiling参数的流程如下：

1. 创建一个单核Tiling对象。
2. 设置Input、Weight、Bias、Output的参数类型信息以及Shape信息，如果存在Padding、Stride、Dilation参数，通过SetPadding、SetStride、SetDilation接口进行相关设置。
3. 调用GetTiling接口，获取Tiling信息。

使用Conv3D Tiling接口获取Tiling参数的样例如下：

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
// 设置输入输出原始规格、单核规格、参数等
conv3dApiTiling.SetGroups(groups);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
conv3dApiTiling.SetOrgInputShape(cin, di, hi, wi);
conv3dApiTiling.SetPadding(padh, padt, padu, padd, padl, padr);
conv3dApiTiling.SetDilation(dilationH, dilationW, dilationD);
conv3dApiTiling.SetStride(strideH, strideW, strideD);
conv3dApiTiling.SetSingleWeightShape(cin, kd, kh, kw);
conv3dApiTiling.SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreMo);
// 设置输入输出type
conv3dApiTiling.SetInputType(TPosition::GM, inputFormat, inputDtype);
conv3dApiTiling.SetWeightType(TPosition::GM, weightFormat, weightDtype);
conv3dApiTiling.SetOutputType(TPosition::CO1, outputFormat, outputDtype);
if (biasFlag) {
   conv3dApiTiling.SetBiasType(TPosition::GM, biasFormat, biasDtype);
}
// 调用GetTiling接口获取核内切分策略，如果返回-1代表获取tiling失败
if (conv3dApiTiling.GetTiling(tilingData.conv3dApiTilingData) == -1) {
   return false;
}
```

#### 需要包含的头文件

```
#include "lib/conv/conv3d/conv3d_tiling.h"
```
